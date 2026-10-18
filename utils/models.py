"""Supervised learning: model registry, validation, training, tuning, evaluation, persistence."""
from __future__ import annotations

import io
import time
import uuid

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils.data import HIGH_CARDINALITY, split_columns
from utils.preprocessing import build_pipeline, prepare_xy

SVM_MAX_TRAIN = 10_000  # kernel SVMs scale ~O(n^2); cap training rows for cloud hosting
HEAVY_MODELS = {"Support Vector Machine", "Support Vector Regression"}

# name -> (factory(seed), hyper-parameter grid used when tuning)
CLASSIFIERS = {
    "Logistic Regression": (lambda s: LogisticRegression(max_iter=2000, random_state=s),
                            {"C": [0.01, 0.1, 1, 10, 100]}),
    "Decision Tree": (lambda s: DecisionTreeClassifier(random_state=s),
                      {"max_depth": [3, 5, 8, 12, None], "min_samples_split": [2, 5, 10]}),
    "Random Forest": (lambda s: RandomForestClassifier(n_estimators=150, random_state=s, n_jobs=-1),
                      {"n_estimators": [100, 200], "max_depth": [None, 8, 16],
                       "min_samples_leaf": [1, 2, 4]}),
    "K-Nearest Neighbors": (lambda s: KNeighborsClassifier(),
                            {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"]}),
    "Naive Bayes": (lambda s: GaussianNB(), {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]}),
    "Support Vector Machine": (lambda s: SVC(probability=True, random_state=s),
                               {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"],
                                "gamma": ["scale", "auto"]}),
}
REGRESSORS = {
    "Linear Regression": (lambda s: LinearRegression(), {"fit_intercept": [True, False]}),
    "Ridge Regression": (lambda s: Ridge(random_state=s), {"alpha": [0.01, 0.1, 1, 10, 100]}),
    "Decision Tree Regressor": (lambda s: DecisionTreeRegressor(random_state=s),
                                {"max_depth": [3, 5, 8, 12, None], "min_samples_split": [2, 5, 10]}),
    "Random Forest Regressor": (lambda s: RandomForestRegressor(n_estimators=150, random_state=s, n_jobs=-1),
                                {"n_estimators": [100, 200], "max_depth": [None, 8, 16],
                                 "min_samples_leaf": [1, 2, 4]}),
    "Support Vector Regression": (lambda s: SVR(),
                                  {"C": [0.1, 1, 10, 100], "kernel": ["rbf", "linear"],
                                   "epsilon": [0.01, 0.1, 0.5]}),
}


def registry(task: str) -> dict:
    return CLASSIFIERS if task == "classification" else REGRESSORS


# --------------------------------------------------------------------------- validation
def validate_target(df: pd.DataFrame, target: str, task: str) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) so incompatible algorithms are never applied silently."""
    errors, warnings = [], []
    s = df[target].dropna()
    n_unique = s.nunique()
    if len(s) < 30:
        errors.append(f"Only {len(s)} rows have a value for '{target}'; at least 30 are required.")
        return errors, warnings
    if n_unique < 2:
        errors.append(f"'{target}' has a single value, so there is nothing to learn.")
        return errors, warnings
    if task == "classification":
        if n_unique > 30:
            errors.append(f"'{target}' has {n_unique} distinct values - too many classes. "
                          "It looks continuous or like an ID; use the Regression page instead.")
        elif pd.api.types.is_float_dtype(s) and n_unique > 15:
            errors.append(f"'{target}' is a decimal column with {n_unique} distinct values. "
                          "Use the Regression page, or bin it into classes first.")
        else:
            counts = s.astype(str).value_counts()
            if counts.min() < 2:
                errors.append(f"Class '{counts.idxmin()}' has only 1 row; every class needs at least 2 "
                              "rows for a stratified split.")
            elif counts.min() / counts.max() < 0.1:
                warnings.append("Classes are strongly imbalanced; accuracy can be misleading - "
                                "look at F1, recall and the confusion matrix.")
    else:
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            errors.append(f"'{target}' is not numeric, so regression cannot be used. "
                          "Use the Classification page instead.")
        elif n_unique <= 10:
            warnings.append(f"'{target}' has only {n_unique} distinct values - this may be a "
                            "classification problem.")
    return errors, warnings


def usable_features(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    """Drop features that cannot be modelled (dates, very high-cardinality text)."""
    keep, notes = [], []
    _, cat, other = split_columns(df, features)
    for c in features:
        if c in other:
            notes.append(f"'{c}' is a date/time column and was excluded.")
        elif c in cat and df[c].nunique() > HIGH_CARDINALITY:
            notes.append(f"'{c}' has {df[c].nunique()} distinct categories (likely free text/ID) and was excluded.")
        else:
            keep.append(c)
    return keep, notes


# --------------------------------------------------------------------------- training
def _adapt_grid(grid: dict, n_train: int, cv: int) -> dict:
    grid = {k: list(v) for k, v in grid.items()}
    if "n_neighbors" in grid:
        cap = max(1, int(n_train * (1 - 1 / max(cv, 2))) - 1)
        grid["n_neighbors"] = [k for k in grid["n_neighbors"] if k <= cap] or [min(3, cap)]
    return grid


def build_schema(X: pd.DataFrame) -> dict:
    """Describe each input feature (used to build the prediction form and validate new data)."""
    num, cat, _ = split_columns(X)
    schema = {}
    for c in X.columns:
        if c in num:
            s = X[c].dropna()
            schema[c] = {"kind": "numeric", "min": float(s.min()), "max": float(s.max()),
                         "median": float(s.median())}
        else:
            vc = X[c].dropna().astype(str).value_counts()
            schema[c] = {"kind": "categorical", "choices": list(vc.index[:100]),
                         "default": str(vc.index[0]) if len(vc) else ""}
    return schema


def train_model(df, features, target, task, model_name, cfg=None, test_size=0.2, cv=5,
                tune="none", n_iter=10, seed=42, dataset_name="") -> dict:
    """Train + evaluate one model. Returns {"record": summary, "artifact": fitted objects}."""
    factory, grid = registry(task)[model_name]
    X, y = prepare_xy(df, features, target)
    classes = None
    if task == "classification":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)))
        classes = [str(c) for c in le.classes_]
    else:
        y = pd.to_numeric(y, errors="coerce").astype(float)
        ok = y.notna()
        X, y = X[ok].reset_index(drop=True), y[ok].reset_index(drop=True)

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed,
            stratify=y if task == "classification" else None)
    except ValueError as exc:
        raise ValueError(f"Could not split the data: {exc}") from exc

    note = ""
    if model_name in HEAVY_MODELS and len(X_tr) > SVM_MAX_TRAIN:
        if task == "classification":
            X_tr, _, y_tr, _ = train_test_split(X_tr, y_tr, train_size=SVM_MAX_TRAIN,
                                                random_state=seed, stratify=y_tr)
        else:
            X_tr = X_tr.sample(SVM_MAX_TRAIN, random_state=seed)
            y_tr = y_tr.loc[X_tr.index]
        X_tr, y_tr = X_tr.reset_index(drop=True), y_tr.reset_index(drop=True)
        note = f"Training rows capped at {SVM_MAX_TRAIN:,} for speed."

    cv_eff = min(cv, int(np.bincount(y_tr).min())) if task == "classification" else cv
    cv_eff = max(min(cv_eff, len(X_tr) // 2), 2)
    splitter = (StratifiedKFold(cv_eff, shuffle=True, random_state=seed)
                if task == "classification" else KFold(cv_eff, shuffle=True, random_state=seed))
    scoring = "f1_weighted" if task == "classification" else "r2"

    num, cat, _ = split_columns(X_tr)
    pipe = build_pipeline(factory(seed), num, cat, cfg, task)

    t0 = time.time()
    best_params = {}
    if tune in ("grid", "random"):
        g = {f"model__{k}": v for k, v in _adapt_grid(grid, len(X_tr), cv_eff).items()}
        if tune == "grid":
            search = GridSearchCV(pipe, g, cv=splitter, scoring=scoring, n_jobs=1)
        else:
            size = int(np.prod([len(v) for v in g.values()]))
            search = RandomizedSearchCV(pipe, g, n_iter=min(n_iter, size), cv=splitter,
                                        scoring=scoring, n_jobs=1, random_state=seed)
        search.fit(X_tr, y_tr)
        fitted = search.best_estimator_
        cv_mean = float(search.best_score_)
        cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
        best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    else:
        scores = cross_val_score(pipe, X_tr, y_tr, cv=splitter, scoring=scoring, n_jobs=1,
                                 error_score="raise")
        cv_mean, cv_std = float(scores.mean()), float(scores.std())
        fitted = pipe.fit(X_tr, y_tr)
    train_time = time.time() - t0

    y_pred = fitted.predict(X_te)
    proba = None
    if task == "classification":
        avg = "binary" if len(classes) == 2 else "weighted"
        metrics = {
            "accuracy": accuracy_score(y_te, y_pred),
            "precision": precision_score(y_te, y_pred, average=avg, zero_division=0),
            "recall": recall_score(y_te, y_pred, average=avg, zero_division=0),
            "f1": f1_score(y_te, y_pred, average=avg, zero_division=0),
            "roc_auc": None,
        }
        if hasattr(fitted, "predict_proba"):
            proba = fitted.predict_proba(X_te)
            try:
                if len(classes) == 2:
                    metrics["roc_auc"] = roc_auc_score(y_te, proba[:, 1])
                elif len(np.unique(y_te)) == len(classes):
                    metrics["roc_auc"] = roc_auc_score(y_te, proba, multi_class="ovr", average="macro")
            except ValueError:
                pass
        train_score = accuracy_score(y_tr, fitted.predict(X_tr))
    else:
        metrics = {"mae": mean_absolute_error(y_te, y_pred),
                   "rmse": float(np.sqrt(mean_squared_error(y_te, y_pred))),
                   "r2": r2_score(y_te, y_pred)}
        train_score = r2_score(y_tr, fitted.predict(X_tr))

    schema = build_schema(X_tr)
    record = {
        "id": uuid.uuid4().hex[:8], "task": task, "model": model_name, "dataset": dataset_name,
        "target": target, "features": list(X.columns), "tuned": tune != "none",
        "best_params": best_params, "cv_metric": scoring, "cv_mean": cv_mean, "cv_std": cv_std,
        "cv_folds": cv_eff, "train_score": float(train_score),
        "train_seconds": round(train_time, 2), "n_train": len(X_tr), "n_test": len(X_te),
        "classes": classes, "note": note, "metrics": {k: (None if v is None else float(v))
                                                       for k, v in metrics.items()},
    }
    artifact = {"pipeline": fitted, "X_test": X_te, "y_test": np.asarray(y_te),
                "y_pred": np.asarray(y_pred), "proba": proba, "schema": schema,
                "cm": (confusion_matrix(y_te, y_pred, labels=range(len(classes)))
                       if classes else None)}
    return {"record": record, "artifact": artifact}


# --------------------------------------------------------------------------- comparison
def leaderboard(records: list[dict], task: str) -> pd.DataFrame:
    """Flatten run records into a comparison table."""
    rows = []
    for r in records:
        row = {"run": r["id"], "model": r["model"], "tuned": r["tuned"]}
        row.update({k: v for k, v in r["metrics"].items()})
        row.update({"cv_mean": r["cv_mean"], "cv_std": r["cv_std"],
                    "train_score": r["train_score"], "train_seconds": r["train_seconds"]})
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    key, asc = ("f1", False) if task == "classification" else ("rmse", True)
    df = df.sort_values(key, ascending=asc).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# --------------------------------------------------------------------------- persistence / inference
def make_bundle(record: dict, artifact: dict) -> dict:
    """Everything needed to predict on new data: the FULL fitted pipeline + metadata."""
    return {"pipeline": artifact["pipeline"], "features": record["features"],
            "target": record["target"], "task": record["task"], "classes": record["classes"],
            "schema": artifact["schema"], "model": record["model"], "metrics": record["metrics"],
            "versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__}}


def bundle_to_bytes(bundle: dict) -> bytes:
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    return buf.getvalue()


def predict_dataframe(bundle: dict, new_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the saved pipeline to new rows (same preprocessing as training)."""
    feats, schema = bundle["features"], bundle["schema"]
    missing = [c for c in feats if c not in new_df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")
    X = new_df[feats].copy()
    for c in feats:
        if schema[c]["kind"] == "numeric":
            X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    pred = bundle["pipeline"].predict(X)
    out = new_df.reset_index(drop=True).copy()
    if bundle["task"] == "classification":
        classes = np.asarray(bundle["classes"])
        out["prediction"] = classes[np.asarray(pred).astype(int)]
        if hasattr(bundle["pipeline"], "predict_proba"):
            proba = bundle["pipeline"].predict_proba(X)
            for i, c in enumerate(classes):
                out[f"P({c})"] = proba[:, i].round(4)
    else:
        out["prediction"] = np.asarray(pred, dtype=float)
    return out


def out_of_range_count(bundle: dict, new_df: pd.DataFrame) -> int:
    """Count numeric cells outside the range seen in training (extrapolation warning)."""
    n = 0
    for c, meta in bundle["schema"].items():
        if meta["kind"] == "numeric" and c in new_df.columns:
            v = pd.to_numeric(new_df[c], errors="coerce")
            n += int(((v < meta["min"]) | (v > meta["max"])).sum())
    return n
