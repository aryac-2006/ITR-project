"""Reusable, leakage-safe preprocessing pipelines.

Every learned statistic (imputation values, scaling parameters, outlier bounds,
encoder categories, selected features) lives *inside* an sklearn Pipeline, so it is
fitted on training data only and re-applied unchanged to test / new data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (SelectKBest, VarianceThreshold, f_classif,
                                       f_regression, mutual_info_classif,
                                       mutual_info_regression)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                                   RobustScaler, StandardScaler)

from utils.data import split_columns

DEFAULT_PREP = {
    "num_impute": "median",        # mean | median | most_frequent | knn
    "cat_impute": "most_frequent",  # most_frequent | missing_label
    "scaler": "standard",          # standard | minmax | robust | none
    "encoding": "onehot",          # onehot | ordinal
    "outliers": "none",            # none | clip (IQR winsorising)
    "iqr_factor": 1.5,
    "selector": "none",            # none | kbest | variance
    "k_features": 10,
}


class IQRClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Winsorise numeric columns to [Q1 - f*IQR, Q3 + f*IQR] using bounds learned on fit."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_, self.upper_ = q1 - self.factor * iqr, q3 + self.factor * iqr
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


class SafeSelectKBest(BaseEstimator, TransformerMixin):
    """SelectKBest that caps k at the number of available features."""

    def __init__(self, k: int = 10, task: str = "classification"):
        self.k, self.task = k, task

    def fit(self, X, y):
        score = f_classif if self.task == "classification" else f_regression
        self.selector_ = SelectKBest(score, k=min(self.k, X.shape[1]))
        self.selector_.fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return self.selector_.transform(X)

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features if input_features is not None
                           else [f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return names[self.selector_.get_support()]


def make_preprocessor(numeric_cols, categorical_cols, cfg: dict | None = None) -> ColumnTransformer:
    """Column-wise preprocessing: impute -> (clip outliers) -> scale for numbers,
    impute -> encode for categories."""
    cfg = {**DEFAULT_PREP, **(cfg or {})}
    transformers = []
    if numeric_cols:
        imp = (KNNImputer(n_neighbors=5) if cfg["num_impute"] == "knn"
               else SimpleImputer(strategy=cfg["num_impute"]))
        steps = [("impute", imp)]
        if cfg["outliers"] == "clip":
            steps.append(("clip", IQRClipper(cfg["iqr_factor"])))
        scalers = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
        if cfg["scaler"] in scalers:
            steps.append(("scale", scalers[cfg["scaler"]]()))
        transformers.append(("num", Pipeline(steps), list(numeric_cols)))
    if categorical_cols:
        if cfg["cat_impute"] == "missing_label":
            cat_imp = SimpleImputer(strategy="constant", fill_value="missing")
        else:
            cat_imp = SimpleImputer(strategy="most_frequent")
        enc = (OneHotEncoder(handle_unknown="ignore", sparse_output=False)
               if cfg["encoding"] == "onehot"
               else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        transformers.append(("cat", Pipeline([("impute", cat_imp), ("encode", enc)]),
                             list(categorical_cols)))
    if not transformers:
        raise ValueError("No usable feature columns were selected.")
    return ColumnTransformer(transformers, remainder="drop")


def build_pipeline(estimator, numeric_cols, categorical_cols, cfg=None, task="classification") -> Pipeline:
    """preprocess -> (optional feature selection) -> model, as ONE estimator."""
    cfg = {**DEFAULT_PREP, **(cfg or {})}
    steps = [("prep", make_preprocessor(numeric_cols, categorical_cols, cfg))]
    if cfg["selector"] == "kbest":
        steps.append(("select", SafeSelectKBest(int(cfg["k_features"]), task)))
    elif cfg["selector"] == "variance":
        steps.append(("select", VarianceThreshold(0.0)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def prepare_xy(df: pd.DataFrame, features: list[str], target: str):
    """Select columns, drop rows with a missing target and normalise dtypes."""
    data = df[list(features) + [target]].dropna(subset=[target]).reset_index(drop=True)
    X, y = data[list(features)].copy(), data[target]
    num, cat, _ = split_columns(X)
    X[num] = X[num].replace([np.inf, -np.inf], np.nan)
    for c in cat:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X, y


def outlier_report(df: pd.DataFrame, cols, factor: float = 1.5) -> pd.DataFrame:
    """IQR and z-score outlier counts per numeric column (analysis only; nothing is removed)."""
    rows = []
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        n_iqr = int(((s < lo) | (s > hi)).sum())
        std = s.std()
        n_z = int((np.abs((s - s.mean()) / std) > 3).sum()) if std and std > 0 else 0
        rows.append({"column": c, "lower_bound": lo, "upper_bound": hi,
                     "iqr_outliers": n_iqr, "iqr_outlier_%": round(100 * n_iqr / len(s), 2),
                     "zscore>3": n_z, "skewness": round(float(s.skew()), 2)})
    return pd.DataFrame(rows)


def rank_features(df, features, target, task, seed=42) -> pd.DataFrame:
    """Exploratory relevance ranking via mutual information (uses all rows; for guidance only,
    the training pipeline refits any selection on the training split)."""
    X, y = prepare_xy(df, features, target)
    num, cat, _ = split_columns(X)
    Xn = pd.DataFrame(index=X.index)
    for c in num:
        Xn[c] = X[c].fillna(X[c].median())
    for c in cat:
        Xn[c] = X[c].fillna("missing").astype("category").cat.codes
    discrete = [c in cat for c in Xn.columns]
    if task == "classification":
        scores = mutual_info_classif(Xn, y.astype(str), discrete_features=discrete, random_state=seed)
    else:
        scores = mutual_info_regression(Xn, pd.to_numeric(y, errors="coerce").fillna(0),
                                        discrete_features=discrete, random_state=seed)
    return (pd.DataFrame({"feature": Xn.columns, "mutual_information": scores})
            .sort_values("mutual_information", ascending=False).reset_index(drop=True))
