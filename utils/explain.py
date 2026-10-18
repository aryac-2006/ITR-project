"""Model explanation helpers: feature importance, permutation importance, optional SHAP."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

TREE_MODELS = ("RandomForest", "DecisionTree")


def shap_available() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False


def transformed_names(pipe) -> list[str]:
    names = pipe[:-1].get_feature_names_out()
    return [str(n).split("__", 1)[-1] for n in names]


def feature_importance(pipe, X_test, y_test, task, method="auto", n_repeats=5, seed=42):
    """Return (DataFrame[feature, importance], description).

    auto: impurity importances (trees), |coefficients| (linear), permutation otherwise.
    permutation: model-agnostic, computed on the held-out test set with original columns."""
    model = pipe[-1]
    if method == "auto":
        try:
            names = transformed_names(pipe)
            if hasattr(model, "feature_importances_"):
                imp, how = np.asarray(model.feature_importances_), "Impurity-based importance (training data)"
            elif hasattr(model, "coef_") and getattr(model, "kernel", "linear") == "linear":
                imp = np.abs(np.asarray(model.coef_))
                imp = imp.mean(axis=0) if imp.ndim > 1 else imp
                how = "Absolute coefficient on scaled/encoded features"
            else:
                imp = None
            if imp is not None and len(imp) == len(names):
                return (pd.DataFrame({"feature": names, "importance": imp.ravel()})
                        .sort_values("importance", ascending=False).reset_index(drop=True), how)
        except Exception:
            pass
    Xs, ys = X_test, y_test
    if len(Xs) > 1000:
        idx = np.random.default_rng(seed).choice(len(Xs), 1000, replace=False)
        Xs, ys = Xs.iloc[idx], np.asarray(ys)[idx]
    scoring = "f1_weighted" if task == "classification" else "r2"
    res = permutation_importance(pipe, Xs, ys, n_repeats=n_repeats, random_state=seed,
                                 scoring=scoring, n_jobs=1)
    df = pd.DataFrame({"feature": Xs.columns, "importance": res.importances_mean,
                       "std": res.importances_std})
    return (df.sort_values("importance", ascending=False).reset_index(drop=True),
            f"Permutation importance on the test set (drop in {scoring})")


def shap_importance(pipe, X_test, max_rows=100, seed=42):
    """Mean |SHAP| per transformed feature plus the arrays needed for a summary plot.

    Returns (DataFrame, shap_values_2d, X_dense, names). Raises on failure so the UI can explain."""
    import shap

    X_s = X_test.sample(min(max_rows, len(X_test)), random_state=seed)
    Xt = np.asarray(pipe[:-1].transform(X_s), dtype=float)
    names = transformed_names(pipe)
    model = pipe[-1]
    kind = type(model).__name__
    if any(kind.startswith(t) for t in TREE_MODELS):
        values = shap.TreeExplainer(model).shap_values(Xt)
    elif kind in ("LogisticRegression", "LinearRegression", "Ridge"):
        values = shap.LinearExplainer(model, Xt).shap_values(Xt)
    else:  # model-agnostic fallback (slow): small background and sample
        f = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        bg = shap.sample(Xt, min(20, len(Xt)), random_state=seed)
        values = shap.KernelExplainer(f, bg).shap_values(Xt[:40], nsamples=100)
        Xt = Xt[:40]
    if isinstance(values, list):                      # old API: list per class
        values = np.stack(values, axis=-1)
    values = np.asarray(values)
    if values.ndim == 3:                              # (rows, features, classes)
        mean_abs = np.abs(values).mean(axis=(0, 2))
        plot_vals = values[:, :, -1] if values.shape[2] == 2 else np.abs(values).mean(axis=2)
    else:
        mean_abs, plot_vals = np.abs(values).mean(axis=0), values
    df = (pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
          .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    return df, plot_vals, Xt, names
