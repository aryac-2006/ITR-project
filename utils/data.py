"""Dataset loading, column typing and data-quality helpers (no Streamlit dependency)."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

MAX_ROWS = 50_000  # keeps computation reasonable on free cloud hosting
HIGH_CARDINALITY = 50  # categorical columns above this are not one-hot encoded

SAMPLE_DATASETS = {
    "Iris (classification / clustering)": "iris",
    "Wine (classification / clustering)": "wine",
    "Breast Cancer (classification)": "breast_cancer",
    "Diabetes (regression)": "diabetes",
    "Synthetic customers (messy: missing values, categoricals, duplicates)": "customers",
}


def make_demo_customers(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic customer table with realistic data-quality problems.

    Contains both a classification target (churn) and a regression target
    (monthly_charges) so every module can be demonstrated on one dataset.
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n).astype(float)
    income = rng.lognormal(10.5, 0.5, n).round(0)
    income[rng.choice(n, 8, replace=False)] *= 12  # a few extreme outliers
    tenure = rng.integers(1, 72, n).astype(float)
    plan = rng.choice(["Basic", "Standard", "Premium"], n, p=[0.45, 0.35, 0.2])
    region = rng.choice(["North", "South", "East", "West"], n)
    calls = rng.poisson(2.5, n).astype(float)
    plan_cost = pd.Series(plan).map({"Basic": 20, "Standard": 45, "Premium": 80}).to_numpy()
    monthly = (plan_cost + 0.0002 * income + 0.3 * tenure + 2.0 * calls
               + rng.normal(0, 5, n)).round(2)
    logit = -1.0 + 0.45 * calls - 0.03 * tenure + 0.6 * (plan == "Basic") - 0.00001 * income
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"age": age, "income": income, "tenure_months": tenure,
                       "plan": plan, "region": region, "support_calls": calls,
                       "monthly_charges": monthly, "churn": churn})
    for col, frac in (("age", 0.05), ("income", 0.06), ("region", 0.04)):
        df.loc[rng.choice(n, int(n * frac), replace=False), col] = np.nan
    return pd.concat([df, df.sample(20, random_state=seed)], ignore_index=True)


def load_sample(key: str) -> pd.DataFrame:
    """Load one of the bundled sample datasets (generated or shipped with scikit-learn)."""
    if key == "customers":
        return make_demo_customers()
    from sklearn import datasets

    bunch = getattr(datasets, f"load_{key}")(as_frame=True)
    df = bunch.frame.copy()
    target_names = {"iris": "species", "wine": "wine_class",
                    "breast_cancer": "diagnosis", "diabetes": "disease_progression"}
    df = df.rename(columns={"target": target_names[key]})
    if key != "diabetes":
        df[target_names[key]] = df[target_names[key]].map(dict(enumerate(bunch.target_names)))
    return df


def load_csv(file) -> pd.DataFrame:
    """Read an uploaded CSV with delimiter/encoding detection and basic validation."""
    raw = file.getvalue() if hasattr(file, "getvalue") else file.read()
    if not raw:
        raise ValueError("The file is empty.")
    df, last_error = None, None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
            break
        except UnicodeDecodeError as exc:
            last_error = exc
        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
            raise ValueError(f"Could not parse the file as CSV: {exc}") from exc
    if df is None:
        raise ValueError(f"Could not decode the file: {last_error}")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if df.shape[0] < 10 or df.shape[1] < 2:
        raise ValueError("The dataset needs at least 10 rows and 2 columns.")
    return df


def limit_rows(df: pd.DataFrame, max_rows: int = MAX_ROWS, seed: int = 42):
    """Randomly down-sample very large datasets. Returns (df, was_sampled)."""
    if len(df) <= max_rows:
        return df, False
    return df.sample(max_rows, random_state=seed).reset_index(drop=True), True


def split_columns(df: pd.DataFrame, cols=None):
    """Split columns into (numeric, categorical, other). Booleans count as categorical."""
    cols = list(df.columns) if cols is None else list(cols)
    num, cat, other = [], [], []
    for c in cols:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            cat.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            num.append(c)
        elif pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
            other.append(c)
        else:
            cat.append(c)
    return num, cat, other


def suggest_drop_columns(df: pd.DataFrame) -> dict:
    """Columns that are usually useless as features: constants and ID-like columns."""
    out = {}
    n = len(df)
    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        if nun <= 1:
            out[c] = "constant column"
        elif not pd.api.types.is_float_dtype(df[c]) and n > 20 and nun / max(df[c].notna().sum(), 1) > 0.98:
            out[c] = "looks like an identifier (almost every value is unique)"
    return out


def column_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column summary: dtype, missing, unique, example value."""
    rows = []
    for c in df.columns:
        s = df[c]
        non_null = s.dropna()
        rows.append({
            "column": c, "dtype": str(s.dtype), "non_null": int(s.notna().sum()),
            "missing": int(s.isna().sum()), "missing_%": round(100 * s.isna().mean(), 2),
            "unique": int(s.nunique(dropna=True)),
            "example": str(non_null.iloc[0]) if len(non_null) else "",
        })
    return pd.DataFrame(rows)


def quality_summary(df: pd.DataFrame) -> dict:
    """Headline data-quality numbers plus a simple 0-100 heuristic score."""
    n_rows, n_cols = df.shape
    total = max(n_rows * n_cols, 1)
    missing = int(df.isna().sum().sum())
    dupes = int(df.duplicated().sum())
    num, cat, other = split_columns(df)
    constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    high_card = [c for c in cat if df[c].nunique() > HIGH_CARDINALITY]
    missing_pct, dup_pct = 100 * missing / total, 100 * dupes / max(n_rows, 1)
    score = (100 - min(40, missing_pct * 2) - min(20, dup_pct * 2)
             - 5 * min(4, len(constant)) - 5 * min(4, len(high_card)))
    return {
        "rows": n_rows, "columns": n_cols, "numeric": len(num), "categorical": len(cat),
        "other": len(other), "missing_cells": missing, "missing_pct": round(missing_pct, 2),
        "duplicates": dupes, "duplicate_pct": round(dup_pct, 2),
        "constant_columns": constant, "high_cardinality_columns": high_card,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "quality_score": round(max(score, 0), 1),
    }


def clean_dataframe(df: pd.DataFrame, drop_duplicates=True, drop_columns=(),
                    max_missing_pct: float = 100.0) -> tuple[pd.DataFrame, list[str]]:
    """Row/column-level cleaning that is safe to do before splitting (no learned statistics).

    Imputation, encoding and scaling are NOT done here; they live in the model pipeline
    so they are fitted on training data only (prevents data leakage).
    """
    log, out = [], df.copy()
    if drop_columns:
        out = out.drop(columns=list(drop_columns))
        log.append(f"Dropped columns: {', '.join(map(str, drop_columns))}")
    if max_missing_pct < 100:
        bad = [c for c in out.columns if 100 * out[c].isna().mean() > max_missing_pct]
        if bad:
            out = out.drop(columns=bad)
            log.append(f"Dropped columns with >{max_missing_pct:.0f}% missing: {', '.join(map(str, bad))}")
    if drop_duplicates:
        before = len(out)
        out = out.drop_duplicates()
        if before - len(out):
            log.append(f"Removed {before - len(out)} duplicate rows")
    return out.reset_index(drop=True), log or ["No cleaning changes applied"]
