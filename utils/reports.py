"""Plain-text analysis summary assembled from what the user actually did in the session."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from utils.data import quality_summary
from utils.models import leaderboard


def _fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


def build_summary(state) -> str:
    lines = [f"# IntelliData AI - Analysis Summary", f"_Generated {datetime.now():%Y-%m-%d %H:%M}_", ""]
    df = state.get("df")
    if df is None:
        return "\n".join(lines + ["No dataset has been loaded yet."])
    q = quality_summary(df)
    lines += [f"## Dataset: {state.get('dataset_name', 'uploaded data')}",
              f"- Shape: {q['rows']:,} rows x {q['columns']} columns "
              f"({q['numeric']} numeric, {q['categorical']} categorical)",
              f"- Missing cells: {q['missing_cells']:,} ({q['missing_pct']}%)",
              f"- Duplicate rows: {q['duplicates']:,}", f"- Data-quality score (heuristic): {q['quality_score']}/100", ""]
    for entry in state.get("cleaning_log", []):
        lines.append(f"- Cleaning: {entry}")
    cfg = state.get("prep_cfg")
    if cfg:
        lines += ["", "## Preprocessing pipeline (fitted on training data only)"]
        lines += [f"- {k}: {v}" for k, v in cfg.items()]
    results = state.get("results", {})
    for task in ("classification", "regression"):
        recs = [r["record"] for r in results.values() if r["record"]["task"] == task]
        if not recs:
            continue
        lines += ["", f"## {task.title()} results"]
        for (ds, tgt), group in pd.DataFrame(recs).groupby(["dataset", "target"]):
            runs = [r for r in recs if r["dataset"] == ds and r["target"] == tgt]
            lb = leaderboard(runs, task)
            best = lb.iloc[0]
            metric = "F1" if task == "classification" else "RMSE"
            val = best["f1"] if task == "classification" else best["rmse"]
            lines.append(f"- Target `{tgt}`: {len(runs)} run(s); best = **{best['model']}** ({metric} {_fmt(val)})")
            lines.append("")
            lines.append(lb.round(4).to_markdown(index=False) if hasattr(lb, "to_markdown") and _has_tabulate()
                         else lb.round(4).to_string(index=False))
    cs = state.get("cluster_info")
    if cs:
        lines += ["", "## Clustering", f"- Algorithm: {cs['algorithm']} on {len(cs['columns'])} feature(s)",
                  f"- Clusters: {cs['quality']['clusters']}, noise points: {cs['quality']['noise_points']}",
                  f"- Silhouette: {_fmt(cs['quality']['silhouette'])}"]
    an = state.get("anomaly_df")
    if an is not None:
        lines += ["", "## Anomaly detection (Isolation Forest)",
                  f"- Flagged {int(an['is_anomaly'].sum())} of {len(an)} rows "
                  f"({100 * an['is_anomaly'].mean():.1f}%)"]
    return "\n".join(lines)


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except Exception:
        return False


def results_table(results: dict) -> pd.DataFrame:
    """One row per trained model with every metric (for the metrics download)."""
    rows = []
    for item in results.values():
        r = item["record"]
        row = {"run": r["id"], "task": r["task"], "dataset": r["dataset"], "target": r["target"],
               "model": r["model"], "tuned": r["tuned"], "best_params": str(r["best_params"]),
               "n_train": r["n_train"], "n_test": r["n_test"], "n_features": len(r["features"])}
        row.update(r["metrics"])
        row.update({"cv_metric": r["cv_metric"], "cv_mean": r["cv_mean"], "cv_std": r["cv_std"],
                    "train_score": r["train_score"], "train_seconds": r["train_seconds"]})
        rows.append(row)
    return pd.DataFrame(rows)
