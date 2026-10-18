import pandas as pd
import streamlit as st

from utils import plots
from utils.models import leaderboard
from utils.reports import results_table
from utils.ui import download_csv, header, setup_page, show

setup_page("Model Comparison", "🏆")
header("🏆 Model Comparison & Leaderboard", "Compare models trained on the same dataset and target.")
results = st.session_state["results"]
if not results:
    st.warning("No models trained yet. Train some on the Classification or Regression page.")
    st.page_link("pages/4_Classification.py", label="Go to Classification", icon="🎯")
    st.page_link("pages/5_Regression.py", label="Go to Regression", icon="📈")
    st.stop()

groups = sorted({(r["record"]["task"], r["record"]["dataset"], r["record"]["target"]) for r in results.values()})
label = lambda g: f"{g[0].title()} · {g[1]} → {g[2]}"
group = st.selectbox("Problem", groups, format_func=label,
                     help="Models are only comparable when they predict the same target on the same data.")
task = group[0]
runs = [r for r in results.values() if (r["record"]["task"], r["record"]["dataset"], r["record"]["target"]) == group]
lb = leaderboard([r["record"] for r in runs], task)

metrics = (["f1", "accuracy", "precision", "recall", "roc_auc", "cv_mean"] if task == "classification"
           else ["rmse", "mae", "r2", "cv_mean"])
lower_better = {"rmse", "mae", "train_seconds"}
sort_by = st.selectbox("Rank by", metrics)
lb = lb.sort_values(sort_by, ascending=sort_by in lower_better, na_position="last").reset_index(drop=True)
lb["rank"] = range(1, len(lb) + 1)
gap = lb["train_score"] - lb["cv_mean"]
lb["overfit_flag"] = gap.map(lambda g: "⚠️ gap > 0.10" if g > 0.10 else "ok")
st.caption(f"{'Lower' if sort_by in lower_better else 'Higher'} is better for **{sort_by}**.")

col_cfg = {}
for m in ("f1", "accuracy", "r2"):
    if m in lb.columns:
        col_cfg[m] = st.column_config.ProgressColumn(m, min_value=0.0, max_value=1.0, format="%.3f")
st.dataframe(lb, hide_index=True, column_config=col_cfg)

best = lb.iloc[0]
st.success(f"🥇 Best by **{sort_by}**: **{best['model']}**{' (tuned)' if best['tuned'] else ''} "
           f"— {sort_by} = {best[sort_by]:.4f}")

a, b = st.columns(2)
with a:
    vals = lb[sort_by].fillna(0).to_numpy()
    show(plots.fig_bar(lb["model"] + " #" + lb["run"], vals, f"Test {sort_by} by model", sort_by))
with b:
    both = lb[["model", "cv_mean"] + (["f1"] if task == "classification" else ["r2"])].copy()
    both.columns = ["model", "cross-validation", "test"]
    long = both.melt(id_vars="model", var_name="split", value_name="score")
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    sns.barplot(data=long, y="model", x="score", hue="split", ax=ax)
    ax.set(title="CV score vs held-out test score", xlabel="F1 (weighted)" if task == "classification" else "R²", ylabel="")
    show(fig)
st.caption("CV and test scores that agree suggest the model generalises; large gaps between training "
           "and CV scores (see overfit_flag) suggest over-fitting.")

download_csv("⬇️ Download leaderboard (CSV)", lb, "leaderboard.csv")
download_csv("⬇️ Download all metrics for every run (CSV)", results_table(results), "all_model_metrics.csv")

with st.expander("Remove runs"):
    drop = st.multiselect("Runs to remove", [r["record"]["id"] for r in runs],
                          format_func=lambda i: f"{results[i]['record']['model']} #{i}")
    if st.button("Remove selected") and drop:
        for i in drop:
            results.pop(i, None)
        st.rerun()
