import numpy as np
import pandas as pd
import streamlit as st

from utils import plots
from utils import unsupervised as un
from utils.data import split_columns, suggest_drop_columns
from utils.ui import download_csv, header, require_data, setup_page, show

setup_page("Anomaly Detection", "🚨")
header("🚨 Anomaly Detection", "Isolation Forest isolates unusual rows without needing labels.")
df = require_data()
cfg = st.session_state["prep_cfg"]
num, cat, _ = split_columns(df)
ignore = suggest_drop_columns(df)
num_ok = [c for c in num if c not in ignore]
if not num_ok:
    st.error("Anomaly detection needs at least one numeric column.")
    st.stop()

cols = st.multiselect("Features to analyse", num_ok, default=num_ok[:12])
c1, c2, c3 = st.columns(3)
contamination = c1.slider("Expected anomaly share", 0.001, 0.30, 0.05, 0.001, format="%.3f",
                          help="Fraction of rows flagged as anomalies.")
n_est = c2.slider("Trees", 50, 500, 200, 50)
seed = int(c3.number_input("Random seed", 0, 9999, 42))
st.caption("Missing values are imputed and features scaled using the Preprocessing page settings. "
           "The model is unsupervised, so there is no train/test split.")

if st.button("Detect anomalies", type="primary", disabled=not cols):
    try:
        with st.spinner("Fitting Isolation Forest…"):
            data = df.reset_index(drop=True)
            X, _ = un.build_matrix(data, cols, cfg)
            flag, score = un.run_isolation_forest(X, contamination, n_est, seed)
            out = data.copy()
            out["anomaly_score"] = score.round(4)
            out["is_anomaly"] = flag
            _, Z = un.run_pca(X, 2) if X.shape[1] >= 2 else (None, np.c_[X[:, 0], np.zeros(len(X))])
        st.session_state["anomaly_df"] = out
        st.session_state["anomaly_aux"] = {"Z": Z, "cols": cols, "threshold": float(score[flag].min()) if flag.any() else None}
    except Exception as exc:
        st.error(f"Anomaly detection failed: {exc}")

out = st.session_state["anomaly_df"]
if out is not None and "anomaly_aux" in st.session_state:
    aux = st.session_state["anomaly_aux"]
    st.divider()
    m = st.columns(3)
    m[0].metric("Rows analysed", f"{len(out):,}")
    m[1].metric("Flagged anomalies", f"{int(out['is_anomaly'].sum()):,}")
    m[2].metric("Share", f"{100 * out['is_anomaly'].mean():.1f}%")
    a, b = st.columns(2)
    with a:
        show(plots.fig_score_hist(out["anomaly_score"], aux["threshold"]))
    with b:
        show(plots.fig_clusters(aux["Z"], out["is_anomaly"].astype(int).to_numpy(), "PC1", "PC2",
                                "Anomalies in PCA space", names={0: "normal", 1: "anomaly"}))
    top = st.slider("Show top N anomalies", 5, 100, 20)
    st.markdown("**Most anomalous rows**")
    st.dataframe(out.sort_values("anomaly_score", ascending=False).head(top))
    used = [c for c in aux["cols"] if c in out.columns]
    prof = out.groupby("is_anomaly")[used].mean().T.rename(columns={False: "normal_mean", True: "anomaly_mean"})
    if {"normal_mean", "anomaly_mean"} <= set(prof.columns):
        prof["difference_%"] = (100 * (prof["anomaly_mean"] - prof["normal_mean"])
                                / prof["normal_mean"].abs().replace(0, np.nan)).round(1)
        st.markdown("**How anomalies differ from normal rows (feature means)**")
        st.dataframe(prof.round(3))
    c1, c2 = st.columns(2)
    with c1:
        download_csv("⬇️ Download full anomaly report", out, "anomaly_report.csv")
    with c2:
        download_csv("⬇️ Download flagged rows only", out[out["is_anomaly"]], "anomalies_only.csv")
