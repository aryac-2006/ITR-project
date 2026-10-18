import streamlit as st

from utils.models import bundle_to_bytes, make_bundle
from utils.reports import build_summary, results_table
from utils.ui import download_csv, header, require_data, setup_page

setup_page("Reports", "📦")
header("📦 Reports & Downloads", "Export everything produced in this session.")
df = require_data()
state = st.session_state
results = state["results"]

st.subheader("Data")
download_csv("⬇️ Cleaned dataset", df, "cleaned_dataset.csv")

st.subheader("Supervised models")
if results:
    download_csv("⬇️ Evaluation metrics (all runs)", results_table(results), "evaluation_metrics.csv")
    rid = st.selectbox("Model file", list(results), format_func=lambda i: (
        f"{results[i]['record']['model']} · {results[i]['record']['target']} · #{i}"))
    rec = results[rid]["record"]
    st.download_button("⬇️ Trained pipeline (.joblib)",
                       bundle_to_bytes(make_bundle(rec, results[rid]["artifact"])),
                       file_name=f"{rec['model'].lower().replace(' ', '_')}_{rid}.joblib",
                       mime="application/octet-stream")
else:
    st.caption("No models trained yet.")

st.subheader("Unsupervised results")
if state["cluster_df"] is not None:
    download_csv("⬇️ Clustering results", state["cluster_df"], "clustered_data.csv")
else:
    st.caption("No clustering run yet.")
if state["anomaly_df"] is not None:
    download_csv("⬇️ Anomaly report", state["anomaly_df"], "anomaly_report.csv")
else:
    st.caption("No anomaly detection run yet.")

st.subheader("Analysis summary")
summary = build_summary(state)
st.download_button("⬇️ Analysis summary (Markdown)", summary.encode("utf-8"),
                   file_name="analysis_summary.md", mime="text/markdown")
with st.expander("Preview summary", expanded=True):
    st.markdown(summary)
