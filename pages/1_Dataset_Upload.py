import streamlit as st

from utils.data import (SAMPLE_DATASETS, column_report, limit_rows, load_csv, load_sample,
                        quality_summary, split_columns)
from utils.ui import header, set_dataset, setup_page

setup_page("Dataset Upload", "📤")
header("📤 Dataset Upload", "Bring your own CSV or start from a bundled sample dataset.")

tab_up, tab_sample = st.tabs(["Upload CSV", "Sample datasets"])
with tab_up:
    file = st.file_uploader("Choose a CSV file (max 50 MB)", type=["csv"])
    if file is not None:
        sig = (file.name, file.size)
        if st.session_state.get("upload_sig") != sig:
            try:
                with st.spinner("Reading file…"):
                    data, sampled = limit_rows(load_csv(file))
                set_dataset(data, file.name)
                st.session_state["upload_sig"] = sig
                st.success(f"Loaded **{file.name}**: {data.shape[0]:,} rows × {data.shape[1]} columns.")
                if sampled:
                    st.warning("File exceeds 50,000 rows; a random sample of 50,000 rows is used.")
            except ValueError as exc:
                st.error(f"Could not load the file: {exc}")
with tab_sample:
    choice = st.selectbox("Sample dataset", list(SAMPLE_DATASETS))
    if st.button("Load sample dataset", type="primary"):
        set_dataset(load_sample(SAMPLE_DATASETS[choice]), choice.split(" (")[0])
        st.session_state["upload_sig"] = None
        st.success(f"Loaded **{choice}**.")

df = st.session_state["df_raw"]
if df is None:
    st.info("Upload a CSV or load a sample to continue.")
    st.stop()

q = quality_summary(df)
st.subheader(f"Overview · {st.session_state['dataset_name']}")
for col, (label, val) in zip(st.columns(6), [
        ("Rows", f"{q['rows']:,}"), ("Columns", q["columns"]),
        ("Missing cells", f"{q['missing_cells']:,} ({q['missing_pct']}%)"),
        ("Duplicate rows", f"{q['duplicates']:,}"), ("Memory", f"{q['memory_mb']} MB"),
        ("Quality score", f"{q['quality_score']}/100")]):
    col.metric(label, val)
st.caption("Quality score is a simple heuristic: it penalises missing cells, duplicate rows, "
           "constant columns and very high-cardinality text columns.")

issues = []
if q["duplicates"]:
    issues.append(f"{q['duplicates']} duplicate rows (remove on the Preprocessing page).")
if q["missing_cells"]:
    issues.append(f"{q['missing_cells']} missing cells (handled by the imputation step of the pipeline).")
if q["constant_columns"]:
    issues.append(f"Constant columns: {', '.join(q['constant_columns'])}.")
if q["high_cardinality_columns"]:
    issues.append(f"High-cardinality text columns: {', '.join(q['high_cardinality_columns'])}.")
if q["other"]:
    issues.append("Date/time columns are present; they are excluded from modelling.")
for i in issues:
    st.warning(i)
if not issues:
    st.success("No obvious data-quality problems detected.")

t1, t2, t3, t4 = st.tabs(["Preview", "Columns & types", "Numeric statistics", "Categorical statistics"])
with t1:
    n = st.slider("Rows to preview", 5, min(200, len(df)), min(20, len(df)))
    st.dataframe(df.head(n))
with t2:
    st.dataframe(column_report(df), hide_index=True)
with t3:
    num, cat, _ = split_columns(df)
    if num:
        st.dataframe(df[num].describe().T.round(3))
    else:
        st.info("No numeric columns.")
with t4:
    if cat:
        st.dataframe(df[cat].astype(object).describe().T)
    else:
        st.info("No categorical columns.")
