import pandas as pd
import streamlit as st

from utils import plots
from utils.data import clean_dataframe, split_columns, suggest_drop_columns
from utils.preprocessing import DEFAULT_PREP, outlier_report, rank_features
from utils.ui import download_csv, header, require_data, setup_page, show
from utils.unsupervised import build_matrix

setup_page("Preprocessing", "🧹")
header("🧹 Data Preprocessing", "Clean the table, then configure the leakage-safe model pipeline.")
require_data()
raw = st.session_state["df_raw"]
df = st.session_state["df"]

st.info("**How leakage is prevented:** this page only performs row/column cleaning that uses no learned "
        "statistics. Imputation values, encoders, scalers, outlier bounds and feature selection are "
        "defined as one scikit-learn `Pipeline` and are **fitted on training data only** during every "
        "split and cross-validation fold.")

# ------------------------------------------------------------------ 1. cleaning
st.subheader("1 · Dataset cleaning")
suggestions = suggest_drop_columns(raw)
c1, c2 = st.columns(2)
drop_dupes = c1.checkbox("Remove duplicate rows", value=True)
max_missing = c2.slider("Drop columns with more than X% missing", 10, 100, 100, 5,
                        help="100 keeps every column.")
drop_cols = st.multiselect("Drop columns manually", list(raw.columns),
                           default=list(suggestions),
                           help="Pre-selected: constant and ID-like columns.")
for c, why in suggestions.items():
    st.caption(f"Suggested: **{c}** - {why}")
a, b = st.columns(2)
if a.button("Apply cleaning", type="primary"):
    if len(drop_cols) >= raw.shape[1] - 1:
        st.error("Keep at least two columns.")
    else:
        cleaned, log = clean_dataframe(raw, drop_dupes, drop_cols, max_missing)
        st.session_state.update({"df": cleaned, "cleaning_log": log, "cluster_run": None,
                                 "cluster_df": None, "cluster_info": None, "anomaly_df": None})
        df = cleaned
        st.success(f"Cleaned dataset: {cleaned.shape[0]:,} rows × {cleaned.shape[1]} columns.")
if b.button("Reset to raw data"):
    st.session_state.update({"df": raw.copy(), "cleaning_log": []})
    df = st.session_state["df"]
for line in st.session_state["cleaning_log"]:
    st.write(f"- {line}")
st.caption(f"Current dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
download_csv("⬇️ Download cleaned dataset (CSV)", df, "cleaned_dataset.csv")

# ------------------------------------------------------------------ 2. pipeline
st.subheader("2 · Pipeline configuration")
cfg = dict(st.session_state["prep_cfg"])


def pick(label, options, key, labels=None):
    return st.selectbox(label, options, index=options.index(cfg[key]),
                        format_func=(lambda o: labels[o]) if labels else str)


c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Missing values**")
    cfg["num_impute"] = pick("Numeric", ["median", "mean", "most_frequent", "knn"], "num_impute",
                             {"median": "Median", "mean": "Mean", "most_frequent": "Most frequent",
                              "knn": "KNN imputer (k=5)"})
    cfg["cat_impute"] = pick("Categorical", ["most_frequent", "missing_label"], "cat_impute",
                             {"most_frequent": "Most frequent", "missing_label": "Constant 'missing'"})
with c2:
    st.markdown("**Encoding & scaling**")
    cfg["encoding"] = pick("Categorical encoding", ["onehot", "ordinal"], "encoding",
                           {"onehot": "One-hot", "ordinal": "Ordinal codes"})
    cfg["scaler"] = pick("Numeric scaling", ["standard", "minmax", "robust", "none"], "scaler",
                         {"standard": "StandardScaler", "minmax": "MinMaxScaler",
                          "robust": "RobustScaler (outlier-resistant)", "none": "None"})
with c3:
    st.markdown("**Outliers & feature selection**")
    cfg["outliers"] = pick("Outlier handling", ["none", "clip"], "outliers",
                           {"none": "Keep as is", "clip": "Clip to IQR bounds (winsorise)"})
    if cfg["outliers"] == "clip":
        cfg["iqr_factor"] = st.slider("IQR factor", 1.0, 3.0, float(cfg["iqr_factor"]), 0.25)
    cfg["selector"] = pick("Feature selection", ["none", "kbest", "variance"], "selector",
                           {"none": "None", "kbest": "SelectKBest (univariate F-test)",
                            "variance": "Remove zero-variance features"})
    if cfg["selector"] == "kbest":
        cfg["k_features"] = st.slider("Features to keep (k)", 1, 50, int(cfg["k_features"]))
st.session_state["prep_cfg"] = cfg
if st.button("Restore defaults"):
    st.session_state["prep_cfg"] = dict(DEFAULT_PREP)
    st.rerun()
if cfg["scaler"] == "none":
    st.warning("Without scaling, distance/margin-based models (KNN, SVM, logistic regression) can perform poorly.")
if cfg["encoding"] == "ordinal":
    st.warning("Ordinal codes imply an order between categories; prefer one-hot for nominal data.")

with st.expander("Preview the transformed feature matrix"):
    cols = st.multiselect("Columns to preview", list(df.columns), default=list(df.columns)[:8])
    if cols:
        try:
            X, names = build_matrix(df, cols, cfg)
            st.write(f"Transformed shape: **{X.shape[0]:,} × {X.shape[1]}**")
            st.dataframe(pd.DataFrame(X[:20], columns=names).round(3))
            st.caption("Preview only - fitted on the whole table. Real training always refits on the training split.")
        except Exception as exc:
            st.error(f"Could not transform these columns: {exc}")

# ------------------------------------------------------------------ 3. outliers
st.subheader("3 · Outlier analysis")
num, cat, _ = split_columns(df)
if num:
    c1, c2 = st.columns([3, 1])
    chosen = c1.multiselect("Numeric columns", num, default=num[:8])
    factor = c2.slider("IQR factor", 1.0, 3.0, 1.5, 0.25, key="iqr_view")
    if chosen:
        rep = outlier_report(df, chosen, factor)
        st.dataframe(rep.round(3), hide_index=True)
        show(plots.fig_box(df, chosen[:10]))
        st.caption("Outliers are only reported here. Choose 'Clip to IQR bounds' above to treat them "
                   "inside the pipeline (bounds learned from training data).")
else:
    st.info("No numeric columns to analyse.")

# ------------------------------------------------------------------ 4. feature relevance
st.subheader("4 · Feature relevance (guidance)")
c1, c2, c3 = st.columns(3)
target = c1.selectbox("Target", list(df.columns), index=len(df.columns) - 1, key="rel_target")
task = c2.radio("Task", ["classification", "regression"], horizontal=True)
if c3.button("Rank features"):
    feats = [c for c in df.columns if c != target and c not in suggestions]
    try:
        with st.spinner("Computing mutual information…"):
            ranking = rank_features(df.sample(min(len(df), 5000), random_state=0), feats, target, task)
        show(plots.fig_bar(ranking["feature"], ranking["mutual_information"],
                           "Mutual information with target", "mutual information"))
        st.dataframe(ranking.round(4), hide_index=True)
    except Exception as exc:
        st.error(f"Could not rank features for this target/task: {exc}")
