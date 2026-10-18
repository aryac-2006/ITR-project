import pandas as pd
import streamlit as st

from utils import plots
from utils.data import split_columns
from utils.ui import header, require_data, setup_page, show

setup_page("Exploratory Analysis", "🔍")
header("🔍 Exploratory Data Analysis", "Understand distributions, relationships and gaps before modelling.")
df = require_data()
num, cat, _ = split_columns(df)

tab_miss, tab_dist, tab_corr, tab_rel = st.tabs(["Missing & duplicates", "Distributions", "Correlations", "Relationships"])

with tab_miss:
    c1, c2 = st.columns(2)
    c1.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c2.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    if df.isna().any().any():
        show(plots.fig_missing(df))
    else:
        st.success("No missing values.")
    if df.duplicated().any():
        with st.expander("Show duplicate rows"):
            st.dataframe(df[df.duplicated(keep=False)].sort_values(list(df.columns)[:3]))

with tab_dist:
    col = st.selectbox("Column", list(df.columns))
    if col in num:
        a, b = st.columns(2)
        with a:
            show(plots.fig_hist(df[col], st.slider("Bins", 5, 80, 30)))
        with b:
            show(plots.fig_box(df, [col]))
        st.dataframe(df[col].describe().to_frame().T.round(3))
    else:
        show(plots.fig_counts(df[col]))
        st.caption(f"{df[col].nunique()} distinct values.")

with tab_corr:
    if len(num) < 2:
        st.info("At least two numeric columns are needed.")
    else:
        method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
        chosen = st.multiselect("Columns", num, default=num[:15])
        if len(chosen) >= 2:
            corr = df[chosen].corr(method=method)
            show(plots.fig_corr(corr))
            pairs = corr.where(~pd.DataFrame(
                [[i >= j for j in range(len(corr))] for i in range(len(corr))],
                index=corr.index, columns=corr.columns)).stack().rename("correlation").reset_index()
            pairs.columns = ["feature_a", "feature_b", "correlation"]
            pairs["abs"] = pairs["correlation"].abs()
            st.markdown("**Strongest pairs** (|r| above 0.8 suggests redundant features)")
            st.dataframe(pairs.sort_values("abs", ascending=False).drop(columns="abs").head(10).round(3), hide_index=True)

with tab_rel:
    if len(num) < 2:
        st.info("At least two numeric columns are needed.")
    else:
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X axis", num, index=0)
        y = c2.selectbox("Y axis", num, index=1)
        hue = c3.selectbox("Colour by", ["(none)"] + [c for c in df.columns if df[c].nunique() <= 12])
        sample = df.sample(min(3000, len(df)), random_state=0)
        show(plots.fig_scatter(sample, x, y, None if hue == "(none)" else hue))
        st.caption("Plots use at most 3,000 randomly sampled rows.")
