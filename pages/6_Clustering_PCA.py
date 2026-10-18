import numpy as np
import pandas as pd
import streamlit as st

from utils import plots
from utils import unsupervised as un
from utils.data import split_columns, suggest_drop_columns
from utils.ui import download_csv, header, require_data, setup_page, show

setup_page("Clustering & PCA", "🔵")
header("🔵 Clustering & PCA", "Find structure without a target column.")
df = require_data()
cfg = st.session_state["prep_cfg"]
num, cat, _ = split_columns(df)
ignore = suggest_drop_columns(df)
num_ok = [c for c in num if c not in ignore]

tab_c, tab_p = st.tabs(["Clustering", "PCA"])

# ======================================================================= clustering
with tab_c:
    if len(num_ok) < 2:
        st.error("Clustering needs at least two numeric columns (ID-like and constant columns are excluded).")
    else:
        cols = st.multiselect("Features to cluster on", num_ok, default=num_ok[:12])
        use_cat = st.checkbox("Also include categorical columns (one-hot encoded)", value=False,
                              disabled=not cat)
        if use_cat:
            cols = cols + [c for c in cat if df[c].nunique() <= 20]
        algo = st.radio("Algorithm", ["K-Means", "Agglomerative (hierarchical)", "DBSCAN"], horizontal=True)
        st.caption("Features are imputed and scaled with the pipeline settings from the Preprocessing page "
                   "(clustering is distance-based, so scaling matters).")

        params = {}
        if algo == "K-Means":
            params["k"] = st.slider("Number of clusters (k)", 2, 12, 3)
        elif algo.startswith("Agglomerative"):
            params["k"] = st.slider("Number of clusters", 2, 12, 3)
            params["linkage"] = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
        else:
            params["eps"] = st.slider("eps (neighbourhood radius)", 0.1, 10.0, 1.5, 0.1)
            params["min_samples"] = st.slider("min_samples", 2, 30, 5)

        if len(cols) < 2:
            st.warning("Select at least two features.")
        else:
            if algo == "K-Means" and st.checkbox("Show elbow / silhouette sweep (k = 2…10)"):
                try:
                    X_all, _ = un.build_matrix(df, cols, cfg)
                    with st.spinner("Fitting K-Means for k = 2…10"):
                        sweep = un.kmeans_sweep(X_all, 2, 10)
                    show(plots.fig_lines(sweep["k"], {"Inertia (elbow)": sweep["inertia"],
                                                      "Silhouette (higher is better)": sweep["silhouette"]}, "k"))
                except Exception as exc:
                    st.error(f"Sweep failed: {exc}")
            if algo == "DBSCAN" and st.checkbox("Show k-distance plot (helps choose eps)"):
                X_all, _ = un.build_matrix(df, cols, cfg)
                kd = un.k_distance(X_all, params["min_samples"])
                show(plots.fig_lines(np.arange(len(kd)), {f"Sorted distance to neighbour #{params['min_samples']}": kd}))
                st.caption("Pick eps near the 'knee' of this curve.")

            if st.button("Run clustering", type="primary"):
                try:
                    data = df.dropna(how="all", subset=cols)
                    note = ""
                    if algo.startswith("Agglomerative") and len(data) > un.AGGLO_MAX_ROWS:
                        data = data.sample(un.AGGLO_MAX_ROWS, random_state=42)
                        note = f"Hierarchical clustering uses a random sample of {un.AGGLO_MAX_ROWS:,} rows."
                    data = data.reset_index(drop=True)
                    with st.spinner("Clustering…"):
                        X, names = un.build_matrix(data, cols, cfg)
                        if X.shape[1] < 2:
                            raise ValueError("Need at least two features after encoding.")
                        if algo == "K-Means":
                            labels = un.run_kmeans(X, params["k"])
                        elif algo.startswith("Agglomerative"):
                            labels = un.run_agglomerative(X, params["k"], params["linkage"])
                        else:
                            labels = un.run_dbscan(X, params["eps"], params["min_samples"])
                        pca2, Z = un.run_pca(X, 2)
                    quality = un.cluster_quality(X, labels)
                    st.session_state["cluster_run"] = {
                        "algo": algo, "params": params, "cols": cols, "labels": labels, "X": X,
                        "Z": Z, "var": pca2.explained_variance_ratio_, "quality": quality, "note": note,
                        "data": data}
                    out = data.copy()
                    out["cluster"] = labels
                    st.session_state["cluster_df"] = out
                    st.session_state["cluster_info"] = {"algorithm": algo, "columns": cols, "quality": quality,
                                                        "params": params}
                except Exception as exc:
                    st.error(f"Clustering failed: {exc}")

        run = st.session_state["cluster_run"]
        if run:
            st.divider()
            st.subheader(f"Results · {run['algo']}")
            if run["note"]:
                st.info(run["note"])
            q = run["quality"]
            m = st.columns(4)
            m[0].metric("Clusters", q["clusters"])
            m[1].metric("Noise points", q["noise_points"])
            m[2].metric("Silhouette", "n/a" if q["silhouette"] is None else f"{q['silhouette']:.3f}")
            m[3].metric("Davies-Bouldin ↓", f"{q['davies_bouldin']:.3f}" if "davies_bouldin" in q else "n/a")
            if q["silhouette"] is None:
                st.warning("Silhouette is undefined: fewer than two clusters were found. "
                           "Adjust the parameters (for DBSCAN, try a larger eps or smaller min_samples).")
            else:
                s = q["silhouette"]
                st.caption("Silhouette interpretation: " + (
                    "strong structure (>0.5)" if s > 0.5 else
                    "reasonable structure (0.25-0.5)" if s > 0.25 else "weak or overlapping clusters (<0.25)"))
            a, b = st.columns(2)
            with a:
                show(plots.fig_clusters(run["Z"], run["labels"], f"PC1 ({run['var'][0]:.0%})",
                                        f"PC2 ({run['var'][1]:.0%})", "Clusters in PCA space"))
            with b:
                counts = pd.Series(run["labels"]).value_counts().sort_index()
                show(plots.fig_bar([("noise" if i == -1 else f"cluster {i}") for i in counts.index],
                                   counts.values, "Cluster sizes", "rows"))
            if run["algo"].startswith("Agglomerative") and st.checkbox("Show dendrogram"):
                show(plots.fig_dendrogram(run["X"], run["params"]["linkage"] if run["params"]["linkage"] != "ward" else "ward"))
            summary_cols = [c for c in run["cols"] if c in num]
            st.markdown("**Cluster profiles** (mean of each numeric feature in original units)")
            st.dataframe(un.cluster_summary(run["data"], run["labels"], summary_cols), hide_index=True)
            download_csv("⬇️ Download data with cluster labels", st.session_state["cluster_df"], "clustered_data.csv")

# ======================================================================= PCA
with tab_p:
    if len(num_ok) < 2:
        st.error("PCA needs at least two numeric columns.")
    else:
        cols = st.multiselect("Numeric features", num_ok, default=num_ok[:15], key="pca_cols")
        if len(cols) < 2:
            st.warning("Select at least two features.")
        else:
            max_c = min(10, len(cols))
            n_comp = st.slider("Components", 2, max_c, min(max_c, 5)) if max_c > 2 else 2
            try:
                data = df.reset_index(drop=True)
                X, names = un.build_matrix(data, cols, {**cfg, "scaler": "standard"})
                pca, Z = un.run_pca(X, n_comp)
                ratio = pca.explained_variance_ratio_
                a, b = st.columns(2)
                with a:
                    show(plots.fig_explained_variance(ratio))
                with b:
                    cum = np.cumsum(ratio)
                    need = int(np.searchsorted(cum, 0.95) + 1)
                    st.metric("Variance explained (shown components)", f"{cum[-1]:.1%}")
                    st.write(f"{need if cum[-1] >= 0.95 else '>' + str(len(cum))} component(s) "
                             "needed for 95% of the variance.")
                    st.dataframe(pd.DataFrame({"component": [f"PC{i + 1}" for i in range(len(ratio))],
                                               "variance_%": (ratio * 100).round(2),
                                               "cumulative_%": (cum * 100).round(2)}), hide_index=True)
                color_opts = ["(none)"] + [c for c in df.columns if df[c].nunique() <= 12]
                color = st.selectbox("Colour the projection by", color_opts)
                if color == "(none)":
                    codes, legend = np.zeros(len(Z), dtype=int), {0: "points"}
                else:
                    cats = data[color].astype(str).astype("category")
                    codes = cats.cat.codes.to_numpy()
                    legend = dict(enumerate(cats.cat.categories))
                show(plots.fig_clusters(Z, codes, f"PC1 ({ratio[0]:.0%})", f"PC2 ({ratio[1]:.0%})",
                                        "PCA projection", names=legend))
                st.markdown("**Component loadings** (how strongly each feature contributes)")
                st.dataframe(un.pca_loadings(pca, names))
                proj = pd.DataFrame(Z, columns=[f"PC{i + 1}" for i in range(Z.shape[1])])
                download_csv("⬇️ Download PCA-transformed data", proj, "pca_transformed.csv")
            except Exception as exc:
                st.error(f"PCA failed: {exc}")
