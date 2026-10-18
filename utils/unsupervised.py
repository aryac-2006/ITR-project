"""Clustering, PCA and anomaly detection helpers (no Streamlit dependency)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from utils.data import split_columns
from utils.preprocessing import make_preprocessor

AGGLO_MAX_ROWS = 5_000  # hierarchical clustering is O(n^2) in memory


def build_matrix(df: pd.DataFrame, cols, cfg=None):
    """Impute + scale (+ one-hot) the chosen columns. Unsupervised methods have no train/test
    split, so fitting on the full table is correct here."""
    d = df[list(cols)].copy()
    num, cat, _ = split_columns(d)
    d[num] = d[num].replace([np.inf, -np.inf], np.nan)
    for c in cat:
        d[c] = d[c].astype(object).where(d[c].notna(), np.nan)
    pre = make_preprocessor(num, cat, cfg)
    X = pre.fit_transform(d)
    names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    return np.asarray(X, dtype=float), names


def safe_silhouette(X, labels, max_samples=5000, seed=42):
    """Silhouette ignoring DBSCAN noise; None when it is undefined."""
    labels = np.asarray(labels)
    mask = labels != -1
    if mask.sum() < 3 or len(set(labels[mask])) < 2 or len(set(labels[mask])) >= mask.sum():
        return None
    return float(silhouette_score(X[mask], labels[mask], sample_size=min(max_samples, int(mask.sum())),
                                  random_state=seed))


def cluster_quality(X, labels) -> dict:
    labels = np.asarray(labels)
    mask = labels != -1
    k = len(set(labels[mask]))
    out = {"clusters": k, "noise_points": int((~mask).sum()), "silhouette": safe_silhouette(X, labels)}
    if k >= 2 and mask.sum() > k:
        out["davies_bouldin"] = float(davies_bouldin_score(X[mask], labels[mask]))
        out["calinski_harabasz"] = float(calinski_harabasz_score(X[mask], labels[mask]))
    return out


def kmeans_sweep(X, k_min=2, k_max=10, seed=42) -> pd.DataFrame:
    rows = []
    for k in range(k_min, min(k_max, len(X) - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": safe_silhouette(X, km.labels_)})
    return pd.DataFrame(rows)


def run_kmeans(X, k, seed=42):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


def run_agglomerative(X, k, linkage="ward"):
    return AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X)


def run_dbscan(X, eps, min_samples):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)


def k_distance(X, k):
    """Sorted distance to each point's k-th neighbour (used to choose DBSCAN eps)."""
    k = min(k, len(X) - 1)
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return np.sort(d[:, -1])


def cluster_summary(df: pd.DataFrame, labels, numeric_cols) -> pd.DataFrame:
    d = df.copy()
    d["cluster"] = np.asarray(labels)
    size = d.groupby("cluster").size().rename("size")
    out = pd.concat([size, (100 * size / len(d)).round(1).rename("share_%"),
                     d.groupby("cluster")[list(numeric_cols)].mean().round(3)], axis=1)
    return out.reset_index()


def run_pca(X, n_components):
    n = int(min(n_components, X.shape[0], X.shape[1]))
    pca = PCA(n_components=n, random_state=42)
    return pca, pca.fit_transform(X)


def pca_loadings(pca, names) -> pd.DataFrame:
    return pd.DataFrame(pca.components_.T, index=names,
                        columns=[f"PC{i + 1}" for i in range(pca.n_components_)]).round(4)


def run_isolation_forest(X, contamination=0.05, n_estimators=200, seed=42):
    """Return (is_anomaly bool array, anomaly score where higher = more unusual)."""
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                          random_state=seed, n_jobs=-1).fit(X)
    return iso.predict(X) == -1, -iso.score_samples(X)
