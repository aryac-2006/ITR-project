"""Dark-themed matplotlib/seaborn figure builders (no Streamlit dependency)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from cycler import cycler  # noqa: E402

BG, PANEL, TEXT, GRID = "#0E1117", "#161B26", "#E6EDF3", "#2A3142"
PALETTE = ["#6C8CFF", "#FF7A90", "#4ADE80", "#FBBF24", "#A78BFA", "#22D3EE",
           "#FB923C", "#F472B6", "#94A3B8", "#84CC16"]


def apply_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT, "grid.color": GRID, "axes.grid": True,
        "grid.alpha": 0.4, "figure.dpi": 100, "axes.prop_cycle": cycler(color=PALETTE),
        "axes.titleweight": "bold", "legend.facecolor": PANEL, "legend.edgecolor": GRID,
    })
    sns.set_palette(PALETTE)


def _fig(w=7, h=4):
    return plt.subplots(figsize=(w, h), constrained_layout=True)


def fig_missing(df: pd.DataFrame, top: int = 25):
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0].head(top)
    fig, ax = _fig(7, max(2.5, 0.35 * len(miss) + 1))
    ax.barh(miss.index[::-1], miss.values[::-1], color=PALETTE[1])
    ax.set(xlabel="% missing", title="Missing values by column")
    return fig


def fig_hist(s: pd.Series, bins: int = 30):
    fig, ax = _fig(6, 3.6)
    sns.histplot(s.dropna(), bins=bins, kde=True, ax=ax, color=PALETTE[0])
    ax.set(title=f"Distribution of {s.name}")
    return fig


def fig_box(df: pd.DataFrame, cols):
    fig, ax = _fig(max(5, 1.3 * len(cols)), 3.6)
    sns.boxplot(data=df[list(cols)], ax=ax, color=PALETTE[0], fliersize=3)
    ax.set(title="Box plots (points beyond whiskers = IQR outliers)")
    return fig


def fig_counts(s: pd.Series, top: int = 20):
    vc = s.astype(str).value_counts().head(top)
    fig, ax = _fig(6, max(2.5, 0.32 * len(vc) + 1))
    ax.barh(vc.index[::-1], vc.values[::-1], color=PALETTE[2])
    ax.set(title=f"Top values of {s.name}", xlabel="count")
    return fig


def fig_corr(corr: pd.DataFrame):
    n = len(corr)
    fig, ax = _fig(min(11, 2 + 0.6 * n), min(9, 1.5 + 0.5 * n))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=n <= 12, fmt=".2f", ax=ax,
                linewidths=0.3, linecolor=BG, cbar_kws={"shrink": 0.8})
    ax.grid(False)
    ax.set_title("Correlation matrix")
    return fig


def fig_scatter(df, x, y, hue=None):
    fig, ax = _fig(6.5, 4.2)
    sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, alpha=0.7, s=28)
    ax.set(title=f"{y} vs {x}")
    return fig


def fig_bar(labels, values, title, xlabel="", top=20, color=None):
    order = np.argsort(values)[::-1][:top]
    labels, values = np.asarray(labels)[order], np.asarray(values)[order]
    fig, ax = _fig(7, max(2.5, 0.33 * len(labels) + 1))
    ax.barh(labels[::-1], values[::-1], color=color or PALETTE[0])
    ax.set(title=title, xlabel=xlabel)
    return fig


def fig_confusion(cm, classes):
    fig, ax = _fig(5.2, 4.4)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes,
                ax=ax, cbar=False)
    ax.grid(False)
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix (test set)")
    return fig


def fig_roc(y_true, proba, classes):
    from sklearn.metrics import auc, roc_curve
    fig, ax = _fig(5.5, 4.4)
    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_true, proba[:, 1])
        ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
    else:
        for i, c in enumerate(classes):
            if (y_true == i).sum() == 0:
                continue
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), proba[:, i])
            ax.plot(fpr, tpr, label=f"{c} (AUC {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "--", color="#64748B")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)")
    ax.legend(loc="lower right")
    return fig


def fig_pred_vs_actual(y_true, y_pred):
    fig, ax = _fig(5.2, 4.4)
    ax.scatter(y_true, y_pred, alpha=0.6, s=20)
    lo, hi = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "--", color=PALETTE[1])
    ax.set(xlabel="Actual", ylabel="Predicted", title="Predicted vs actual (test set)")
    return fig


def fig_residuals(y_true, y_pred):
    fig, ax = _fig(5.2, 4.4)
    ax.scatter(y_pred, y_true - y_pred, alpha=0.6, s=20)
    ax.axhline(0, ls="--", color=PALETTE[1])
    ax.set(xlabel="Predicted", ylabel="Residual (actual - predicted)", title="Residuals (test set)")
    return fig


def fig_clusters(coords, labels, xlabel="PC1", ylabel="PC2", title="Clusters", names=None):
    fig, ax = _fig(6.5, 4.8)
    labels = np.asarray(labels)
    for i, lab in enumerate(sorted(set(labels))):
        m = labels == lab
        name = (names or {}).get(lab) or ("noise" if lab == -1 else f"cluster {lab}")
        color = "#64748B" if lab == -1 else PALETTE[i % len(PALETTE)]
        ax.scatter(coords[m, 0], coords[m, 1], s=22, alpha=0.75, label=name, color=color)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(loc="best", fontsize=8)
    return fig


def fig_lines(x, series: dict, xlabel="", title=""):
    """One axis per series (different scales), stacked side by side."""
    fig, axes = plt.subplots(1, len(series), figsize=(5 * len(series), 3.6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (name, y) in zip(axes, series.items()):
        ax.plot(x, y, marker="o")
        ax.set(xlabel=xlabel, title=name)
    return fig


def fig_explained_variance(ratio):
    fig, ax = _fig(6, 3.8)
    comps = np.arange(1, len(ratio) + 1)
    ax.bar(comps, ratio * 100, label="per component")
    ax.plot(comps, np.cumsum(ratio) * 100, color=PALETTE[1], marker="o", label="cumulative")
    ax.set(xlabel="Principal component", ylabel="% variance explained", title="PCA explained variance")
    ax.legend()
    return fig


def fig_dendrogram(X, linkage_method="ward", max_points=300):
    from scipy.cluster.hierarchy import dendrogram, linkage
    if len(X) > max_points:
        X = X[np.random.default_rng(0).choice(len(X), max_points, replace=False)]
    fig, ax = _fig(8, 4)
    dendrogram(linkage(X, method=linkage_method), no_labels=True, ax=ax,
               color_threshold=None, above_threshold_color=PALETTE[0])
    ax.set(title=f"Dendrogram ({len(X)} sampled points)", ylabel="distance")
    return fig


def fig_score_hist(scores, threshold=None):
    fig, ax = _fig(6, 3.6)
    sns.histplot(scores, bins=40, ax=ax, color=PALETTE[0])
    if threshold is not None:
        ax.axvline(threshold, color=PALETTE[1], ls="--", label="anomaly threshold")
        ax.legend()
    ax.set(xlabel="anomaly score (higher = more unusual)", title="Anomaly score distribution")
    return fig
