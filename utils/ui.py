"""Shared Streamlit UI helpers: page setup, theming, state, downloads."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from utils.plots import apply_theme
from utils.preprocessing import DEFAULT_PREP

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.hero {background: linear-gradient(120deg, #1B2540 0%, #2B1F4B 100%); border: 1px solid #2A3142;
       border-radius: 16px; padding: 1.6rem 1.8rem; margin-bottom: 1.2rem;}
.hero h1 {margin: 0 0 .3rem 0; font-size: 2.1rem;}
.hero p {margin: 0; color: #B6C2D9; font-size: 1.02rem;}
.card {background: #161B26; border: 1px solid #2A3142; border-radius: 12px; padding: 1rem 1.2rem; height: 100%;}
.card h4 {margin: 0 0 .4rem 0; color: #9DB4FF;}
.card p {margin: 0; color: #B6C2D9; font-size: .92rem;}
div[data-testid="stMetric"] {background: #161B26; border: 1px solid #2A3142; border-radius: 10px; padding: .6rem .9rem;}
</style>
"""


def init_state() -> None:
    defaults = {"df_raw": None, "df": None, "dataset_name": "", "cleaning_log": [],
                "prep_cfg": dict(DEFAULT_PREP), "results": {}, "cluster_run": None,
                "cluster_df": None, "cluster_info": None, "anomaly_df": None}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def setup_page(title: str, icon: str = "🧠") -> None:
    """Must be the first Streamlit call on every page."""
    st.set_page_config(page_title=f"{title} · IntelliData AI", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    apply_theme()
    init_state()
    with st.sidebar:
        st.markdown("### 🧠 IntelliData AI")
        df = st.session_state["df"]
        if df is None:
            st.caption("No dataset loaded")
        else:
            st.caption(f"**{st.session_state['dataset_name']}**  \n{df.shape[0]:,} rows × {df.shape[1]} cols")
        n = len(st.session_state["results"])
        st.caption(f"{n} trained model{'s' if n != 1 else ''} this session")


def header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def require_data() -> pd.DataFrame:
    df = st.session_state["df"]
    if df is None:
        st.warning("No dataset loaded yet. Upload a CSV or load a sample dataset first.")
        st.page_link("pages/1_Dataset_Upload.py", label="Go to Dataset Upload", icon="📤")
        st.stop()
    return df


def set_dataset(df: pd.DataFrame, name: str) -> None:
    """Install a new dataset and discard everything derived from the previous one."""
    st.session_state.update({
        "df_raw": df.copy(), "df": df.copy(), "dataset_name": name, "cleaning_log": [],
        "results": {}, "cluster_run": None, "cluster_df": None, "cluster_info": None,
        "anomaly_df": None, "prep_cfg": dict(DEFAULT_PREP)})


def download_csv(label: str, df: pd.DataFrame, filename: str, key: str | None = None) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename,
                       mime="text/csv", key=key)


def fmt_metric(v, digits: int = 4) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"
