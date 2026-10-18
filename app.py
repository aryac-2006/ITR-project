"""IntelliData AI - home dashboard. Run with:  streamlit run app.py"""
import streamlit as st

from utils.ui import setup_page

setup_page("Home", "🧠")

st.markdown("""
<div class="hero">
  <h1>🧠 IntelliData AI</h1>
  <p>Intelligent Data Analysis and Machine Learning Platform — upload data, explore it, preprocess it
  without leakage, train and compare models, discover clusters and anomalies, explain predictions
  and download everything.</p>
</div>
""", unsafe_allow_html=True)

df, results = st.session_state["df"], st.session_state["results"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset", st.session_state["dataset_name"] or "—")
c2.metric("Rows × columns", f"{df.shape[0]:,} × {df.shape[1]}" if df is not None else "—")
c3.metric("Trained models", len(results))
c4.metric("Clustering / anomaly runs",
          int(st.session_state["cluster_run"] is not None) + int(st.session_state["anomaly_df"] is not None))

st.subheader("Workflow")
cards = [
    ("1 · Data", "Upload a CSV or load a sample, review quality, explore distributions and correlations.",
     "pages/1_Dataset_Upload.py", "📤 Dataset Upload"),
    ("2 · Prepare", "Clean duplicates, then configure imputation, encoding, scaling, outlier clipping and "
     "feature selection inside a leakage-safe pipeline.", "pages/3_Preprocessing.py", "🧹 Preprocessing"),
    ("3 · Supervised ML", "6 classifiers and 5 regressors with train/test split, cross-validation, "
     "GridSearchCV / RandomizedSearchCV and full metrics.", "pages/4_Classification.py", "🎯 Classification"),
    ("4 · Unsupervised ML", "K-Means, Agglomerative, DBSCAN and PCA with silhouette scores; "
     "Isolation Forest for anomalies.", "pages/6_Clustering_PCA.py", "🔵 Clustering & PCA"),
    ("5 · Understand", "Leaderboard, feature importance, optional SHAP, and a prediction playground "
     "using the saved pipeline.", "pages/7_Model_Comparison.py", "🏆 Model Comparison"),
    ("6 · Export", "Download cleaned data, metrics, clusters, anomaly reports, model files and a summary.",
     "pages/11_Reports.py", "📦 Reports"),
]
for row in (cards[:3], cards[3:]):
    for col, (title, text, page, label) in zip(st.columns(3), row):
        with col:
            st.markdown(f'<div class="card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)
            st.page_link(page, label=label)

st.subheader("Quick start")
st.markdown("""
1. Open **Dataset Upload** and load a sample (e.g. *Synthetic customers*) or upload your own CSV.
2. Optionally visit **Preprocessing** to clean the data and configure the pipeline.
3. Train models on **Classification** or **Regression**, then compare them on **Model Comparison**.
4. Try **Clustering & PCA** or **Anomaly Detection** — no target column needed.
5. Open **Model Explanations**, test inputs in the **Prediction Playground**, and export from **Reports**.
""")
st.info("All results are computed live from your data — nothing is pre-baked. Large files are "
        "down-sampled to 50,000 rows to keep the app responsive on free cloud hosting.")
