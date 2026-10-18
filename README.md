# 🧠 IntelliData AI — Intelligent Data Analysis and Machine Learning Platform

An end-to-end, interactive machine-learning workbench built with **Python and Streamlit**. Upload a CSV
(or load a bundled sample), explore it, preprocess it without data leakage, train and compare supervised
models, discover clusters and anomalies, explain predictions, and download everything.

Every number shown is computed live from your data — there are no hard-coded metrics or fake results.

## Features

| Area | What it does |
|---|---|
| **Dataset upload** | CSV upload (auto-detects delimiter/encoding) or sample datasets; shape, dtypes, missing values, duplicates, descriptive statistics, data-quality score |
| **Exploratory analysis** | Missing-value chart, histograms/box plots, category counts, correlation heatmap (Pearson/Spearman) and strongest pairs, scatter plots |
| **Preprocessing** | Duplicate removal, column dropping, median/mean/mode/KNN imputation, one-hot/ordinal encoding, standard/min-max/robust scaling, IQR outlier analysis and clipping, `SelectKBest`/variance feature selection, mutual-information ranking |
| **Classification** | Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, SVM |
| **Regression** | Linear, Ridge, Decision Tree, Random Forest, SVR |
| **Evaluation** | Stratified train/test split, k-fold cross-validation, optional `GridSearchCV` / `RandomizedSearchCV`; accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve (classification); MAE, RMSE, R², residual plots (regression) |
| **Clustering & PCA** | K-Means (elbow/silhouette sweep), Agglomerative (+ dendrogram), DBSCAN (+ k-distance plot), PCA with loadings; silhouette, Davies-Bouldin, cluster profiles |
| **Anomaly detection** | Isolation Forest with score distribution, PCA view, top anomalies and normal-vs-anomaly profile |
| **Model comparison** | Leaderboard sortable by any metric, CV-vs-test chart, over-fitting flag |
| **Explanations** | Impurity / coefficient / permutation importance; optional SHAP |
| **Prediction playground** | Single-row form and batch CSV scoring using the saved full pipeline |
| **Reports** | Download cleaned data, metrics, cluster labels, anomaly report, `.joblib` pipeline and a Markdown summary |

### Safeguards against misuse
- A numeric/continuous target is **blocked** on the Classification page; a text target is **blocked** on the Regression page, each with an explanation.
- Classes with a single row, single-valued targets and datasets with fewer than 30 labelled rows are rejected.
- Date columns and very high-cardinality text columns are excluded from features automatically; ID-like/constant columns are flagged.
- Class imbalance triggers a warning to look beyond accuracy.

### How data leakage is prevented
Imputation values, encoders, scalers, outlier bounds and feature selection are all steps of one scikit-learn
`Pipeline` (`preprocess → select → model`). The pipeline is fitted only on the training split, re-fitted inside every
cross-validation fold, and hyper-parameter search runs on the training split only. The test set is used once, for the
final metrics. The saved `.joblib` file contains the *entire* fitted pipeline, so new data is transformed exactly as
training data was.

## Project structure

```
app.py                     Home dashboard
pages/                     One file per Streamlit page (1_Dataset_Upload … 11_Reports)
utils/
  data.py                  Loading, typing, quality summary, cleaning, sample data
  preprocessing.py         Pipeline builder, IQR clipper, outlier report, feature ranking
  models.py                Model registry, validation, training/tuning/evaluation, save & predict
  unsupervised.py          Clustering, PCA, Isolation Forest
  explain.py               Feature importance, permutation importance, optional SHAP
  plots.py                 Dark-themed matplotlib/seaborn figures
  reports.py               Summary text and metrics table
  supervised_ui.py         Shared body of the Classification/Regression pages
  ui.py                    Page setup, theme CSS, session helpers
tests/test_core.py         Unit tests for the non-UI logic
.streamlit/config.toml     Dark theme and upload limit
requirements.txt           Dependencies (minimum versions, latest installed)
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Works on Python 3.10 – 3.14. Optional SHAP support: `pip install shap` (may not yet support the newest Python).

Run the tests: `pip install pytest && pytest -q`

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and push this project (the `.gitignore` already excludes virtual environments,
   caches, secrets and generated model files):
   ```bash
   git init
   git add .
   git commit -m "IntelliData AI"
   git branch -M main
   git remote add origin https://github.com/<your-username>/intellidata-ai.git
   git push -u origin main
   ```
2. Go to <https://share.streamlit.io>, sign in with GitHub and click **Create app**.
3. Choose your repository, branch `main`, and main file path `app.py`.
4. Open **Advanced settings** and select a recent Python version (3.12 or 3.13 is safest).
5. Click **Deploy**. The first build installs the dependencies and takes a few minutes.
6. *(Optional)* To enable SHAP, uncomment the `shap` line in `requirements.txt`, commit and push.

No secrets or API keys are needed.

## Performance notes (free hosting)
- Uploads are limited to 50 MB and down-sampled to 50,000 rows.
- SVM/SVR training is capped at 10,000 training rows; hierarchical clustering at 5,000 rows.
- Hyper-parameter search multiplies runtime — keep the number of models and iterations small on shared CPUs.
- Session data lives in memory and is lost when the browser session ends; download your outputs.

## Security note
`.joblib` files are Python pickles. The Prediction Playground can load an uploaded model file, so only open files
you created or trust.

## Suggested demo script
1. **Dataset Upload** → load *Synthetic customers* (has missing values, outliers, duplicates, categoricals).
2. **Preprocessing** → apply cleaning, enable IQR clipping, review the outlier table.
3. **Classification** → target `churn`, train all six models, tick tuning for one.
4. **Model Comparison** → rank by F1/ROC-AUC; **Explanations** → importance.
5. **Regression** → target `monthly_charges`.
6. **Clustering & PCA**, **Anomaly Detection** on the same data.
7. **Prediction Playground** → score a new customer; **Reports** → download the summary.
