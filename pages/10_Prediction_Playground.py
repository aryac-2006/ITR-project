import io

import joblib
import pandas as pd
import streamlit as st

from utils import plots
from utils.models import make_bundle, out_of_range_count, predict_dataframe
from utils.ui import download_csv, header, setup_page, show

setup_page("Prediction Playground", "🎮")
header("🎮 Prediction Playground", "Use a trained pipeline on new inputs - preprocessing is applied automatically.")

results = st.session_state["results"]
source = st.radio("Model source", ["Trained in this session", "Upload a saved .joblib pipeline"], horizontal=True)
bundle = None
if source.startswith("Trained"):
    if not results:
        st.warning("No trained models yet. Train one on the Classification or Regression page.")
        st.stop()
    rid = st.selectbox("Model", list(results), format_func=lambda i: (
        f"{results[i]['record']['model']} · {results[i]['record']['target']} · #{i}"))
    bundle = make_bundle(results[rid]["record"], results[rid]["artifact"])
else:
    st.warning("Only load .joblib files you created yourself or fully trust: they are Python pickles "
               "and can execute code when opened.")
    up = st.file_uploader("Saved pipeline (.joblib)", type=["joblib"])
    if up is None:
        st.stop()
    try:
        bundle = joblib.load(io.BytesIO(up.getvalue()))
        if not isinstance(bundle, dict) or not {"pipeline", "features", "schema", "task"} <= set(bundle):
            raise ValueError("This is not an IntelliData AI model file.")
    except Exception as exc:
        st.error(f"Could not load the model: {exc}")
        st.stop()

feats, schema, task = bundle["features"], bundle["schema"], bundle["task"]
st.markdown(f"**{bundle['model']}** · {task} · predicts `{bundle['target']}` from {len(feats)} feature(s)")

tab1, tab2 = st.tabs(["Single prediction", "Batch prediction (CSV)"])
with tab1:
    row, cols = {}, st.columns(3)
    for i, f in enumerate(feats):
        meta = schema[f]
        with cols[i % 3]:
            if meta["kind"] == "numeric":
                row[f] = st.number_input(f, value=float(meta["median"]),
                                         help=f"Training range: {meta['min']:.4g} to {meta['max']:.4g}")
            else:
                choices = meta["choices"] or [""]
                row[f] = st.selectbox(f, choices, index=choices.index(meta["default"]) if meta["default"] in choices else 0)
    if st.button("Predict", type="primary"):
        new = pd.DataFrame([row])
        try:
            res = predict_dataframe(bundle, new)
            if out_of_range_count(bundle, new):
                st.warning("Some values are outside the range seen in training; the prediction is an extrapolation.")
            if task == "classification":
                st.success(f"Predicted **{bundle['target']}**: **{res['prediction'].iloc[0]}**")
                prob_cols = [c for c in res.columns if c.startswith("P(")]
                if prob_cols:
                    p = res[prob_cols].iloc[0]
                    show(plots.fig_bar([c[2:-1] for c in prob_cols], p.to_numpy(), "Class probabilities", "probability"))
            else:
                st.success(f"Predicted **{bundle['target']}**: **{res['prediction'].iloc[0]:.4f}**")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with tab2:
    template = pd.DataFrame([{f: (schema[f]["median"] if schema[f]["kind"] == "numeric" else schema[f]["default"])
                              for f in feats}])
    download_csv("⬇️ Download CSV template", template, "prediction_template.csv")
    file = st.file_uploader("CSV with the feature columns", type=["csv"], key="batch")
    if file is not None:
        try:
            new = pd.read_csv(io.BytesIO(file.getvalue()), sep=None, engine="python")
            if len(new) > 50_000:
                st.warning("Only the first 50,000 rows are scored.")
                new = new.head(50_000)
            res = predict_dataframe(bundle, new)
            bad = out_of_range_count(bundle, new)
            if bad:
                st.warning(f"{bad} numeric value(s) fall outside the training range.")
            st.dataframe(res.head(200))
            download_csv("⬇️ Download predictions", res, "predictions.csv")
        except Exception as exc:
            st.error(f"Could not score the file: {exc}")
