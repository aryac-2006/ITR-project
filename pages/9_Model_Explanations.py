import streamlit as st

from utils import plots
from utils.explain import feature_importance, shap_available, shap_importance
from utils.ui import header, setup_page, show

setup_page("Model Explanations", "💡")
header("💡 Model Explanations", "Which inputs drive a model's predictions?")
results = st.session_state["results"]
if not results:
    st.warning("Train a model first (Classification or Regression page).")
    st.stop()

run_id = st.selectbox("Trained model", list(results), format_func=lambda i: (
    f"{results[i]['record']['model']} · {results[i]['record']['target']} · #{i}"))
rec, art = results[run_id]["record"], results[run_id]["artifact"]

tab_fi, tab_shap = st.tabs(["Feature importance", "SHAP (optional)"])
with tab_fi:
    method = st.radio("Method", ["auto", "permutation"], horizontal=True,
                      help="auto: model-native importance when available. permutation: model-agnostic, "
                           "measured on the held-out test set with the original columns.")
    try:
        with st.spinner("Computing importance…"):
            imp, how = feature_importance(art["pipeline"], art["X_test"], art["y_test"], rec["task"], method)
        st.caption(how)
        show(plots.fig_bar(imp["feature"], imp["importance"].to_numpy(), "Feature importance", "importance"))
        st.dataframe(imp.round(4), hide_index=True)
        st.caption("Importance shows association with the model's predictions, not causation. Correlated "
                   "features can share or hide importance.")
    except Exception as exc:
        st.error(f"Could not compute importance: {exc}")

with tab_shap:
    if not shap_available():
        st.info("SHAP is not installed. For local use run `pip install shap`; on Streamlit Community Cloud "
                "uncomment the `shap` line in `requirements.txt` and redeploy.")
    else:
        st.caption("Tree models and linear models use fast exact explainers. Other models use a slower "
                   "sampled KernelExplainer (about 40 rows).")
        if st.button("Compute SHAP values"):
            try:
                with st.spinner("Computing SHAP values…"):
                    table, vals, Xt, names = shap_importance(art["pipeline"], art["X_test"])
                show(plots.fig_bar(table["feature"], table["mean_abs_shap"].to_numpy(),
                                   "Mean |SHAP value|", "mean |SHAP|"))
                try:
                    import matplotlib.pyplot as plt
                    import shap
                    plt.figure()
                    shap.summary_plot(vals, Xt, feature_names=names, show=False)
                    show(plt.gcf())
                except Exception as exc:
                    st.caption(f"Beeswarm plot unavailable: {exc}")
                st.dataframe(table.round(4), hide_index=True)
            except Exception as exc:
                st.error(f"SHAP could not explain this model: {exc}")
