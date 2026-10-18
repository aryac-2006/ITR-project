"""Shared page body for Classification and Regression (identical workflow, different task)."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report

from utils import plots
from utils.data import split_columns, suggest_drop_columns
from utils.models import (bundle_to_bytes, leaderboard, make_bundle, registry, train_model,
                          usable_features, validate_target)
from utils.ui import fmt_metric, header, require_data, setup_page, show


def run(task: str) -> None:
    is_clf = task == "classification"
    setup_page("Classification" if is_clf else "Regression", "🎯" if is_clf else "📈")
    header("🎯 Classification" if is_clf else "📈 Regression",
           "Predict a category from the other columns." if is_clf
           else "Predict a number from the other columns.")
    df = require_data()
    results, cfg = st.session_state["results"], st.session_state["prep_cfg"]
    cols = list(df.columns)

    # ---- 1. problem definition --------------------------------------------------------
    st.subheader("1 · Define the problem")
    num_cols, _, _ = split_columns(df)
    # sensible default: last column for classification; last numeric column with many distinct values for regression
    cont = [c for c in num_cols if df[c].nunique() > 10]
    default_idx = len(cols) - 1 if is_clf or not (cont or num_cols) else cols.index((cont or num_cols)[-1])
    target = st.selectbox("Target column (what to predict)", cols, index=default_idx)
    errors, warnings = validate_target(df, target, task)
    for w in warnings:
        st.warning(w)
    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    options = [c for c in cols if c != target]
    ignore = suggest_drop_columns(df)
    features = st.multiselect("Feature columns", options,
                              default=[c for c in options if c not in ignore])
    feats, notes = usable_features(df, features)
    for n in notes:
        st.info(n)
    for c, why in ignore.items():
        if c in features:
            st.warning(f"'{c}' {why} - consider removing it from the features.")
    if not feats:
        st.error("Select at least one usable feature column.")
        st.stop()

    # ---- 2. training options ----------------------------------------------------------
    st.subheader("2 · Choose models and evaluation")
    names = list(registry(task))
    models = st.multiselect("Algorithms to train and compare", names, default=names[:3])
    c1, c2, c3, c4 = st.columns(4)
    test_size = c1.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    cv = c2.slider("Cross-validation folds", 3, 10, 5)
    seed = int(c3.number_input("Random seed", 0, 9999, 42))
    tune_label = c4.selectbox("Hyper-parameter tuning", ["None", "GridSearchCV", "RandomizedSearchCV"])
    tune = {"None": "none", "GridSearchCV": "grid", "RandomizedSearchCV": "random"}[tune_label]
    n_iter = 10
    if tune == "random":
        n_iter = st.slider("Random search iterations", 3, 30, 10)
    if tune != "none":
        st.caption("Tuning multiplies training time. Cross-validation runs on the training split only; "
                   "the test set is never used for tuning.")
    with st.expander("Preprocessing pipeline that will be applied (configure on the Preprocessing page)"):
        st.json(cfg)
        st.caption("Fitted inside every cross-validation fold and on the training split only, so no "
                   "information from validation/test rows leaks into training.")

    if st.button("🚀 Train selected models", type="primary", disabled=not models):
        bar = st.progress(0.0, text="Starting…")
        for i, m in enumerate(models):
            bar.progress(i / len(models), text=f"Training {m} ({i + 1}/{len(models)})…")
            try:
                out = train_model(df, feats, target, task, m, cfg, test_size, cv, tune, n_iter,
                                  seed, st.session_state["dataset_name"])
                results[out["record"]["id"]] = out
            except Exception as exc:  # show the failure but keep training the others
                st.error(f"**{m}** failed: {exc}")
        bar.progress(1.0, text="Finished")

    # ---- 3. results -------------------------------------------------------------------
    runs = [v for v in results.values() if v["record"]["task"] == task
            and v["record"]["target"] == target
            and v["record"]["dataset"] == st.session_state["dataset_name"]]
    if not runs:
        st.info("Train at least one model to see results.")
        return
    st.subheader("3 · Results")
    lb = leaderboard([r["record"] for r in runs], task)
    st.dataframe(lb, hide_index=True)
    st.caption("Metrics are computed on the held-out test set. `cv_mean` is the cross-validated "
               f"{'F1 (weighted)' if is_clf else 'R²'} on the training split. A large gap between "
               "`train_score` and `cv_mean` signals over-fitting.")

    st.subheader("4 · Inspect a model")
    by_id = {r["record"]["id"]: r for r in runs}
    run_id = st.selectbox("Trained model", list(lb["run"]),
                          format_func=lambda i: f"#{lb.index[lb['run'] == i][0] + 1}  {by_id[i]['record']['model']}"
                                                f"{' (tuned)' if by_id[i]['record']['tuned'] else ''}")
    rec, art = by_id[run_id]["record"], by_id[run_id]["artifact"]
    if rec["note"]:
        st.info(rec["note"])
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"] if is_clf else ["mae", "rmse", "r2"]
    for col, k in zip(st.columns(len(keys)), keys):
        col.metric(k.upper().replace("_", "-"), fmt_metric(rec["metrics"][k]))
    if rec["best_params"]:
        st.write("**Best hyper-parameters:**", rec["best_params"])
    st.caption(f"Train {rec['n_train']:,} rows · test {rec['n_test']:,} rows · {len(rec['features'])} input "
               f"features · {rec['cv_folds']}-fold CV {fmt_metric(rec['cv_mean'], 3)} ± "
               f"{fmt_metric(rec['cv_std'], 3)} · {rec['train_seconds']}s")

    left, right = st.columns(2)
    with left:
        if is_clf:
            show(plots.fig_confusion(art["cm"], rec["classes"]))
        else:
            show(plots.fig_pred_vs_actual(art["y_test"], art["y_pred"]))
    with right:
        if is_clf and art["proba"] is not None:
            show(plots.fig_roc(art["y_test"], art["proba"], rec["classes"]))
        elif not is_clf:
            show(plots.fig_residuals(art["y_test"], art["y_pred"]))
    if is_clf:
        rep = classification_report(art["y_test"], art["y_pred"], target_names=rec["classes"],
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3))
        if len(rec["classes"]) == 2:
            st.caption(f"Binary metrics treat **{rec['classes'][1]}** as the positive class.")

    st.download_button("💾 Download trained pipeline (.joblib)",
                       bundle_to_bytes(make_bundle(rec, art)),
                       file_name=f"{rec['model'].lower().replace(' ', '_')}_{rec['id']}.joblib",
                       mime="application/octet-stream",
                       help="Contains the fitted preprocessing + model, ready for new data.")
    if st.button("🗑️ Clear all runs for this target"):
        for r in runs:
            results.pop(r["record"]["id"], None)
        st.rerun()
