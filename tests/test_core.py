"""Unit tests for the Streamlit-free core logic. Run with:  pytest -q"""
import io
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from utils import unsupervised as un  # noqa: E402
from utils.data import clean_dataframe, load_csv, load_sample, quality_summary  # noqa: E402
from utils.models import (CLASSIFIERS, REGRESSORS, bundle_to_bytes, make_bundle,  # noqa: E402
                          predict_dataframe, train_model, validate_target)
from utils.preprocessing import IQRClipper, make_preprocessor  # noqa: E402

CUST = load_sample("customers").drop_duplicates()
FEATS = ["age", "income", "tenure_months", "plan", "region", "support_calls"]


class Upload:
    def __init__(self, raw: bytes):
        self.raw = raw

    def getvalue(self):
        return self.raw


def test_quality_and_cleaning():
    df = load_sample("customers")
    q = quality_summary(df)
    assert q["duplicates"] == 20 and q["missing_cells"] > 0
    cleaned, log = clean_dataframe(df, drop_duplicates=True)
    assert cleaned.duplicated().sum() == 0 and "duplicate" in log[0]


def test_load_csv_detects_delimiter_and_rejects_bad_files():
    raw = "a;b\n" + "\n".join(f"{i};{i * 2}" for i in range(20))
    assert load_csv(Upload(raw.encode())).shape == (20, 2)
    with pytest.raises(ValueError):
        load_csv(Upload(b""))
    with pytest.raises(ValueError):
        load_csv(Upload(b"a,b\n1,2\n"))


def test_no_leakage_in_preprocessing():
    """Scaler statistics must come from the training data only."""
    train, test = CUST.iloc[:300][["age", "income"]], CUST.iloc[300:][["age", "income"]]
    pre = make_preprocessor(["age", "income"], [], {"scaler": "standard"}).fit(train)
    scaler = pre.named_transformers_["num"]["scale"]
    assert np.allclose(scaler.mean_, train.fillna(train.median()).mean().to_numpy(), rtol=0.05)
    assert not np.allclose(scaler.mean_, test.fillna(test.median()).mean().to_numpy(), rtol=1e-6)


def test_iqr_clipper_bounds_learned_on_fit():
    c = IQRClipper(1.5).fit(np.arange(100.0).reshape(-1, 1))
    assert c.transform(np.array([[1000.0]]))[0, 0] == pytest.approx(c.upper_[0])


@pytest.mark.parametrize("name", list(CLASSIFIERS))
def test_every_classifier_trains(name):
    out = train_model(CUST, FEATS, "churn", "classification", name, cv=3)
    m = out["record"]["metrics"]
    assert 0 <= m["accuracy"] <= 1 and m["roc_auc"] is not None


@pytest.mark.parametrize("name", list(REGRESSORS))
def test_every_regressor_trains(name):
    out = train_model(CUST, FEATS, "monthly_charges", "regression", name, cv=3)
    assert out["record"]["metrics"]["rmse"] > 0


def test_tuning_and_saved_pipeline_roundtrip():
    iris = load_sample("iris")
    feats = list(iris.columns[:4])
    out = train_model(iris, feats, "species", "classification", "K-Nearest Neighbors", tune="grid", cv=3)
    assert out["record"]["best_params"]
    bundle = joblib.load(io.BytesIO(bundle_to_bytes(make_bundle(out["record"], out["artifact"]))))
    pred = predict_dataframe(bundle, iris.head(5))
    assert list(pred["prediction"]) == ["setosa"] * 5


def test_incompatible_targets_are_rejected():
    assert validate_target(CUST, "monthly_charges", "classification")[0]
    assert validate_target(CUST, "plan", "regression")[0]
    assert not validate_target(CUST, "churn", "classification")[0]


def test_clustering_pca_and_anomaly():
    wine = load_sample("wine")
    cols = [c for c in wine.columns if c != "wine_class"]
    X, names = un.build_matrix(wine, cols)
    labels = un.run_kmeans(X, 3)
    assert un.cluster_quality(X, labels)["silhouette"] > 0.2
    pca, Z = un.run_pca(X, 3)
    assert Z.shape == (len(wine), 3) and pca.explained_variance_ratio_.sum() < 1
    flag, score = un.run_isolation_forest(X, 0.05)
    assert 0 < flag.sum() < len(X) and len(score) == len(X)
    assert un.safe_silhouette(X, np.zeros(len(X))) is None  # single cluster -> undefined
