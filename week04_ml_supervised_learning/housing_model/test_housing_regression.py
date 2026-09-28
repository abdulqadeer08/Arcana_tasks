# ---------------------------------------------------------------------------
# AUTOMATED TESTS FOR housing_regression.py
#
# Run with:
#     pytest week04_ml_supervised_learning/housing_model/test_housing_regression.py -v
# ---------------------------------------------------------------------------
import sys
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import housing_regression as hr


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def raw_df():
    return hr.load_data()


@pytest.fixture(scope="module")
def encoded_df(raw_df):
    return hr.encode_features(raw_df)


@pytest.fixture(scope="module")
def prepared(encoded_df):
    return hr.prepare(encoded_df)


@pytest.fixture(scope="session")
def run_full_script():
    """Run the script exactly as a user would, end to end, in a subprocess."""
    script = hr.BASE_DIR / "housing_regression.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(hr.BASE_DIR),
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


# ---------------------------------------------------------------------------
# STEP 1 : DATA LOADING
# ---------------------------------------------------------------------------
def test_csv_loads_with_expected_shape(raw_df):
    assert raw_df.shape[0] > 0
    assert raw_df.shape[1] == 13
    assert "price" in raw_df.columns
    assert raw_df.isnull().sum().sum() == 0
    assert (raw_df["price"] > 0).all()
    assert (raw_df["area"] > 0).all()


# ---------------------------------------------------------------------------
# STEP 2 : ENCODING
# ---------------------------------------------------------------------------
def test_encoding_produces_all_numeric_columns(encoded_df):
    assert all(pd.api.types.is_numeric_dtype(encoded_df[c]) for c in encoded_df.columns)
    assert encoded_df.isnull().sum().sum() == 0


def test_binary_columns_are_0_or_1(encoded_df):
    for col in hr.BINARY_COLS:
        assert set(encoded_df[col].unique()) <= {0, 1}


def test_furnishingstatus_one_hot_encoded_with_first_level_dropped(encoded_df):
    assert "furnishingstatus" not in encoded_df.columns
    assert "furnishingstatus_semi-furnished" in encoded_df.columns
    assert "furnishingstatus_unfurnished" in encoded_df.columns
    assert "furnishingstatus_furnished" not in encoded_df.columns


# ---------------------------------------------------------------------------
# STEP 3 : SCALING AND SPLIT
# ---------------------------------------------------------------------------
def test_scaling_and_split(prepared):
    X, y, scaler_x, scaler_y, X_train, X_test, y_train, y_test = prepared

    assert X_train.shape[0] + X_test.shape[0] == len(X)
    assert X_train.shape[1] == X.shape[1] == X_test.shape[1]
    assert X_train.min() >= -1e-9 and X_train.max() <= 1 + 1e-9
    assert y_train.min() >= -1e-9 and y_train.max() <= 1 + 1e-9

    expected_test_size = round(len(X) * hr.TEST_SIZE)
    assert abs(X_test.shape[0] - expected_test_size) <= 1


# ---------------------------------------------------------------------------
# STEP 4 : MODEL TRAINING
# ---------------------------------------------------------------------------
def test_models_train_and_produce_sane_metrics(prepared):
    X, y, scaler_x, scaler_y, X_train, X_test, y_train, y_test = prepared
    results, predictions, y_test_price, ridge, best_alpha = hr.train_models(
        X, scaler_y, X_train, X_test, y_train, y_test)

    assert len(results) == 4
    for row in results:
        assert np.isfinite(row["R2 Score"])
        assert row["RMSE"] >= 0
        assert row["MAE"] >= 0

    results_df = pd.DataFrame(results).set_index("Model")

    # on this dataset, Linear/Ridge should explain a meaningful share of price variance
    assert results_df.loc["Linear Regression", "R2 Score"] > 0.5
    assert results_df.loc["Ridge Regression (L2)", "R2 Score"] > 0.5

    # regularization should not hurt Ridge noticeably compared to plain Linear
    assert results_df.loc["Ridge Regression (L2)", "R2 Score"] >= \
        results_df.loc["Linear Regression", "R2 Score"] - 0.01

    # every model's predictions should be finite, positive prices matching the test set size
    for name, preds in predictions.items():
        assert preds.shape[0] == y_test_price.shape[0], name
        assert np.all(np.isfinite(preds)), name
        assert np.all(preds > 0), name


def test_polynomial_model_overfits_relative_to_linear(prepared):
    """Documents the overfitting behaviour the script itself reports."""
    X, y, scaler_x, scaler_y, X_train, X_test, y_train, y_test = prepared
    results, *_ = hr.train_models(X, scaler_y, X_train, X_test, y_train, y_test)
    results_df = pd.DataFrame(results).set_index("Model")

    assert results_df.loc["Polynomial Regression (degree 2)", "R2 Score"] < \
        results_df.loc["Linear Regression", "R2 Score"]


# ---------------------------------------------------------------------------
# FULL PIPELINE (subprocess, exactly as a user would run it)
# ---------------------------------------------------------------------------
def test_full_script_runs_end_to_end(run_full_script):
    assert "DONE - ALL STEPS COMPLETED SUCCESSFULLY" in run_full_script.stdout
    assert "Traceback" not in run_full_script.stderr

    assert hr.MODEL_PATH.exists()

    expected_plots = [
        "01_correlation_heatmap.png",
        "02_price_distribution.png",
        "03_actual_vs_predicted_linear_regression.png",
        "04_actual_vs_predicted_ridge_regression.png",
        "05_actual_vs_predicted_lasso_regression.png",
        "06_actual_vs_predicted_polynomial_regression.png",
        "07_model_comparison.png",
    ]
    for name in expected_plots:
        plot_path = hr.PLOTS_DIR / name
        assert plot_path.exists(), name
        assert plot_path.stat().st_size > 0, name


# ---------------------------------------------------------------------------
# SAVED MODEL ARTIFACT
# ---------------------------------------------------------------------------
def test_saved_model_artifact_is_usable(run_full_script, raw_df):
    artifact = joblib.load(hr.MODEL_PATH)

    for key in ("model", "scaler_x", "scaler_y", "feature_names", "best_alpha", "metrics"):
        assert key in artifact

    feature_names = artifact["feature_names"]
    new_house = pd.DataFrame([{
        "area": 7420, "bedrooms": 4, "bathrooms": 2, "stories": 3,
        "mainroad": 1, "guestroom": 0, "basement": 0, "hotwaterheating": 0,
        "airconditioning": 1, "parking": 2, "prefarea": 1,
        "furnishingstatus_semi-furnished": 0, "furnishingstatus_unfurnished": 0,
    }])[feature_names]

    scaled = artifact["scaler_x"].transform(new_house)
    price = artifact["scaler_y"].inverse_transform(
        np.asarray(artifact["model"].predict(scaled)).reshape(-1, 1))[0][0]

    assert np.isfinite(price)
    assert 0 < price < raw_df["price"].max() * 2


def test_saved_model_is_competitive_with_the_best_deployable_alternative(run_full_script, prepared):
    """The script always persists the Ridge model (a deliberate, regularized
    default). This guards against a future change silently making that choice
    a bad one: Ridge's test R2 must stay close to whichever of the simple,
    directly-deployable models (Linear/Ridge/Lasso) scores best. Polynomial is
    excluded because it needs an extra feature-transform step before it could
    ever be deployed the same way, and is already shown to overfit."""
    artifact = joblib.load(hr.MODEL_PATH)
    saved_r2 = artifact["metrics"]["r2"]

    X, y, scaler_x, scaler_y, X_train, X_test, y_train, y_test = prepared
    results, *_ = hr.train_models(X, scaler_y, X_train, X_test, y_train, y_test)

    deployable = [r for r in results if r["Model"] != "Polynomial Regression (degree 2)"]
    best_deployable_r2 = max(r["R2 Score"] for r in deployable)

    assert saved_r2 >= best_deployable_r2 - 0.01
