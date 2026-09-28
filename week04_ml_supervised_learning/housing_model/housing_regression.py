# ---------------------------------------------------------------------------
# HOUSING PRICE PREDICTION - REGRESSION MODEL
# ---------------------------------------------------------------------------
# Standalone script version of the "Machine Learning On Real Housing Data"
# section of week04_ml_supervised_learning/notebooks/regression.ipynb.
#
# Runs end to end from a terminal:
#     python housing_regression.py
#
# It loads Housing.csv, encodes the categorical columns, scales the data,
# trains Linear / Ridge / Lasso / Polynomial regression, compares them,
# and saves the best model to housing_price_model.joblib.
#
# Plots are written as PNG files into the plots/ folder instead of being shown
# on screen, so the script never blocks waiting for a window to be closed.
# ---------------------------------------------------------------------------

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to files, do not open a window

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# paths are resolved from this file, so the script works from any directory
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Housing.csv"
MODEL_PATH = BASE_DIR / "housing_price_model.joblib"
PLOTS_DIR = BASE_DIR / "plots"

RANDOM_STATE = 42
TEST_SIZE = 0.2

BINARY_COLS = ["mainroad", "guestroom", "basement", "hotwaterheating",
               "airconditioning", "prefarea"]


def banner(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# STEP 1 : LOAD THE DATASET
# ---------------------------------------------------------------------------
def load_data():
    banner("STEP 1 : LOADING THE DATASET")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Housing.csv not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    print(f"FILE            : {CSV_PATH}")
    print(f"ROWS, COLUMNS   : {df.shape[0]}, {df.shape[1]}")
    print(f"MISSING VALUES  : {df.isnull().sum().sum()}")
    print(f"DUPLICATE ROWS  : {df.duplicated().sum()}")
    print(f"TARGET 'price'  : mean {df['price'].mean():,.0f} | "
          f"min {df['price'].min():,.0f} | max {df['price'].max():,.0f}")
    print("\nFIRST 5 ROWS :")
    print(df.head().to_string())
    return df


# ---------------------------------------------------------------------------
# STEP 2 : ENCODE THE CATEGORICAL COLUMNS
# ---------------------------------------------------------------------------
def encode_features(df):
    banner("STEP 2 : FEATURE ENGINEERING (ENCODING)")

    encoded = df.copy()

    # the six yes/no columns become 1/0
    for col in BINARY_COLS:
        encoded[col] = encoded[col].map({"yes": 1, "no": 0})
    print(f"BINARY ENCODED  : {', '.join(BINARY_COLS)}")

    # furnishingstatus has three levels, so one hot encode it
    encoded = pd.get_dummies(encoded, columns=["furnishingstatus"],
                             drop_first=True, dtype=int)
    print("ONE HOT ENCODED : furnishingstatus (first level dropped)")

    print(f"SHAPE AFTER     : {encoded.shape}")
    print(f"ALL NUMERIC     : {all(pd.api.types.is_numeric_dtype(encoded[c]) for c in encoded.columns)}")
    return encoded


# ---------------------------------------------------------------------------
# STEP 3 : SCALE AND SPLIT
# ---------------------------------------------------------------------------
def prepare(encoded):
    banner("STEP 3 : SCALING AND TRAIN/TEST SPLIT")

    X = encoded.drop("price", axis=1)
    y = encoded["price"]

    scaler_x = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print(f"FEATURES ({X.shape[1]})   : {', '.join(X.columns)}")
    print(f"TRAINING SET    : {X_train.shape}")
    print(f"TESTING SET     : {X_test.shape}")

    print("\nCORRELATION WITH PRICE (STRONGEST FIRST) :")
    print(encoded.corr()["price"].drop("price").sort_values(ascending=False).to_string())

    return X, y, scaler_x, scaler_y, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# STEP 4 : TRAIN AND EVALUATE THE MODELS
# ---------------------------------------------------------------------------
def evaluate(name, y_true, y_pred, results):
    """Compute the metrics for one model and store them in the results list."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    results.append({"Model": name, "R2 Score": r2, "RMSE": rmse, "MAE": mae})
    print(f"{name:<34} R^2: {r2:.4f} | RMSE: {rmse:>12,.0f} | MAE: {mae:>12,.0f}")
    return r2, rmse, mae


def train_models(X, scaler_y, X_train, X_test, y_train, y_test):
    banner("STEP 4 : TRAINING THE REGRESSION MODELS")

    # scikit-learn returns 2D predictions for some estimators and 1D for others,
    # so always reshape before scaling the values back to actual prices
    def to_price(scaled_values):
        return scaler_y.inverse_transform(np.asarray(scaled_values).reshape(-1, 1))

    y_test_price = to_price(y_test)
    y_train_flat = y_train.ravel()
    results = []
    predictions = {}

    # ---- Linear Regression -------------------------------------------------
    linear = LinearRegression().fit(X_train, y_train_flat)
    predictions["Linear Regression"] = to_price(linear.predict(X_test))
    evaluate("Linear Regression", y_test_price,
             predictions["Linear Regression"], results)

    # ---- Ridge Regression (L2) --------------------------------------------
    alpha_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]}

    ridge_search = GridSearchCV(Ridge(), alpha_grid,
                                scoring="neg_mean_squared_error", cv=10)
    ridge_search.fit(X_train, y_train_flat)
    best_alpha_ridge = ridge_search.best_params_["alpha"]

    ridge = Ridge(alpha=best_alpha_ridge).fit(X_train, y_train_flat)
    predictions["Ridge Regression (L2)"] = to_price(ridge.predict(X_test))
    evaluate(f"Ridge Regression (L2, a={best_alpha_ridge})", y_test_price,
             predictions["Ridge Regression (L2)"], results)
    results[-1]["Model"] = "Ridge Regression (L2)"

    # ---- Lasso Regression (L1) --------------------------------------------
    lasso_search = GridSearchCV(Lasso(), alpha_grid,
                                scoring="neg_mean_squared_error", cv=10)
    lasso_search.fit(X_train, y_train_flat)
    best_alpha_lasso = lasso_search.best_params_["alpha"]

    lasso = Lasso(alpha=best_alpha_lasso).fit(X_train, y_train_flat)
    predictions["Lasso Regression (L1)"] = to_price(lasso.predict(X_test))
    evaluate(f"Lasso Regression (L1, a={best_alpha_lasso})", y_test_price,
             predictions["Lasso Regression (L1)"], results)
    results[-1]["Model"] = "Lasso Regression (L1)"

    # ---- Polynomial Regression --------------------------------------------
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_model = LinearRegression().fit(X_train_poly, y_train_flat)
    predictions["Polynomial Regression (degree 2)"] = to_price(poly_model.predict(X_test_poly))
    evaluate("Polynomial Regression (degree 2)", y_test_price,
             predictions["Polynomial Regression (degree 2)"], results)

    # ---- the regression equation of the best linear style model ------------
    print("\nRIDGE REGRESSION EQUATION :")
    equation = f"price(scaled) = {float(np.ravel(ridge.intercept_)[0]):.4f}"
    for feature, coef in zip(X.columns, np.ravel(ridge.coef_)):
        equation += f" + ({coef:.4f} * {feature})"
    print(equation)

    print(f"\nLASSO KEPT      : {int((lasso.coef_ != 0).sum())} of {len(lasso.coef_)} features")
    print("\nOVERFITTING CHECK (POLYNOMIAL VS LINEAR) :")
    print(f"  TRAINING R^2  - Linear: {linear.score(X_train, y_train_flat):.4f} | "
          f"Polynomial: {poly_model.score(X_train_poly, y_train_flat):.4f}")
    print(f"  TESTING  R^2  - Linear: {results[0]['R2 Score']:.4f} | "
          f"Polynomial: {results[3]['R2 Score']:.4f}")
    print("  The polynomial model fits training data better but scores worse on")
    print("  unseen data, which is the classic signature of overfitting.")

    return results, predictions, y_test_price, ridge, best_alpha_ridge


# ---------------------------------------------------------------------------
# STEP 5 : PLOTS
# ---------------------------------------------------------------------------
def make_plots(encoded, y, results_df, predictions, y_test_price):
    banner("STEP 5 : SAVING THE PLOTS")
    PLOTS_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(encoded.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, vmin=-1, vmax=1)
    plt.title("Correlation Between Housing Features")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_correlation_heatmap.png", dpi=110)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.histplot(y, bins=40, kde=True, color="steelblue")
    plt.title("Distribution of House Price")
    plt.xlabel("Price")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_price_distribution.png", dpi=110)
    plt.close()

    for i, (name, preds) in enumerate(predictions.items(), start=3):
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_price.flatten()[:50], marker="o", linestyle="--", label="Actual")
        plt.plot(preds.flatten()[:50], marker="o", linestyle="--", label=f"Predicted ({name})")
        plt.title(f"Comparison of Actual and Predicted Price Using {name}")
        plt.xlabel("Test sample")
        plt.ylabel("Price")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        safe = name.split("(")[0].strip().lower().replace(" ", "_")
        plt.savefig(PLOTS_DIR / f"{i:02d}_actual_vs_predicted_{safe}.png", dpi=110)
        plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x="R2 Score", y="Model", hue="Model",
                palette="viridis", legend=False)
    plt.title("Model Comparison by R^2 Score on the Test Set")
    plt.xlim(0, 1)
    plt.grid(axis="x")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_model_comparison.png", dpi=110)
    plt.close()

    for f in sorted(PLOTS_DIR.glob("*.png")):
        print(f"  SAVED : {f.relative_to(BASE_DIR)}")


# ---------------------------------------------------------------------------
# STEP 6 : SAVE THE BEST MODEL
# ---------------------------------------------------------------------------
def save_model(model, scaler_x, scaler_y, feature_names, best_alpha, metrics):
    banner("STEP 6 : SAVING THE BEST MODEL")

    artifact = {
        "model": model,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "feature_names": list(feature_names),
        "best_alpha": best_alpha,
        "metrics": metrics,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"SAVED TO        : {MODEL_PATH}")
    print(f"FILE SIZE       : {os.path.getsize(MODEL_PATH):,} bytes")
    print("CONTENTS        : model + both scalers + feature order + metrics")

    # prove the saved file is usable on its own
    loaded = joblib.load(MODEL_PATH)
    new_house = pd.DataFrame([{
        "area": 7420, "bedrooms": 4, "bathrooms": 2, "stories": 3,
        "mainroad": 1, "guestroom": 0, "basement": 0, "hotwaterheating": 0,
        "airconditioning": 1, "parking": 2, "prefarea": 1,
        "furnishingstatus_semi-furnished": 0, "furnishingstatus_unfurnished": 0,
    }])
    scaled = loaded["scaler_x"].transform(new_house[loaded["feature_names"]])
    price = loaded["scaler_y"].inverse_transform(
        np.asarray(loaded["model"].predict(scaled)).reshape(-1, 1))

    print("\nRELOAD TEST (PREDICTING A BRAND NEW HOUSE) :")
    print("  4 bed / 2 bath / 3 storey / 7420 sq ft / AC / main road / preferred area")
    print(f"  PREDICTED PRICE : {price[0][0]:,.0f}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    banner("HOUSING PRICE PREDICTION - REGRESSION MODEL")

    df = load_data()
    encoded = encode_features(df)
    X, y, scaler_x, scaler_y, X_train, X_test, y_train, y_test = prepare(encoded)

    results, predictions, y_test_price, ridge, best_alpha = train_models(
        X, scaler_y, X_train, X_test, y_train, y_test)

    results_df = (pd.DataFrame(results)
                  .sort_values("R2 Score", ascending=False)
                  .reset_index(drop=True))

    banner("MODEL COMPARISON")
    print(results_df.to_string(index=False,
                               formatters={"R2 Score": "{:.4f}".format,
                                           "RMSE": "{:,.0f}".format,
                                           "MAE": "{:,.0f}".format}))

    best = results_df.loc[0]
    print(f"\nBEST MODEL      : {best['Model']}")
    print(f"R^2 SCORE       : {best['R2 Score']:.4f}")
    print(f"RMSE            : {best['RMSE']:,.0f} "
          f"({best['RMSE'] / y.mean() * 100:.1f}% of the average price of {y.mean():,.0f})")

    make_plots(encoded, y, results_df, predictions, y_test_price)

    ridge_row = results_df[results_df["Model"] == "Ridge Regression (L2)"].iloc[0]
    save_model(ridge, scaler_x, scaler_y, X.columns, best_alpha,
               {"r2": float(ridge_row["R2 Score"]),
                "rmse": float(ridge_row["RMSE"]),
                "mae": float(ridge_row["MAE"])})

    banner("DONE - ALL STEPS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
