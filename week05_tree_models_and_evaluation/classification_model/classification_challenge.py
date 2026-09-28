# ---------------------------------------------------------------------------
# BREAST CANCER DIAGNOSIS - CLASSIFICATION CHALLENGE
# ---------------------------------------------------------------------------
# Week 5 practical: Decision Trees, Random Forest and XGBoost compared on the
# Breast Cancer Wisconsin (Diagnostic) dataset, with cross-validated
# hyperparameter tuning and a full metric + confusion matrix analysis.
#
# Runs end to end from a terminal:
#     python classification_challenge.py
#
# It loads Breast_Cancer.csv, tunes a Decision Tree / Random Forest / XGBoost
# classifier with cross-validation, evaluates all three on the held-out test
# set (accuracy, precision, recall, F1, ROC-AUC, confusion matrix), and saves
# the best model to breast_cancer_model.joblib.
#
# Plots are written as PNG files into the plots/ folder instead of being
# shown on screen, so the script never blocks waiting for a window to close.
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
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve,
                              confusion_matrix, classification_report)

# paths are resolved from this file, so the script works from any directory
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Breast_Cancer.csv"
MODEL_PATH = BASE_DIR / "breast_cancer_model.joblib"
PLOTS_DIR = BASE_DIR / "plots"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


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
        raise FileNotFoundError(f"Breast_Cancer.csv not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    print(f"FILE            : {CSV_PATH}")
    print(f"ROWS, COLUMNS   : {df.shape[0]}, {df.shape[1]}")
    print(f"MISSING VALUES  : {df.isnull().sum().sum()}")
    print(f"DUPLICATE ROWS  : {df.duplicated().sum()}")

    counts = df["diagnosis"].value_counts()
    print(f"CLASS BALANCE   : B (benign) = {counts['B']} | "
          f"M (malignant) = {counts['M']} "
          f"({counts['M'] / len(df) * 100:.1f}% positive class)")
    print("\nFIRST 5 ROWS :")
    print(df.head().to_string())
    return df


# ---------------------------------------------------------------------------
# STEP 2 : PREPARE FEATURES / TARGET AND SPLIT
# ---------------------------------------------------------------------------
def prepare(df):
    banner("STEP 2 : PREPARING FEATURES AND TRAIN/TEST SPLIT")

    X = df.drop(columns=["id", "diagnosis"])
    y = df["diagnosis"].map({"M": 1, "B": 0})

    # stratify keeps the same malignant/benign ratio in both the train and
    # test sets, which matters on an imbalanced dataset like this one
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    print(f"FEATURES ({X.shape[1]})  : {', '.join(X.columns[:5])}, ...")
    print(f"TRAINING SET    : {X_train.shape} | positive rate "
          f"{y_train.mean() * 100:.1f}%")
    print(f"TESTING SET     : {X_test.shape} | positive rate "
          f"{y_test.mean() * 100:.1f}%")

    # tree-based models split on raw thresholds, so unlike linear models
    # they do not need the features to be scaled first
    print("\nNOTE            : no feature scaling applied - decision trees, "
          "random forests and XGBoost split on raw thresholds, so scaling "
          "would not change how they behave.")

    return X, y, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# STEP 3 : CROSS-VALIDATED HYPERPARAMETER TUNING
# ---------------------------------------------------------------------------
def tune_model(name, estimator, param_grid, X_train, y_train):
    """Grid-search the estimator with stratified k-fold CV and return the
    best fitted model along with its cross-validated ROC-AUC score."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(estimator, param_grid, scoring="roc_auc",
                          cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)

    print(f"\n{name}")
    print(f"  BEST PARAMS   : {search.best_params_}")
    print(f"  CV ROC-AUC    : {search.best_score_:.4f} "
          f"(mean over {CV_FOLDS} folds)")

    return search.best_estimator_, search.best_score_


def train_models(X_train, y_train):
    banner("STEP 3 : CROSS-VALIDATED HYPERPARAMETER TUNING")

    tree, tree_cv_auc = tune_model(
        "Decision Tree",
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {
            "max_depth": [3, 4, 5, 6, 8, None],
            "min_samples_leaf": [1, 2, 5, 10],
            "criterion": ["gini", "entropy"],
        },
        X_train, y_train,
    )

    forest, forest_cv_auc = tune_model(
        "Random Forest",
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "n_estimators": [100, 200, 400],
            "max_depth": [4, 6, 8, None],
            "min_samples_leaf": [1, 2, 5],
        },
        X_train, y_train,
    )

    xgb, xgb_cv_auc = tune_model(
        "XGBoost",
        XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
        {
            "n_estimators": [100, 200, 400],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
        },
        X_train, y_train,
    )

    models = {"Decision Tree": tree, "Random Forest": forest, "XGBoost": xgb}
    cv_scores = {"Decision Tree": tree_cv_auc, "Random Forest": forest_cv_auc,
                 "XGBoost": xgb_cv_auc}
    return models, cv_scores


# ---------------------------------------------------------------------------
# STEP 4 : TEST SET EVALUATION + CONFUSION MATRIX ANALYSIS
# ---------------------------------------------------------------------------
def evaluate(name, model, X_test, y_test, cv_auc, results):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        "Model": name,
        "CV ROC-AUC": cv_auc,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Test ROC-AUC": roc_auc_score(y_test, y_proba),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }
    results.append(metrics)

    print(f"\n{name}")
    print(f"  CONFUSION MATRIX (rows = actual, cols = predicted) :")
    print(f"                   Pred Benign   Pred Malignant")
    print(f"  Actual Benign    {tn:>11}   {fp:>14}")
    print(f"  Actual Malignant {fn:>11}   {tp:>14}")
    print(f"  Accuracy: {metrics['Accuracy']:.4f} | Precision: {metrics['Precision']:.4f} | "
          f"Recall: {metrics['Recall']:.4f} | F1: {metrics['F1 Score']:.4f} | "
          f"Test ROC-AUC: {metrics['Test ROC-AUC']:.4f}")

    if fn > 0:
        print(f"  {fn} malignant case(s) were missed (false negatives) - "
              f"the costliest error in a diagnostic setting.")
    if fp > 0:
        print(f"  {fp} benign case(s) were flagged as malignant (false positives) - "
              f"would lead to an unnecessary follow-up biopsy.")

    return y_pred, y_proba


def evaluate_models(models, cv_scores, X_test, y_test):
    banner("STEP 4 : TEST SET EVALUATION AND CONFUSION MATRIX ANALYSIS")

    results = []
    predictions = {}
    probabilities = {}
    for name, model in models.items():
        y_pred, y_proba = evaluate(name, model, X_test, y_test,
                                   cv_scores[name], results)
        predictions[name] = y_pred
        probabilities[name] = y_proba

    return results, predictions, probabilities


# ---------------------------------------------------------------------------
# STEP 5 : PLOTS
# ---------------------------------------------------------------------------
def make_plots(models, results_df, X_test, y_test, predictions, probabilities, feature_names):
    banner("STEP 5 : SAVING THE PLOTS")
    PLOTS_DIR.mkdir(exist_ok=True)

    # confusion matrix heatmap for every model
    for i, (name, y_pred) in enumerate(predictions.items(), start=1):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Benign", "Malignant"],
                    yticklabels=["Benign", "Malignant"])
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        safe = name.lower().replace(" ", "_")
        plt.savefig(PLOTS_DIR / f"0{i}_confusion_matrix_{safe}.png", dpi=110)
        plt.close()

    # ROC curves, all models on one chart
    plt.figure(figsize=(7, 6))
    for name, y_proba in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = results_df.loc[results_df["Model"] == name, "Test ROC-AUC"].values[0]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.title("ROC Curves - Model Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_roc_curves.png", dpi=110)
    plt.close()

    # feature importance from the random forest
    forest = models["Random Forest"]
    importances = pd.Series(forest.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_features.values, y=top_features.index, hue=top_features.index,
                palette="viridis", legend=False)
    plt.title("Top 10 Most Important Features (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_feature_importance.png", dpi=110)
    plt.close()

    # model comparison across the four headline metrics
    plot_df = results_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "Test ROC-AUC"],
        var_name="Metric", value_name="Score")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette="Set2")
    plt.title("Model Comparison Across Metrics")
    plt.ylim(0, 1.05)
    plt.legend(title=None)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_model_comparison.png", dpi=110)
    plt.close()

    for f in sorted(PLOTS_DIR.glob("*.png")):
        print(f"  SAVED : {f.relative_to(BASE_DIR)}")


# ---------------------------------------------------------------------------
# STEP 6 : SAVE THE BEST MODEL
# ---------------------------------------------------------------------------
def save_model(model, model_name, feature_names, metrics):
    banner("STEP 6 : SAVING THE BEST MODEL")

    artifact = {
        "model": model,
        "model_name": model_name,
        "feature_names": list(feature_names),
        "metrics": metrics,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"SAVED TO        : {MODEL_PATH}")
    print(f"FILE SIZE       : {os.path.getsize(MODEL_PATH):,} bytes")
    print("CONTENTS        : model + feature order + test set metrics")

    # prove the saved file is usable on its own
    loaded = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([{
        "radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.8,
        "area_mean": 1001.0, "smoothness_mean": 0.1184, "compactness_mean": 0.2776,
        "concavity_mean": 0.3001, "concave_points_mean": 0.1471, "symmetry_mean": 0.2419,
        "fractal_dimension_mean": 0.07871, "radius_se": 1.095, "texture_se": 0.9053,
        "perimeter_se": 8.589, "area_se": 153.4, "smoothness_se": 0.006399,
        "compactness_se": 0.04904, "concavity_se": 0.05373, "concave_points_se": 0.01587,
        "symmetry_se": 0.03003, "fractal_dimension_se": 0.006193, "radius_worst": 25.38,
        "texture_worst": 17.33, "perimeter_worst": 184.6, "area_worst": 2019.0,
        "smoothness_worst": 0.1622, "compactness_worst": 0.6656, "concavity_worst": 0.7119,
        "concave_points_worst": 0.2654, "symmetry_worst": 0.4601,
        "fractal_dimension_worst": 0.1189,
    }])[loaded["feature_names"]]

    label = loaded["model"].predict(sample)[0]
    proba = loaded["model"].predict_proba(sample)[0][1]

    print("\nRELOAD TEST (PREDICTING A SAMPLE TUMOR) :")
    print(f"  PREDICTED DIAGNOSIS : {'Malignant' if label == 1 else 'Benign'} "
          f"(malignant probability {proba:.3f})")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    banner("BREAST CANCER DIAGNOSIS - CLASSIFICATION CHALLENGE")

    df = load_data()
    X, y, X_train, X_test, y_train, y_test = prepare(df)

    models, cv_scores = train_models(X_train, y_train)
    results, predictions, probabilities = evaluate_models(models, cv_scores, X_test, y_test)

    results_df = (pd.DataFrame(results)
                  .sort_values("Test ROC-AUC", ascending=False)
                  .reset_index(drop=True))

    banner("MODEL COMPARISON")
    print(results_df.drop(columns=["TP", "TN", "FP", "FN"]).to_string(
        index=False,
        formatters={col: "{:.4f}".format for col in
                    ["CV ROC-AUC", "Accuracy", "Precision", "Recall", "F1 Score", "Test ROC-AUC"]}))

    best = results_df.loc[0]
    print(f"\nBEST MODEL      : {best['Model']} (ranked by test ROC-AUC)")
    print(f"TEST ROC-AUC    : {best['Test ROC-AUC']:.4f}")
    print(f"RECALL          : {best['Recall']:.4f} "
          f"({int(best['FN'])} malignant case(s) missed out of "
          f"{int(best['FN'] + best['TP'])})")

    make_plots(models, results_df, X_test, y_test, predictions, probabilities, X.columns)

    best_model = models[best["Model"]]
    save_model(best_model, best["Model"], X.columns,
               {"accuracy": float(best["Accuracy"]), "precision": float(best["Precision"]),
                "recall": float(best["Recall"]), "f1": float(best["F1 Score"]),
                "roc_auc": float(best["Test ROC-AUC"])})

    banner("DONE - ALL STEPS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
