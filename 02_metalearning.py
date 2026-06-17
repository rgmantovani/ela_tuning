# Usage:
# python 02_metalearning.py --seed 42 > results/execution_seed_42.log 2>&1

import os
import argparse
import itertools
import numpy as np
import pandas as pd
from scipy.io import arff

from preprocessing.preprocessing import remove_constant_features, remove_correlated_features

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

# ---------------------------------
# COMMAND LINE ARGUMENT
# ---------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random state seed")
args = parser.parse_args()

SEED = args.seed

print(f"\nUsing random seed: {SEED}")

os.makedirs("results", exist_ok=True)

# ---------------------------------
# Creating ELA metadataset
# ---------------------------------

raw_data, meta = arff.loadarff("data/classif_svm_169d_95_average.arff")
df_baseline = pd.DataFrame(raw_data)

for col in df_baseline.columns:
    if df_baseline[col].dtype == object:
        df_baseline[col] = df_baseline[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )

df_baseline.rename(columns={df_baseline.columns[0]: "dataset"}, inplace=True)

df_features = pd.read_csv("data/ela_svm_hyperspace_features.csv")

ela_metadataset = pd.merge(df_baseline[["dataset", "Class"]], df_features, on="dataset")
ela_metadataset.to_csv("data/ela_svm_metadataset.csv", index=False)

df_combined = pd.read_csv("data/combined_svm_metadataset.csv")

# ---------------------------------
# Experiment Configuration
# ---------------------------------

TARGET_COLUMN = "Class"

datasets = {
    "Baseline":        df_baseline,
    "Ela_metadataset": ela_metadataset,
    "Combined":        df_combined,
}

algorithms = {
    "SVM_Linear":        SVC(kernel="linear", probability=True, random_state=SEED),
    "SVM_RBF":           SVC(kernel="rbf",    probability=True, random_state=SEED),
    "RandomForest":      RandomForestClassifier(random_state=SEED),
    "KNN":               KNeighborsClassifier(),
    "DecisionTree":      DecisionTreeClassifier(random_state=SEED),
    "NaiveBayes":        GaussianNB(),
    "LogisticRegression":LogisticRegression(max_iter=1000, random_state=SEED),
    "XGBoost":           XGBClassifier(random_state=SEED),
}

correlation_thresholds = [0.80, 0.85, 0.90, 0.95]

loo = LeaveOneOut()
all_predictions = []
evaluation_results = []

# ---------------------------------
# Experiment
# ---------------------------------

combinations = itertools.product(datasets.items(), algorithms.items(), correlation_thresholds)

for (dataset_name, df), (alg_name, model), corr_threshold in combinations:

    print("\n======================================")
    print(f"Dataset: {dataset_name}")
    print(f"Algorithm: {alg_name}")
    print(f"Correlation Threshold: {corr_threshold}")
    print("======================================")

    X = df.drop(columns=[TARGET_COLUMN])
    original_shape = X.shape

    X = remove_constant_features(X, threshold=0)
    X = remove_correlated_features(X, threshold=corr_threshold)

    print(f"\n[Shape] Original: {original_shape} → After preprocessing: {X.shape}")
    print(f"[Features Used] {list(X.columns)}\n")

    y_raw = df[TARGET_COLUMN].values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)

    X_vals = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64).values
    n_classes = len(le.classes_)

    print(f"Dataset shape : {X_vals.shape}")
    print(f"Classes       : {n_classes}\n")

    pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
    probs = cross_val_predict(pipeline, X_vals, y_encoded, cv=loo, method="predict_proba", n_jobs=4)
    y_pred = np.argmax(probs, axis=1)

    fscore  = f1_score(y_encoded, y_pred, average="weighted")
    bal_acc = balanced_accuracy_score(y_encoded, y_pred)

    try:
        if n_classes == 2:
            auc = roc_auc_score(y_encoded, probs[:, 1])
        else:
            auc = roc_auc_score(y_encoded, probs, multi_class="ovr", average="weighted")
    except Exception:
        auc = np.nan

    evaluation_results.append({
        "Dataset":              dataset_name,
        "Algorithm":            alg_name,
        "CorrelationThreshold": corr_threshold,
        "NumFeatures":          X_vals.shape[1],
        "F1Score":              fscore,
        "BalancedAccuracy":     bal_acc,
        "AUC":                  auc,
    })

    for i, (true_enc, pred_enc, prob_row) in enumerate(zip(y_encoded, y_pred, probs)):
        row = {
            "Dataset":              dataset_name,
            "Algorithm":            alg_name,
            "CorrelationThreshold": corr_threshold,
            "TrueLabel":            le.inverse_transform([true_enc])[0],
            "PredictedLabel":       le.inverse_transform([pred_enc])[0],
        }
        row.update({f"Prob_{cls}": prob_row[j] for j, cls in enumerate(le.classes_)})
        all_predictions.append(row)

# ---------------------------------
# Saving results
# ---------------------------------

predictions_file = f"results/loo_probabilities_seed_{SEED}.csv"
metrics_file     = f"results/loo_metrics_seed_{SEED}.csv"

pd.DataFrame(all_predictions).to_csv(predictions_file, index=False)
pd.DataFrame(evaluation_results).to_csv(metrics_file, index=False)

print("\nFinished!")
print(f"Predictions saved to: {predictions_file}")
print(f"Metrics saved to: {metrics_file}")
