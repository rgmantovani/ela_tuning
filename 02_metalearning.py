# ---------------------------------
# Required Python packages
# ---------------------------------

import os
import pandas as pd
from scipy.io import arff

# For preprocessing
from preprocessing.preprocessing import *

# Algorithms
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Experimental evaluation
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Metrics
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    balanced_accuracy_score
)

# ---------------------------------
# CREATE RESULTS DIRECTORY
# ---------------------------------

os.makedirs("results", exist_ok=True)

# ---------------------------------
# Creating ELA metadataset
# ---------------------------------

data, meta = arff.loadarff("data/classif_svm_169d_95_average.arff")
df_baseline = pd.DataFrame(data)

# Decode bytes
for col in df_baseline.columns:
    if df_baseline[col].dtype == object:
        df_baseline[col] = df_baseline[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )

# Rename first column safely
first_col = df_baseline.columns[0]
df_baseline.rename(columns={first_col: "dataset"}, inplace=True)

# Read features
df_features = pd.read_csv("data/ela_svm_hyperspace_features.csv")

# Select columns
sub = df_baseline[["dataset", "Class"]]

# Merge
ela_metadataset = pd.merge(sub, df_features, on="dataset")

# Save
ela_metadataset.to_csv("data/ela_svm_metadataset.csv", index=False)

# ---------------------------------
# Experimenti Configuration
# ---------------------------------

TARGET_COLUMN = "Class"

datasets = {
    "Baseline":        df_baseline,
    "Ela_metadataset": ela_metadataset
}

algorithms = {
    "SVM": SVC(probability=True),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "NaiveBayes": GaussianNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

loo = LeaveOneOut()

all_predictions = []
evaluation_results = []

# ---------------------------------
# TRAINING + PROBABILITY PREDICTION
# ---------------------------------

for dataset_name, df in datasets.items():

    print(f"\n===== {dataset_name} =====")

    # Features and target
    X = df.drop(columns=[TARGET_COLUMN])
    # y = df[TARGET_COLUMN]

    original_shape = X.shape

    # Remove constant / near-constant features
    data = remove_constant_features(X, threshold=0)

    # Remove highly correlated features
    data = remove_correlated_features(data, threshold=0.95)

    print(f"\n[Shape] Original: {original_shape} → After preprocessing: {data.shape}")
    print(f"[Features Used] {list(data.columns)}\n")

    # Drop non-feature columns
    X = data
    y_raw = df[TARGET_COLUMN].values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)

    # Replace inf/nan
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64).values

    print(f"Dataset shape : {X.shape}")
    print(f"Classes       : {len(le.classes_)}\n")

    n_classes = len(le.classes_)

    for alg_name, model in algorithms.items():

        print(f"Running {alg_name}...")

        # Pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        probs = cross_val_predict(pipeline, X, y_encoded, cv=loo, method="predict_proba", n_jobs=-1)

        # Predicted classes
        y_pred = np.argmax(probs, axis=1)

        # ---------------------------------
        # EVALUATION METRICS
        # ---------------------------------

        fscore = f1_score(y_encoded, y_pred, average="weighted")
        bal_acc = balanced_accuracy_score(y_encoded, y_pred)
        
        # AUC
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_encoded, probs[:, 1])
            else:
                auc = roc_auc_score(y_encoded, probs, multi_class="ovr", average="weighted")
        except Exception:
            auc = np.nan

        evaluation_results.append({
            "Dataset": dataset_name,
            "Algorithm": alg_name,
            "F1Score": fscore,
            "BalancedAccuracy": bal_acc,
            "AUC": auc
        })

        # ---------------------------------
        # STORE SAMPLE PREDICTIONS
        # ---------------------------------

        for i in range(len(y_encoded)):

            row = {
                "Dataset": dataset_name,
                "Algorithm": alg_name,
                "TrueLabel": le.inverse_transform([y_encoded[i]])[0],
                "PredictedLabel": le.inverse_transform([y_pred[i]])[0]
            }

            for class_idx, class_name in enumerate(le.classes_):
                row[f"Prob_{class_name}"] = probs[i][class_idx]

            all_predictions.append(row)

# ---------------------------------
# FINAL DATAFRAMES
# ---------------------------------

predictions_df = pd.DataFrame(all_predictions)
metrics_df = pd.DataFrame(evaluation_results)

# ---------------------------------
# SHOW RESULTS
# ---------------------------------

print("\n===== METRICS =====")
print(metrics_df)

print("\n===== PREDICTIONS =====")
print(predictions_df.head())

# ---------------------------------
# SAVE RESULTS
# ---------------------------------

predictions_df.to_csv("results/loo_probabilities.csv", index=False)
metrics_df.to_csv("results/loo_metrics.csv", index=False)
