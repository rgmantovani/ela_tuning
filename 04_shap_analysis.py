import pandas as pd
import numpy as np
import os
import shap
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from preprocessing.preprocessing import remove_constant_features, remove_correlated_features

TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)

# Best setup: ELA Metadataset, tau = 0.95
THRESHOLD = 0.95

df = pd.read_csv("data/ela_svm_metadataset.csv")
X = df.drop(columns=["dataset", "Class"])
y = LabelEncoder().fit_transform(df["Class"])

X = remove_constant_features(X)
X = remove_correlated_features(X, threshold=THRESHOLD)

print(f"Features after preprocessing (tau={THRESHOLD}): {X.shape[1]}")

model = XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0)
model.fit(X, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print("SHAP values shape:", shap_values.shape)

mean_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
print("\nTop 10 features by mean |SHAP|:")
print(mean_shap.sort_values(ascending=False).head(10).to_string())

pd.DataFrame(shap_values, columns=X.columns).to_csv("tmp/shap_values.csv", index=False)
X.to_csv("tmp/shap_X.csv", index=False)

print("\nSHAP values saved to tmp/shap_values.csv")
print("Feature matrix saved to tmp/shap_X.csv")
print("Run 04_automated_analysis.R to generate the PDF plots.")
