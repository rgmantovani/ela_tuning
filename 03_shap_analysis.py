import pandas as pd
import numpy as np
import os
import shap
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from preprocessing.preprocessing import remove_constant_features, remove_correlated_features

TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)

configs = [
    {"name": "ela",      "file": "data/ela_svm_metadataset.csv",    "threshold": 0.95},
    {"name": "combined", "file": "data/combined_svm_metadataset.csv", "threshold": 0.85},
]

for cfg in configs:
    print(f"\n{'='*50}")
    print(f"Dataset: {cfg['name']}  |  tau = {cfg['threshold']}")
    print('='*50)

    df = pd.read_csv(cfg["file"])
    X = df.drop(columns=["dataset", "Class"])
    y = LabelEncoder().fit_transform(df["Class"])

    X = remove_constant_features(X)
    X = remove_correlated_features(X, threshold=cfg["threshold"])

    print(f"Features after preprocessing: {X.shape[1]}")

    model = XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
    print("\nTop 10 features by mean |SHAP|:")
    print(mean_shap.sort_values(ascending=False).head(10).to_string())

    pd.DataFrame(shap_values, columns=X.columns).to_csv(
        f"{TMP_DIR}/shap_values_{cfg['name']}.csv", index=False)
    X.to_csv(f"{TMP_DIR}/shap_X_{cfg['name']}.csv", index=False)

    print(f"\nSaved: tmp/shap_values_{cfg['name']}.csv")
    print(f"Saved: tmp/shap_X_{cfg['name']}.csv")

print("\nDone. Run 05_automated_analysis.R to generate the PDF plots.")
