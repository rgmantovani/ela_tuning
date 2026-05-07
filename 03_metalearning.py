import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from preprocessing.preprocessing import *

# ── Leave-One-Out + Decision Tree ─────────────────────────────────────────────

df = pd.read_csv("./data/metadataset_3classes.csv")
# df = pd.read_csv("./data/metadataset_4classes.csv")
original_shape = df.shape
LABEL_COL = "target"

# Remove constant / near-constant features
data = remove_constant_features(df, threshold=0)

# Remove highly correlated features
data = remove_correlated_features(data, threshold=0.9)

print(f"\n[Shape] Original: {original_shape} → After preprocessing: {data.shape}")
print(f"[Features Used] {list(data.columns)}\n")

# Normalize features to [0, 1]
data = normalize_features(data)


# Drop non-feature columns
X = data
y_raw = df[LABEL_COL].values

# Encode string labels to integers
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Replace inf/nan
X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64).values

print(f"Dataset shape : {X.shape}")
print(f"Classes       : {len(le.classes_)}\n")

# ── Leave-One-Out ─────────────────────────────────────────────

loo = LeaveOneOut()
neighbors_str = [f"Nearest neighbors({i})" for i in range(3,16)]
classifier_knn = [KNeighborsClassifier(i) for i in range(3,16)]

names = [
    "Decision Tree",
    "Random Forest",
    "XGBoost"
]

names = names + neighbors_str


print(names)

classifiers = [
    DecisionTreeClassifier(random_state=42),
    RandomForestClassifier(random_state=42),
    XGBClassifier(random_state=42)
]
classifiers += classifier_knn

# iterate over classifiers
for name, clf in zip(names, classifiers):
    print(f"- {name}")
    y_true, y_pred = [], []

    n = len(y)
    for i, (train_idx, test_idx) in enumerate(loo.split(X), start=1):
        print(f"\r[{i:>4}/{n}] running LOO folds...", end="", flush=True)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        y_pred.append(clf.predict(X_test)[0])
        y_true.append(y_test[0])

    print("\nDone.\n")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)


    accuracy = accuracy_score(y_true, y_pred)
    print(f"LOO Accuracy: {accuracy:.4f}  ({int(accuracy * n)}/{n} correct)\n")

    bac = balanced_accuracy_score(y_true, y_pred)
    print(f"LOO Balanced Accuracy: {bac:.4f}  ({int(bac * n)}/{n} correct)\n")

    # f1 = f1_score(y_true, y_pred)
    # print(f"LOO Accuracy: {f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=le.classes_,
        zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=clf.classes_)
    disp.plot()
    plt.savefig(f"results/confusion_matrix_{name}.png")
    plt.close()
    # plt.show()

    # ── Save predictions ──────────────────────────────────────────────────────────

    results_df = pd.DataFrame({
        "dataset"    : df['dataset'],
        "true_label" : le.inverse_transform(y_true),
        "pred_label" : le.inverse_transform(y_pred),
        "correct"    : y_true == y_pred,
    })

    #TODO: fix file name, remove empty spaces
    outputfile = "results/" + name + "_metalevel_loo_predictions.csv"
    results_df.to_csv(outputfile, index=False)
    print("Predictions saved to " + outputfile)