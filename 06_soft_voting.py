# Usage:
# python3 06_soft_voting.py

import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

# ---------------------------------
# Load all seed probability files
# ---------------------------------

files = sorted(glob.glob("results/loo_probabilities_seed_*.csv"))
print(f"Found {len(files)} seed files: {[os.path.basename(f) for f in files]}\n")

all_probs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

prob_cols = [c for c in all_probs.columns if c.startswith("Prob_")]
classes   = [c.replace("Prob_", "") for c in prob_cols]

# ---------------------------------
# Soft voting per dataset × threshold
# Group by (Dataset, CorrelationThreshold, instance index within group)
# Average probabilities across all algorithms and seeds
# ---------------------------------

results = []

for (dataset, threshold), group in all_probs.groupby(["Dataset", "CorrelationThreshold"]):

    # Within a (dataset, threshold) group each algorithm × seed produces one
    # prediction per instance. We rely on the fact that instances appear in the
    # same order for every algorithm/seed combination.
    n_instances = group["TrueLabel"].nunique() if False else None

    # Assign a within-group instance index based on row order per algorithm/seed
    group = group.copy()
    group["instance_idx"] = group.groupby(["Algorithm", "Dataset"]).cumcount()

    # Average probabilities across algorithms and seeds for each instance
    agg = group.groupby("instance_idx")[prob_cols + ["TrueLabel"]].agg(
        {**{c: "mean" for c in prob_cols}, "TrueLabel": "first"}
    )

    y_true  = agg["TrueLabel"].values
    probs   = agg[prob_cols].values
    y_pred  = np.array([classes[i] for i in np.argmax(probs, axis=1)])

    f1      = f1_score(y_true, y_pred, average="weighted")
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y_true)
    try:
        if len(classes) == 2:
            auc = roc_auc_score(y_enc, probs[:, 1])
        else:
            auc = roc_auc_score(y_enc, probs, multi_class="ovr", average="weighted")
    except Exception:
        auc = np.nan

    results.append({
        "Dataset":              dataset,
        "CorrelationThreshold": threshold,
        "NumInstances":         len(agg),
        "F1Score":              round(f1,      4),
        "BalancedAccuracy":     round(bal_acc, 4),
        "AUC":                  round(auc,     4),
    })

    print(f"[{dataset}] threshold={threshold}  F1={f1:.4f}  BAC={bal_acc:.4f}  AUC={auc:.4f}")

# ---------------------------------
# Aggregate across thresholds (mean ± std)
# ---------------------------------

df_results = pd.DataFrame(results)

print("\n=== Soft Voting — mean ± std across correlation thresholds ===\n")
summary = df_results.groupby("Dataset")[["F1Score", "BalancedAccuracy", "AUC"]].agg(["mean", "std"])
print(summary.round(4).to_string())

# ---------------------------------
# Save
# ---------------------------------

os.makedirs("results", exist_ok=True)
out_file = "results/soft_voting_metrics.csv"
df_results.to_csv(out_file, index=False)
print(f"\nDetailed results saved to: {out_file}")
