# ELA Tuning

This project investigates whether **Exploratory Landscape Analysis (ELA) features** extracted from SVM hyperparameter search spaces can improve algorithm selection/configuration via meta-learning.

The pipeline is composed of four sequential steps, each corresponding to a numbered script.

---

## Pipeline Overview

```
data/landscapeInputs/classif.svm/*.csv
            │
            ▼
  01_extract_ELA_features.py
            │
            ▼
  data/ela_svm_hyperspace_features.csv
            │
  data/classif_svm_169d_95_average.arff
  data/combined_svm_metadataset.csv
            │
            ▼
  02_metalearning.py
            │
            ▼
  data/ela_svm_metadataset.csv
  results/loo_metrics_seed_<SEED>.csv
  results/loo_probabilities_seed_<SEED>.csv
            │
            ▼
  03_shap_analysis.py
            │
            ▼
  tmp/shap_values_ela.csv
  tmp/shap_X_ela.csv
  tmp/shap_values_combined.csv
  tmp/shap_X_combined.csv
            │
            ▼
  04_plotting.R
            │
            ▼
  plots/hyperspaces/*.pdf
  plots/analysis/*.pdf
```

---

## Step 1 — Extract ELA Features

**Script:** `01_extract_ELA_features.py`

Reads the pre-sampled SVM hyperparameter landscapes (one CSV per dataset) and computes ELA features for each landscape using the extractor defined in `features/ELA_features.py`.

### Input

| Path | Description |
|------|-------------|
| `data/landscapeInputs/classif.svm/*.csv` | One CSV per dataset. Each row is a sampled point in the SVM hyperparameter space (`cost`, `gamma`) with the corresponding cross-validated BER (`ber.test.mean`). Each file contains ~30 000 rows. |

### Output

| Path | Description |
|------|-------------|
| `data/ela_svm_hyperspace_features.csv` | One row per dataset, one column per ELA feature (~40 features covering meta-model fit, dispersion, information content, cell mapping, landscape correlation, NBC, FDC, gradient homogeneity, and GP length scales). |

### How to run

```bash
python3 01_extract_ELA_features.py
```

---

## Step 2 — Meta-learning Experiment

**Script:** `02_metalearning.py`

Builds the ELA meta-dataset by merging the extracted features with the baseline performance data, then evaluates eight classifiers under Leave-One-Out Cross-Validation (LOOCV) on three meta-datasets: **Baseline**, **ELA meta-dataset**, and **Combined**.

### Input

| Path | Description |
|------|-------------|
| `data/classif_svm_169d_95_average.arff` | Baseline meta-dataset with 169 network/graph features and a binary class label (`Defaults` vs `Tuning`). |
| `data/ela_svm_hyperspace_features.csv` | ELA features produced by Step 1. |
| `data/combined_svm_metadataset.csv` | Pre-built meta-dataset combining baseline and ELA features. |

### Output

| Path | Description |
|------|-------------|
| `data/ela_svm_metadataset.csv` | Merged meta-dataset combining the class label from the baseline ARFF with the ELA features. |
| `results/loo_metrics_seed_<SEED>.csv` | Per-combination evaluation metrics (F1, Balanced Accuracy, AUC) across all algorithms, meta-datasets, and correlation thresholds. |
| `results/loo_probabilities_seed_<SEED>.csv` | Per-instance predicted probabilities and true/predicted labels for every LOOCV fold. |
| `results/execution_seed_<SEED>.log` | Full console output captured during execution. |

### How to run

```bash
# single seed
python3 02_metalearning.py --seed 42 > results/execution_seed_42.log 2>&1
```

To reproduce the full experiment (seeds used: 0–7, 42, 51):

```bash
for SEED in 0 1 2 3 4 5 6 7 42 51; do
  python3 02_metalearning.py --seed $SEED > results/execution_seed_${SEED}.log 2>&1 &
done
```

### Algorithms evaluated

| Identifier | Algorithm |
|------------|-----------|
| `SVM_Linear` | Support Vector Machine (linear kernel) |
| `SVM_RBF` | Support Vector Machine (RBF kernel) |
| `RandomForest` | Random Forest |
| `KNN` | K-Nearest Neighbours |
| `DecisionTree` | Decision Tree |
| `NaiveBayes` | Gaussian Naïve Bayes |
| `LogisticRegression` | Logistic Regression |
| `XGBoost` | Gradient Boosted Trees |

### Meta-datasets

| Identifier | Description |
|------------|-------------|
| `Baseline` | 169 network/graph meta-features only |
| `Ela_metadataset` | ELA features extracted from SVM hyperparameter landscapes |
| `Combined` | Baseline + ELA features concatenated |

### Preprocessing applied per run

1. Remove constant/near-constant features.
2. Remove highly correlated features at four thresholds: 0.80, 0.85, 0.90, 0.95.
3. Standard scaling inside the LOOCV pipeline (no data leakage).

---

## Step 3 — SHAP Analysis

**Script:** `03_shap_analysis.py`

Trains XGBoost on the ELA and Combined meta-datasets using the best experimental setup and computes SHAP values to explain individual predictions.

### Input

| Path | Description |
|------|-------------|
| `data/ela_svm_metadataset.csv` | ELA meta-dataset produced by Step 2. |
| `data/combined_svm_metadataset.csv` | Combined meta-dataset. |

### Output

| Path | Description |
|------|-------------|
| `tmp/shap_values_ela.csv` | SHAP values for the ELA meta-dataset. |
| `tmp/shap_X_ela.csv` | Preprocessed feature matrix for the ELA meta-dataset. |
| `tmp/shap_values_combined.csv` | SHAP values for the Combined meta-dataset. |
| `tmp/shap_X_combined.csv` | Preprocessed feature matrix for the Combined meta-dataset. |

### How to run

```bash
python3 03_shap_analysis.py
```

---

## Step 4 — Plotting

**Script:** `04_plotting.R`

Generates all plots: scatter plots of each SVM hyperparameter landscape and all analysis figures from the experimental results, including metric distributions, paired comparisons, heatmaps, and SHAP visualisations.

> **Note:** run `03_shap_analysis.py` before `04_plotting.R` to generate the SHAP plots. If the SHAP files are missing from `tmp/`, the SHAP plots are skipped automatically.

### Input

| Path | Description |
|------|-------------|
| `data/landscapeInputs/classif.svm/*.csv` | Landscape CSVs used in Step 1. |
| `results/loo_metrics_seed_*.csv` | Metric files produced by Step 2 (one per seed). |
| `tmp/shap_values_ela.csv` | SHAP values produced by Step 3. |
| `tmp/shap_X_ela.csv` | Feature matrix produced by Step 3. |
| `tmp/shap_values_combined.csv` | SHAP values produced by Step 3. |
| `tmp/shap_X_combined.csv` | Feature matrix produced by Step 3. |

### Output

| Path | Description |
|------|-------------|
| `plots/hyperspaces/<dataset>_hyperspace.pdf` | Scatter plot of the SVM hyperparameter space for each dataset, coloured by BAC. |
| `plots/analysis/1_auc_by_dataset.pdf` | Violin + boxplot: AUC distribution by meta-dataset. |
| `plots/analysis/2_paired_auc.pdf` | Paired dot plot: ELA/Combined vs Baseline per algorithm × threshold. |
| `plots/analysis/4_heatmap_alg_threshold.pdf` | Heatmap: algorithm × threshold coloured by mean AUC. |
| `plots/analysis/10_shap_ela.pdf` | SHAP beeswarm + bar chart for the ELA meta-dataset. |
| `plots/analysis/12_shap_combined.pdf` | SHAP beeswarm + bar chart for the Combined meta-dataset. |

### How to run

```bash
Rscript 04_plotting.R
```

---

## Project Structure

```
ela_tuning/
├── 01_extract_ELA_features.py   # Step 1: ELA feature extraction
├── 02_metalearning.py           # Step 2: meta-learning experiment (LOOCV)
├── 03_shap_analysis.py          # Step 3: SHAP explanation of XGBoost
├── 04_plotting.R                # Step 4: all plots (hyperspaces + analysis)
│
├── features/
│   └── ELA_features.py          # ELA extractor classes and public API
│
├── preprocessing/
│   └── preprocessing.py         # Feature filtering utilities
│
├── data/
│   ├── classif_svm_169d_95_average.arff   # Baseline meta-dataset
│   ├── combined_svm_metadataset.csv       # Combined meta-dataset
│   ├── ela_svm_hyperspace_features.csv    # Output of Step 1
│   ├── ela_svm_metadataset.csv            # Output of Step 2
│   ├── landscapeInputs/classif.svm/       # Input landscapes (one CSV per dataset)
│   └── datasets/                          # Dataset metadata and OpenML info
│
├── results/                     # Output of Step 2 (metrics, probabilities, logs)
├── tmp/                         # Intermediate files (SHAP values, feature matrices)
│   └── generate_pdf_tables.py   # Utility script to generate supplementary PDF
└── plots/
    ├── hyperspaces/             # Output of Step 4 (PDF hyperspace plots)
    └── analysis/                # Output of Step 4 (PDF analysis plots)
```

---

## Dependencies

### Python

```bash
pip install numpy pandas scipy scikit-learn xgboost shap
```

### R

```r
install.packages(c("ggplot2", "reshape2", "patchwork", "RColorBrewer"))
```

---

## Contact

**Rafael Gomes Mantovani**
Universidade Tecnológica Federal do Paraná (UTFPR) — campus Apucarana
✉ [rafaelmantovani@utfpr.edu.br](mailto:rafaelmantovani@utfpr.edu.br)
