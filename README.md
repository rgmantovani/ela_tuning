# ELA Tuning

This project investigates whether **Exploratory Landscape Analysis (ELA) features** extracted from SVM hyperparameter search spaces can improve algorithm selection/configuration via meta-learning.

The pipeline is composed of three sequential steps, each corresponding to a numbered script.

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
  03_automatic_plotting.R
            │
            ▼
  plots/hyperspaces/*.pdf
            │
            ▼
  04_shap_analysis.py
            │
            ▼
  tmp/shap_values.csv
  tmp/shap_X.csv
            │
            ▼
  05_automated_analysis.R
            │
            ▼
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
python 01_extract_ELA_features.py
```

---

## Step 2 — Meta-learning Experiment

**Script:** `02_metalearning.py`

Builds the ELA meta-dataset by merging the extracted features with the baseline performance data, then evaluates seven classifiers under Leave-One-Out Cross-Validation (LOOCV) on two meta-datasets: the **Baseline** (network/graph features only) and the **ELA meta-dataset** (ELA features appended).

### Input

| Path | Description |
|------|-------------|
| `data/classif_svm_169d_95_average.arff` | Baseline meta-dataset with 169 network/graph features and a binary class label (`Defaults` vs `Tuning`). One row per dataset. |
| `data/ela_svm_hyperspace_features.csv` | ELA features produced by Step 1. |

### Output

| Path | Description |
|------|-------------|
| `data/ela_svm_metadataset.csv` | Merged meta-dataset combining the class label from the baseline ARFF with the ELA features. |
| `results/loo_metrics_seed_<SEED>.csv` | Per-combination evaluation metrics (F1, Balanced Accuracy, AUC) across all algorithms, datasets, and correlation thresholds. |
| `results/loo_probabilities_seed_<SEED>.csv` | Per-instance predicted probabilities and true/predicted labels for every LOOCV fold. |
| `results/execution_seed_<SEED>.log` | Full console output captured during execution. |

### How to run

Run once per random seed. Results across seeds are aggregated in Step 3.

```bash
# single seed
python 02_metalearning.py --seed 42

# capture log (recommended)
python 02_metalearning.py --seed 42 > results/execution_seed_42.log 2>&1
```

To reproduce the full experiment (seeds used: 0–7, 42, 51):

```bash
for SEED in 0 1 2 3 4 5 6 7 42 51; do
  python 02_metalearning.py --seed $SEED > results/execution_seed_${SEED}.log 2>&1
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

### Preprocessing applied per run

1. Remove constant/near-constant features.
2. Remove highly correlated features at four thresholds: 0.80, 0.85, 0.90, 0.95.
3. Standard scaling inside the LOOCV pipeline (no data leakage).

---

## Step 3 — Automatic Plotting

**Script:** `03_automatic_plotting.R`

Generates two sets of outputs: (1) scatter plots of each SVM hyperparameter landscape coloured by Balanced Accuracy; (2) a summary table of average metrics (mean ± SD across seeds) ordered by AUC.

### Input

| Path | Description |
|------|-------------|
| `data/landscapeInputs/classif.svm/*.csv` | Same landscape CSVs used in Step 1. |
| `results/loo_metrics_seed_*.csv` | All metric files produced by Step 2 (one per seed). |

### Output

| Path | Description |
|------|-------------|
| `plots/hyperspaces/<dataset>_hyperspace.pdf` | Scatter plot of the SVM hyperparameter space for each dataset, coloured by BAC (red = 0, white = 0.5, blue = 1). |
| `df_full` (in memory) | Data frame with averaged metrics across seeds, ordered by descending AUC. Print or export as needed. |

### How to run

```bash
Rscript 03_automatic_plotting.R
```

Or interactively inside an R session:

```r
source("03_automatic_plotting.R")
```

---

## Step 4 — SHAP Analysis

**Script:** `04_shap_analysis.py`

Trains XGBoost on the ELA meta-dataset using the best experimental setup (τ = 0.95) and computes SHAP values to explain individual predictions.

### Input

| Path | Description |
|------|-------------|
| `data/ela_svm_metadataset.csv` | ELA meta-dataset produced by Step 2. |

### Output

| Path | Description |
|------|-------------|
| `tmp/shap_values.csv` | SHAP values matrix (one row per instance, one column per feature). |
| `tmp/shap_X.csv` | Feature matrix after preprocessing (used by Step 5 for plotting). |

### How to run

```bash
python3 04_shap_analysis.py
```

---

## Step 5 — Automated Analysis Plots

**Script:** `05_automated_analysis.R`

Generates all analysis PDFs from the experimental results: metric distributions, paired comparisons, heatmaps, feature importances, and SHAP visualisations.

### Input

| Path | Description |
|------|-------------|
| `results/loo_metrics_seed_*.csv` | Metric files produced by Step 2 (one per seed). |
| `tmp/shap_values.csv` | SHAP values produced by Step 4. |
| `tmp/shap_X.csv` | Feature matrix produced by Step 4. |

### Output

| Path | Description |
|------|-------------|
| `plots/analysis/1_auc_by_dataset.pdf` | Violin + boxplot: AUC distribution by meta-dataset. |
| `plots/analysis/2_paired_auc.pdf` | Paired dot plot: ELA vs Baseline per algorithm × threshold. |
| `plots/analysis/3_auc_by_algorithm.pdf` | Grouped barplot with error bars by algorithm. |
| `plots/analysis/4_heatmap_alg_threshold.pdf` | Heatmap: algorithm × threshold coloured by AUC. |
| `plots/analysis/5_auc_vs_threshold.pdf` | Line + ribbon: AUC vs correlation threshold per algorithm. |
| `plots/analysis/6_all_metrics_facet.pdf` | Facet grid: F1, BAC, and AUC by algorithm. |
| `plots/analysis/7_seed_stability.pdf` | Seed stability for stochastic algorithms. |
| `plots/analysis/8_xgboost_top10_features.pdf` | Top-10 features by XGBoost gain importance. |
| `plots/analysis/9_top10_features_by_class.pdf` | Distribution of top-10 features split by class. |
| `plots/analysis/10_shap_beeswarm.pdf` | SHAP beeswarm plot coloured by feature value. |
| `plots/analysis/11_shap_bar.pdf` | Mean \|SHAP\| bar chart for top-10 features. |

### How to run

```bash
Rscript 05_automated_analysis.R
```

> **Note:** run `04_shap_analysis.py` before `05_automated_analysis.R` to generate the SHAP plots. If `tmp/shap_values.csv` is missing, the SHAP plots are skipped automatically.

---

## Project Structure

```
ela_tuning/
├── 01_extract_ELA_features.py   # Step 1: ELA feature extraction
├── 02_metalearning.py           # Step 2: meta-learning experiment
├── 03_automatic_plotting.R      # Step 3: hyperspace visualisation
├── 04_shap_analysis.py          # Step 4: SHAP explanation of XGBoost
├── 05_automated_analysis.R      # Step 5: analysis plots
│
├── features/
│   └── ELA_features.py          # ELA extractor classes and public API
│
├── preprocessing/
│   └── preprocessing.py         # Feature filtering utilities
│
├── data/
│   ├── classif_svm_169d_95_average.arff   # Baseline meta-dataset
│   ├── ela_svm_hyperspace_features.csv    # Output of Step 1
│   ├── ela_svm_metadataset.csv            # Output of Step 2
│   ├── landscapeInputs/classif.svm/       # Input landscapes (one CSV per dataset)
│   └── datasets/                          # Dataset metadata and OpenML info
│
├── results/                     # Output of Step 2 (metrics, probabilities, logs)
├── tmp/                         # Intermediate files (SHAP values, feature matrix)
└── plots/
    ├── hyperspaces/             # Output of Step 3 (PDF hyperspace plots)
    └── analysis/                # Output of Step 5 (PDF analysis plots)
```

---

## Dependencies

### Python

```bash
pip install numpy pandas scipy scikit-learn xgboost shap
```

### R

```r
install.packages(c("ggplot2", "reshape2"))
```

---

## Contact

**Rafael Gomes Mantovani**
Universidade Tecnológica Federal do Paraná (UTFPR) — campus Apucarana
✉ [rafaelmantovani@utfpr.edu.br](mailto:rafaelmantovani@utfpr.edu.br)
