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

## Project Structure

```
ela_tuning/
├── 01_extract_ELA_features.py   # Step 1: ELA feature extraction
├── 02_metalearning.py           # Step 2: meta-learning experiment
├── 03_automatic_plotting.R      # Step 3: visualisation and aggregation
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
└── plots/                       # Output of Step 3 (PDF hyperspace plots)
```

---

## Dependencies

### Python

```bash
pip install numpy pandas scipy scikit-learn xgboost
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
