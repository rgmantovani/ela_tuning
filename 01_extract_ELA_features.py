import pandas as pd
from pathlib import Path
from features.ELA_features import extract_from_dataframe

if __name__ == "__main__":

    landscape_files = list(Path("data/landscapeInputs/classif.svm/").rglob("*.csv"))

    print("\n" + "═" * 60)
    print("  Extract_from_dataframe")
    print("═" * 60)

    dfs = [pd.read_csv(p) for p in landscape_files]
    datasets = [str(s).split("/")[3].replace(".csv", "") for s in landscape_files]

    result = extract_from_dataframe(
        data             = dfs,
        x_cols           = ["cost", "gamma"],
        y_col            = "ber.test.mean",
        datasets         = datasets,
        bounds           = [(-15.0, 15.0), (-15.0, 15.0)],
        label_col        = None,
        extractor_kwargs = {"cm_grid_size": 5, "gp_max_fit": 200},
    )

    result['dataset'] = datasets
    result.to_csv("data/ela_svm_hyperspace_features.csv", index=False)
    print("\nSaved: ela_svm_hyperspace_features.csv")
