# -----------------------------------------------
# -----------------------------------------------

import time
from pathlib import Path
from features.ELA_features import *

# -----------------------------------------------
# -----------------------------------------------

if __name__ == "__main__":

    # -----------------------------
    # Reading landscape features
    # -----------------------------

    landscape_files  = list(Path("data/landscapeInputs/classif.svm/").rglob("*.csv"))  # filter by extension

    # for debug 
    # landscape_files = landscape_files[0:5]

    # -----------------------------
    # -----------------------------

    print("\n" + "═" * 60)
    print("  Extract_from_dataframe")
    print("═" * 60)

    # reading all data frames with input landcapes
    dfs = [pd.read_csv(p) for p in landscape_files]

    datasets = [str(s).split("/")[3].replace(".csv", "") for s in landscape_files]
    
    # t0 = time.perf_counter()
    # result = parallel_extract_from_dataframe(
    result = extract_from_dataframe(
        data             = dfs,
        x_cols           = ["cost", "gamma"],
        y_col            = "ber.test.mean",
        datasets         = datasets,
        bounds           = [(-15.0, 15.0), (-15.0, 15.0)],
        label_col        = None,
        extractor_kwargs = {"cm_grid_size": 5, "gp_max_fit": 200},
    )

    # print(f"  {len(dfs)} landscapes in {time.perf_counter() - t0:.2f}s "
        #   f"using {os.cpu_count()} cores")

    # adding datatsets to the resultant dataframe 
    result['dataset'] = datasets
    
    # ──────────────────────────────────────────
    # ── Save result  
    # ──────────────────────────────────────────

    result.to_csv("data/ela_svm_hyperspace_feautures.csv", index=False)
    print("\nSaved: ela_svm_hyperspace_feautures.csv")

# -----------------------------------------------
# -----------------------------------------------