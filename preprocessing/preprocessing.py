import numpy as np
import pandas as pd


def remove_constant_features(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    variances = df.var(numeric_only=True)
    cols_to_keep = variances[variances > threshold].index
    removed = set(df.select_dtypes(include=[np.number]).columns) - set(cols_to_keep)
    if removed:
        print(f"[Constant Features] Removed: {sorted(removed)}")
    return df[cols_to_keep]


def remove_correlated_features(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    corr_matrix = df.corr(numeric_only=True).abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    cols_to_drop = [
        col for col in upper_triangle.columns
        if any(upper_triangle[col] > threshold)
    ]
    if cols_to_drop:
        print(f"[Correlated Features] Removed (threshold={threshold}): {sorted(cols_to_drop)}")
    return df.drop(columns=cols_to_drop)
