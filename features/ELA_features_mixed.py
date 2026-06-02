"""
ELA_features_mixed.py

Exploratory Landscape Analysis feature extractor for hyperparameter spaces
with mixed types (categorical, integer, and real-valued dimensions).

Designed primarily for the rpart hyperparameter space:

    usesurrogate   — categorical : {0, 1, 2}
    surrogatestyle — categorical : {0, 1}
    cp             — real        : [0.0, 1.0]
    minsplit       — integer     : [1, 60]
    minbucket      — integer     : [1, 20]
    maxdepth       — integer     : [1, 30]

The Gower distance is used as the unified metric for all distance-based
features, handling mixed types naturally:
  - numerical dimensions : |xi - xj| / range
  - categorical dimensions: 0 if equal, 1 if different
  Final value is averaged over all dimensions.

Feature groups
--------------
ela_meta.*         Linear/quadratic surrogate fit on one-hot + numerical space
ela_disp.*         Dispersion ratio good vs. rest via Gower distances
ela_ic.*           Information content along Gower-NN walk
cat_landscape.*    Per-categorical-column fitness variation across categories
ela_cm.*           Cell mapping: categorical combos × numerical grid
ela_lc.*           Landscape correlation: Gower distance to centroid + NN autocorr.
nbc.*              Nearest better clustering via Gower distances
fdc.*              Fitness-distance correlation via Gower distances
gh.*               Gradient homogeneity on the numerical subspace only
ls.*               GP length scales on the numerical subspace only
sample.*           Raw sample statistics
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import product as iproduct

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures


# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MixedTypeConfig:
    """
    Describes the type structure of a mixed hyperparameter space.

    Parameters
    ----------
    col_names   : names of all columns in the order they appear in X
    cat_indices : column indices of categorical hyperparameters
    num_indices : column indices of numerical hyperparameters (real + integer)
    int_indices : subset of num_indices that are integer-valued
    cat_values  : possible values per categorical column, keyed by column index
    num_bounds  : (lo, hi) bounds per numerical column, keyed by column index
    """
    col_names   : list[str]
    cat_indices : list[int]
    num_indices : list[int]
    int_indices : list[int]
    cat_values  : dict[int, list]
    num_bounds  : dict[int, tuple[float, float]]

    @property
    def n_dims(self) -> int:
        return len(self.cat_indices) + len(self.num_indices)


# Ready-to-use preset for the rpart hyperparameter space
RPART_CONFIG = MixedTypeConfig(
    col_names   = ["usesurrogate", "surrogatestyle", "cp", "minsplit", "minbucket", "maxdepth"],
    cat_indices = [0, 1],
    num_indices = [2, 3, 4, 5],
    int_indices = [3, 4, 5],
    cat_values  = {
        0: [0, 1, 2],   # usesurrogate:   never / use / use+display
        1: [0, 1],      # surrogatestyle: total count / non-missing count
    },
    num_bounds  = {
        2: (0.0001, 0.1),  # cp
        3: (1,      50),   # minsplit
        4: (1,      50),   # minbucket
        5: (1,      30),   # maxdepth
    },
)


# ═════════════════════════════════════════════════════════════════════════════
# Core utilities
# ═════════════════════════════════════════════════════════════════════════════

def gower_distance(X: np.ndarray, config: MixedTypeConfig) -> np.ndarray:
    """
    Compute the full n×n Gower distance matrix for a mixed-type dataset.

    Numerical : contribution = |xi - xj| / range_k   (range from num_bounds)
    Categorical: contribution = 0 if equal, 1 if different
    Result is the average contribution across all dimensions.
    """
    n = len(X)
    D = np.zeros((n, n))

    for k in config.num_indices:
        lo, hi = config.num_bounds[k]
        rng = (hi - lo) or 1.0
        col = X[:, k].astype(float)
        D += np.abs(col[:, None] - col[None, :]) / rng

    for k in config.cat_indices:
        col = X[:, k]
        D += (col[:, None] != col[None, :]).astype(float)

    return D / config.n_dims


def _mixed_centroid(X: np.ndarray, config: MixedTypeConfig) -> np.ndarray:
    """Mode for categorical dims, mean for numerical dims."""
    c = np.empty(X.shape[1], dtype=object)
    for k in config.cat_indices:
        vals, counts = np.unique(X[:, k], return_counts=True)
        c[k] = vals[np.argmax(counts)]
    for k in config.num_indices:
        c[k] = X[:, k].astype(float).mean()
    return c


def _gower_to_point(X: np.ndarray, point: np.ndarray, config: MixedTypeConfig) -> np.ndarray:
    """Gower distance from each row of X to a single reference point."""
    d = np.zeros(len(X))
    for k in config.num_indices:
        lo, hi = config.num_bounds[k]
        rng = (hi - lo) or 1.0
        d += np.abs(X[:, k].astype(float) - float(point[k])) / rng
    for k in config.cat_indices:
        d += (X[:, k] != point[k]).astype(float)
    return d / config.n_dims


def _encode_X(X: np.ndarray, config: MixedTypeConfig) -> np.ndarray:
    """One-hot encode categorical dims and concatenate with numerical dims."""
    parts = []
    for k in config.cat_indices:
        enc = OneHotEncoder(
            categories=[config.cat_values[k]], sparse_output=False, handle_unknown="ignore"
        )
        enc.fit(np.array(config.cat_values[k]).reshape(-1, 1))
        parts.append(enc.transform(X[:, k].reshape(-1, 1)))
    if config.num_indices:
        parts.append(X[:, config.num_indices].astype(float))
    return np.hstack(parts)


def _nn_path_from_dist(D: np.ndarray) -> list[int]:
    """Greedy nearest-neighbour tour given a precomputed distance matrix."""
    n = len(D)
    D_work = D.copy()
    np.fill_diagonal(D_work, np.inf)
    visited = [False] * n
    path = [0]
    visited[0] = True
    for _ in range(n - 1):
        row = D_work[path[-1]].copy()
        row[[i for i, v in enumerate(visited) if v]] = np.inf
        nxt = int(np.argmin(row))
        path.append(nxt)
        visited[nxt] = True
    return path


def _normalise(y: np.ndarray) -> np.ndarray:
    rng = y.max() - y.min()
    return (y - y.min()) / (rng + 1e-12)


def _skew(x: np.ndarray) -> float:
    m, s = x.mean(), x.std()
    return float(np.mean(((x - m) / (s + 1e-12)) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    m, s = x.mean(), x.std()
    return float(np.mean(((x - m) / (s + 1e-12)) ** 4) - 3)


# ═════════════════════════════════════════════════════════════════════════════
# Feature classes
# ═════════════════════════════════════════════════════════════════════════════

class MixedMetaModelFeatures:
    """
    Fit linear and quadratic surrogates on the encoded (one-hot + numerical)
    space and report adjusted R² and coefficient statistics.

    High linear adj-R²  → smooth, globally predictable landscape.
    High quadratic adj-R² → curved but structured landscape.
    """

    def compute(self, X: np.ndarray, y: np.ndarray, config: MixedTypeConfig) -> dict:
        Xe = _encode_X(X, config)
        features = {}

        lin = LinearRegression().fit(Xe, y)
        features["ela_meta.lin_simple.adj_r2"]    = self._adj_r2(lin, Xe, y)
        features["ela_meta.lin_simple.intercept"] = float(lin.intercept_)
        features["ela_meta.coef_min"]             = float(lin.coef_.min())
        features["ela_meta.coef_max"]             = float(lin.coef_.max())
        features["ela_meta.coef_range"]           = float(lin.coef_.max() - lin.coef_.min())

        quad = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("reg",  LinearRegression()),
        ])
        quad.fit(Xe, y)
        features["ela_meta.quad_simple.adj_r2"] = self._adj_r2(quad, Xe, y)

        return features

    @staticmethod
    def _adj_r2(model, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        n, p = X.shape
        return float(1 - (1 - r2) * (n - 1) / max(n - p - 1, 1))


class MixedDispersionFeatures:
    """
    Compare mean pairwise Gower distances of good (top-q%) vs. remaining samples.
    Ratio < 1 → good configurations cluster together (exploitable structure).
    """

    def __init__(self, quantile: float = 0.1):
        self.quantile = quantile

    def compute(self, y: np.ndarray, D: np.ndarray) -> dict:
        threshold = np.quantile(y, self.quantile)
        good = y <= threshold
        bad  = ~good

        base = {"ela_disp.dist_all_mean": float(D.mean())}

        if good.sum() < 2 or bad.sum() < 2:
            return {
                "ela_disp.ratio_mean":   np.nan,
                "ela_disp.ratio_median": np.nan,
                "ela_disp.diff_mean":    np.nan,
                **base,
            }

        dg = D[np.ix_(good, good)]
        db = D[np.ix_(bad,  bad)]
        mg,  mb  = dg.mean(),     db.mean()
        mdg, mdb = np.median(dg), np.median(db)

        return {
            "ela_disp.ratio_mean":   float(mg  / (mb  + 1e-12)),
            "ela_disp.ratio_median": float(mdg / (mdb + 1e-12)),
            "ela_disp.diff_mean":    float(mg - mb),
            **base,
        }


class MixedInformationContentFeatures:
    """
    Shannon entropy of consecutive symbol pairs along the Gower-NN walk.
    High entropy → rugged landscape; low → smooth or plateau-like.
    """

    def __init__(self, epsilons: list[float] | None = None):
        self.epsilons = epsilons or [0.0, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.5]

    def compute(self, y: np.ndarray, D: np.ndarray) -> dict:
        path   = _nn_path_from_dist(D)
        y_path = y[path]
        H_vals = []
        phi0   = None

        for eps in self.epsilons:
            phi = self._phi(y_path, eps)
            H_vals.append(self._entropy(phi))
            if phi0 is None:
                phi0 = phi

        best_i = int(np.argmax(H_vals))
        return {
            "ela_ic.h_max":     float(max(H_vals)),
            "ela_ic.eps_s":     float(self.epsilons[best_i]),
            "ela_ic.eps_ratio": float(self.epsilons[-1] / (self.epsilons[best_i] + 1e-12)),
            "ela_ic.h_zero":    float(H_vals[0]),
            "ela_ic.m0":        float(np.mean(np.array(phi0) == 0)) if phi0 else np.nan,
        }

    @staticmethod
    def _phi(y_path: np.ndarray, eps: float) -> list[int]:
        return [
            1  if y_path[i+1] - y_path[i] >  eps else
            (-1 if y_path[i+1] - y_path[i] < -eps else 0)
            for i in range(len(y_path) - 1)
        ]

    @staticmethod
    def _entropy(phi: list[int]) -> float:
        if len(phi) < 2:
            return 0.0
        pairs  = list(zip(phi[:-1], phi[1:]))
        total  = len(pairs)
        counts: dict = {}
        for p in pairs:
            counts[p] = counts.get(p, 0) + 1
        return -sum((c / total) * np.log2(c / total + 1e-12) for c in counts.values())


class CategoricalLandscapeFeatures:
    """
    Per-categorical-column analysis of how fitness varies across category values.

    For each categorical column c with values {v0, v1, ...}:
      mean_std   : std of per-value mean fitness  (large → this variable matters)
      mean_range : max − min of per-value means
      best_cat   : category value with lowest mean fitness
      worst_cat  : category value with highest mean fitness
      mean_cat<v>: mean fitness for each specific value

    Prefix: cat_landscape.<col_name>.*
    """

    def compute(self, X: np.ndarray, y: np.ndarray, config: MixedTypeConfig) -> dict:
        features: dict = {}
        for k in config.cat_indices:
            name   = config.col_names[k]
            values = config.cat_values[k]
            means  = [
                float(y[X[:, k] == v].mean()) if (X[:, k] == v).any() else np.nan
                for v in values
            ]
            means_arr = np.array(means, dtype=float)

            prefix = f"cat_landscape.{name}"
            features[f"{prefix}.mean_std"]   = float(np.nanstd(means_arr))
            features[f"{prefix}.mean_range"]  = float(np.nanmax(means_arr) - np.nanmin(means_arr))
            features[f"{prefix}.best_cat"]    = float(values[int(np.nanargmin(means_arr))])
            features[f"{prefix}.worst_cat"]   = float(values[int(np.nanargmax(means_arr))])
            for v, m in zip(values, means):
                features[f"{prefix}.mean_cat{v}"] = m

        return features


class MixedCellMappingFeatures:
    """
    Two-level partition of the hyperparameter space:
      1. Outer level: all combinations of categorical values.
      2. Inner level: a regular grid over the numerical dimensions.

    Features summarise how fitness statistics vary across cells.

    num_grid_size : resolution of the numerical grid per dimension.
                    Default 3 gives 3^4 = 81 numerical cells per categorical combo
                    for the rpart space (6 combos × 81 = 486 max cells).
    """

    def __init__(self, num_grid_size: int = 3):
        self.g = num_grid_size

    def compute(self, X: np.ndarray, y: np.ndarray, config: MixedTypeConfig) -> dict:
        g = self.g
        num_edges = {
            k: np.linspace(lo, hi, g + 1)
            for k, (lo, hi) in config.num_bounds.items()
        }
        cat_combos = list(iproduct(*[config.cat_values[k] for k in config.cat_indices]))

        cell_means      = []
        occupied_combos = 0

        for combo in cat_combos:
            mask = np.ones(len(X), dtype=bool)
            for i, k in enumerate(config.cat_indices):
                mask &= (X[:, k] == combo[i])

            if not mask.any():
                continue
            occupied_combos += 1

            X_sub, y_sub = X[mask], y[mask]
            num_cells: dict[tuple, list] = {}

            for xi, yi in zip(X_sub, y_sub):
                cell_idx = tuple(
                    min(int(np.digitize(xi[k], num_edges[k]) - 1), g - 1)
                    for k in config.num_indices
                )
                num_cells.setdefault(cell_idx, []).append(yi)

            for vals in num_cells.values():
                cell_means.append(float(np.mean(vals)))

        means_arr = np.array(cell_means)
        n_cells   = len(means_arr)
        max_cells = len(cat_combos) * (g ** len(config.num_indices))

        return {
            "ela_cm.mean_std":     float(means_arr.std())                               if n_cells > 1 else np.nan,
            "ela_cm.mean_range":   float(means_arr.max() - means_arr.min())             if n_cells > 1 else np.nan,
            "ela_cm.diff_mean":    float(np.mean(np.abs(np.diff(sorted(means_arr)))))  if n_cells > 1 else np.nan,
            "ela_cm.n_cells":      float(n_cells),
            "ela_cm.coverage":     float(n_cells / max_cells)                           if max_cells > 0 else np.nan,
            "ela_cm.n_cat_groups": float(occupied_combos),
        }


class MixedLandscapeCorrelationFeatures:
    """
    Pearson correlation between fitness and:
      - Gower distance to the mixed-type centroid
      - Fitness of the Gower-nearest neighbour (spatial autocorrelation)

    Also reports distributional stats of the objective.
    """

    def compute(
        self, X: np.ndarray, y: np.ndarray, D: np.ndarray, config: MixedTypeConfig
    ) -> dict:
        centroid = _mixed_centroid(X, config)
        d_cent   = _gower_to_point(X, centroid, config)

        D_inf  = D.copy()
        np.fill_diagonal(D_inf, np.inf)
        y_nn   = y[np.argmin(D_inf, axis=1)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_cent, _ = pearsonr(d_cent, y)
            r_nn,   _ = pearsonr(y, y_nn)

        return {
            "ela_lc.r_pearson_dist_centroid": float(r_cent if not np.isnan(r_cent) else 0.0),
            "ela_lc.r_pearson_nn":            float(r_nn   if not np.isnan(r_nn)   else 0.0),
            "ela_lc.y_std":                   float(y.std()),
            "ela_lc.y_skewness":              _skew(y),
            "ela_lc.y_kurtosis":              _kurtosis(y),
        }


class MixedNBCFeatures:
    """
    For each point, compare Gower distance to its nearest neighbour vs. its
    nearest *better* neighbour.
    Ratio ≈ 1 → funnel; large ratio → multi-modal / deceptive landscape.
    n_candidates: fraction of points with no better neighbour (proxy for basin count).
    """

    def compute(self, y: np.ndarray, D: np.ndarray) -> dict:
        n = len(y)
        D_inf = D.copy()
        np.fill_diagonal(D_inf, np.inf)

        nn_dist  = D_inf.min(axis=1)
        nb_dist  = np.full(n, np.nan)
        nb_ratio = np.full(n, np.nan)

        for i in range(n):
            better = y < y[i]
            if better.any():
                D_row = D_inf[i].copy()
                D_row[~better] = np.inf
                nb_idx = int(np.argmin(D_row))
                if y[nb_idx] < y[i]:
                    nb_dist[i]  = D_inf[i, nb_idx]
                    nb_ratio[i] = y[nb_idx] / (abs(y[i]) + 1e-12)

        valid = ~np.isnan(nb_dist)
        ratio = nb_dist[valid] / (nn_dist[valid] + 1e-12)
        n_cand = int((~valid).sum())

        return {
            "nbc.nn_nb.dist.ratio.mean": float(ratio.mean())            if valid.any() else np.nan,
            "nbc.nn_nb.dist.ratio.sd":   float(ratio.std())             if valid.any() else np.nan,
            "nbc.nb.fitness.ratio.mean": float(nb_ratio[valid].mean())  if valid.any() else np.nan,
            "nbc.nb.fitness.ratio.sd":   float(nb_ratio[valid].std())   if valid.any() else np.nan,
            "nbc.n_candidates":          float(n_cand / n),
            "nbc.dist.ratio.coeff_var":  float(ratio.std() / (ratio.mean() + 1e-12)) if valid.any() else np.nan,
        }


class MixedFDCFeatures:
    """
    Pearson correlation between fitness and Gower distance to the best-observed
    configuration.
    fdc.r < −0.15 → big-valley (easy); fdc.r > 0.15 → deceptive landscape.
    """

    def compute(self, y: np.ndarray, D: np.ndarray) -> dict:
        best  = int(np.argmin(y))
        dists = D[best].copy()

        mask  = np.ones(len(y), dtype=bool)
        mask[best] = False
        d_s, y_s = dists[mask], y[mask]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, pv = pearsonr(d_s, y_s)

        cov   = np.cov(d_s, y_s)[0, 1]
        slope = cov / (np.var(d_s, ddof=1) + 1e-12)

        return {
            "fdc.r":         float(r)  if not np.isnan(r)  else 0.0,
            "fdc.p_value":   float(pv) if not np.isnan(pv) else 1.0,
            "fdc.dist.mean": float(d_s.mean()),
            "fdc.dist.sd":   float(d_s.std()),
            "fdc.slope":     float(slope),
        }


class NumericalSubspaceGHFeatures:
    """
    Gradient homogeneity estimated via WLS over k nearest Euclidean neighbours,
    computed exclusively on the numerical subspace.

    gh.mean → 1: clean gradient funnel; gh.mean → 0: chaotic gradients.
    Note: categorical dimensions are ignored here — use CategoricalLandscapeFeatures
    to capture categorical structure.
    """

    def __init__(self, k_neighbours: int = 6):
        self.k = k_neighbours

    def compute(self, X: np.ndarray, y: np.ndarray, config: MixedTypeConfig) -> dict:
        X_num = X[:, config.num_indices].astype(float)
        n = len(y)
        k = min(self.k, n - 1)

        D = cdist(X_num, X_num)
        np.fill_diagonal(D, np.inf)

        grads = np.zeros((n, X_num.shape[1]))
        for i in range(n):
            nb = np.argsort(D[i])[:k]
            dX = X_num[nb] - X_num[i]
            dy = y[nb] - y[i]
            w  = 1.0 / (D[i, nb] + 1e-12)
            W  = np.diag(w)
            try:
                g = np.linalg.lstsq(dX.T @ W @ dX, dX.T @ W @ dy, rcond=None)[0]
            except np.linalg.LinAlgError:
                g = np.zeros(X_num.shape[1])
            grads[i] = g

        norms  = np.linalg.norm(grads, axis=1, keepdims=True) + 1e-12
        g_unit = grads / norms
        upper  = (g_unit @ g_unit.T)[np.triu_indices(n, k=1)]

        path   = _nn_path_from_dist(D)
        angles = [
            np.degrees(np.arccos(float(np.clip(g_unit[a] @ g_unit[b], -1, 1))))
            for a, b in zip(path[:-1], path[1:])
        ]

        gdir = X_num[np.argmin(y)] - X_num.mean(axis=0)
        gdir /= (np.linalg.norm(gdir) + 1e-12)

        return {
            "gh.mean":       float(upper.mean()),
            "gh.sd":         float(upper.std()),
            "gh.norm.mean":  float(norms.mean()),
            "gh.norm.sd":    float(norms.std()),
            "gh.angle.mean": float(np.mean(angles)) if angles else np.nan,
            "gh.neg_ratio":  float(np.mean(grads @ gdir < 0)),
        }


class NumericalSubspaceLSFeatures:
    """
    Fit a GP with an anisotropic Matérn-5/2 kernel on the numerical subspace;
    extract optimised length scales per numerical dimension.

    Large ls → smooth along that axis; small ls → rugged.
    ls.ratio captures anisotropy across numerical dimensions.
    Note: categorical dimensions are not included in the GP fit.
    """

    def __init__(self, max_fit: int = 200, nu: float = 2.5):
        self.max_fit = max_fit
        self.nu      = nu

    def compute(self, X: np.ndarray, y: np.ndarray, config: MixedTypeConfig) -> dict:
        X_num = X[:, config.num_indices].astype(float)

        if len(y) > self.max_fit:
            idx   = np.random.choice(len(y), self.max_fit, replace=False)
            X_num = X_num[idx]
            y     = y[idx]

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=[1.0] * X_num.shape[1],
            length_scale_bounds=(1e-3, 1e3),
            nu=self.nu,
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=3, normalize_y=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X_num, y)

        ls = np.atleast_1d(gp.kernel_.k2.length_scale)

        # Per-dimension length scales mapped to column names
        features: dict = {
            "ls.mean":     float(ls.mean()),
            "ls.min":      float(ls.min()),
            "ls.max":      float(ls.max()),
            "ls.ratio":    float(ls.max() / (ls.min() + 1e-12)),
            "ls.log_mean": float(np.log10(ls.mean() + 1e-12)),
            "ls.lml":      float(gp.log_marginal_likelihood_value_),
        }
        for k, ls_k in zip(config.num_indices, ls):
            name = config.col_names[k] if config.col_names else f"num{k}"
            features[f"ls.{name}"] = float(ls_k)

        return features


# ═════════════════════════════════════════════════════════════════════════════
# Unified extractor
# ═════════════════════════════════════════════════════════════════════════════

class MixedELAExtractor:
    """
    Unified ELA extractor for mixed-type hyperparameter spaces.

    Parameters
    ----------
    config          : MixedTypeConfig describing column types and domains.
                      Use RPART_CONFIG for the rpart hyperparameter space.
    ic_quantile     : top-q fraction defining "good" samples in Dispersion.
    gp_max_fit      : max points for GP fitting (subsampled if larger).
    cm_num_grid_size: numerical grid resolution per dimension in CellMapping.
    gh_k_neighbours : k for WLS gradient estimation.
    """

    def __init__(
        self,
        config           : MixedTypeConfig,
        ic_quantile      : float = 0.1,
        gp_max_fit       : int   = 200,
        cm_num_grid_size : int   = 3,
        gh_k_neighbours  : int   = 6,
    ):
        self.config = config
        self.ic_q   = ic_quantile
        self.gp_max = gp_max_fit
        self.cm_g   = cm_num_grid_size
        self.gh_k   = gh_k_neighbours

    def compute(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """
        Compute all ELA features for a single mixed-type landscape.

        Parameters
        ----------
        X : (n, d) array — rows are configurations, columns match config.col_names
        y : (n,)   array — objective/performance values (lower is better assumed)
        """
        y      = np.asarray(y, dtype=float).ravel()
        y_norm = _normalise(y)
        config = self.config

        D = gower_distance(X, config)

        features: dict[str, float] = {}
        features.update(MixedMetaModelFeatures().compute(X, y_norm, config))
        features.update(MixedDispersionFeatures(self.ic_q).compute(y_norm, D))
        features.update(MixedInformationContentFeatures().compute(y_norm, D))
        features.update(CategoricalLandscapeFeatures().compute(X, y_norm, config))
        features.update(MixedCellMappingFeatures(self.cm_g).compute(X, y_norm, config))
        features.update(MixedLandscapeCorrelationFeatures().compute(X, y_norm, D, config))
        features.update(MixedNBCFeatures().compute(y_norm, D))
        features.update(MixedFDCFeatures().compute(y_norm, D))
        features.update(NumericalSubspaceGHFeatures(self.gh_k).compute(X, y_norm, config))
        features.update(NumericalSubspaceLSFeatures(self.gp_max).compute(X, y_norm, config))

        features["sample.n"]      = float(len(y))
        features["sample.y_min"]  = float(y.min())
        features["sample.y_max"]  = float(y.max())
        features["sample.y_mean"] = float(y.mean())

        return features


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def extract_mixed_from_dataframe(
    data             : pd.DataFrame | list[pd.DataFrame],
    x_cols           : list[str],
    y_col            : str,
    config           : MixedTypeConfig,
    datasets         : list[str] | None = None,
    extractor_kwargs : dict | None = None,
) -> pd.DataFrame:
    """
    Compute mixed-type ELA features from one or more pre-sampled DataFrames.

    Parameters
    ----------
    data             : a single DataFrame or a list of DataFrames.
                       Each must contain columns `x_cols` and `y_col`.
    x_cols           : column names for the hyperparameters, in the same order
                       as config.col_names.
    y_col            : column name for the objective/performance value.
    config           : MixedTypeConfig (e.g. RPART_CONFIG).
    datasets         : optional list of dataset labels (one per DataFrame).
    extractor_kwargs : extra keyword arguments forwarded to MixedELAExtractor.

    Returns
    -------
    pd.DataFrame with one row per landscape and one column per feature.
    The first column is 'dataset' (integer index if `datasets` is None).

    Example
    -------
    >>> from features.ELA_features_mixed import extract_mixed_from_dataframe, RPART_CONFIG
    >>> result = extract_mixed_from_dataframe(
    ...     data     = df,
    ...     x_cols   = RPART_CONFIG.col_names,
    ...     y_col    = "error",
    ...     config   = RPART_CONFIG,
    ...     datasets = ["iris"],
    ... )
    """
    extractor_kwargs = extractor_kwargs or {}
    extractor = MixedELAExtractor(config=config, **extractor_kwargs)

    if isinstance(data, pd.DataFrame):
        data = [data]

    if datasets is None:
        datasets = [str(i) for i in range(len(data))]

    rows = []
    for i, df in enumerate(data):
        print(f"{i+1}/{len(datasets)} - {datasets[i]}")
        X = df[x_cols].to_numpy()
        y = df[y_col].to_numpy(dtype=float)
        rows.append(extractor.compute(X, y))

    result = pd.DataFrame(rows)
    result.insert(0, "dataset", datasets)
    return result
