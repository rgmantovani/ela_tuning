"""
ELA_features.py

===============
Exploratory Landscape Analysis (ELA) feature extractor for 2-dimensional continuous 
hyperspace. It includes the following feature categories:

    ela_meta   — linear / quadratic surrogate fit quality
    ela_disp   — dispersion of best vs. rest samples
    ela_ic     — information content / entropy along NN path
    ela_cm     — cell-mapping grid statistics
    ela_lc     — landscape correlation (distance to centroid, NN autocorr.)
    nbc        — nearest better clustering
    fdc        — fitness distance correlation
    gh         — gradient homogeneity
    ls         — GP length-scale features
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from concurrent.futures import ProcessPoolExecutor, as_completed

# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _latin_hypercube_sample(n: int, bounds: list[tuple]) -> np.ndarray:
    """Latin Hypercube Sample over an arbitrary d-dimensional box."""
    d = len(bounds)
    result = np.zeros((n, d))
    for i, (lo, hi) in enumerate(bounds):
        cut    = np.linspace(0, 1, n + 1)
        points = np.random.uniform(cut[:n], cut[1:])
        np.random.shuffle(points)
        result[:, i] = lo + points * (hi - lo)
    return result


def _normalise(y: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    rng = y.max() - y.min()
    return (y - y.min()) / (rng + 1e-12)


def _skew(x: np.ndarray) -> float:
    m, s = x.mean(), x.std()
    return float(np.mean(((x - m) / (s + 1e-12)) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    m, s = x.mean(), x.std()
    return float(np.mean(((x - m) / (s + 1e-12)) ** 4) - 3)


def _nearest_neighbour_path(X: np.ndarray) -> list[int]:
    n = len(X)
    dist_matrix = cdist(X, X)
    np.fill_diagonal(dist_matrix, np.inf)
    visited = [False] * n
    path    = [0]
    visited[0] = True
    for _ in range(n - 1):
        d = dist_matrix[path[-1]].copy()
        d[[i for i, v in enumerate(visited) if v]] = np.inf
        nxt = int(np.argmin(d))
        path.append(nxt)
        visited[nxt] = True
    return path


# ═════════════════════════════════════════════════════════════════════════════
# Feature classes — Script 1
# ═════════════════════════════════════════════════════════════════════════════

class MetaModelFeatures:
    """
    Fit linear and quadratic surrogate models; report adjusted R² and
    coefficient statistics.  High linear R² → smooth bowl-like landscape.
    High quadratic-only R² → curved but not chaotic.
    """

    def compute(self, X: np.ndarray, y: np.ndarray) -> dict:
        features = {}

        lin = LinearRegression().fit(X, y)
        features["ela_meta.lin_simple.adj_r2"]       = self._adj_r2(lin, X, y)
        features["ela_meta.lin_simple.intercept"]    = float(lin.intercept_)
        features["ela_meta.lin_w_interact.coef_min"] = float(lin.coef_.min())
        features["ela_meta.lin_w_interact.coef_max"] = float(lin.coef_.max())

        quad_pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("reg",  LinearRegression()),
        ])
        quad_pipe.fit(X, y)
        features["ela_meta.quad_simple.adj_r2"] = self._adj_r2(quad_pipe, X, y)

        return features

    @staticmethod
    def _adj_r2(model, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        n, p = X.shape[0], X.shape[1]
        return float(1 - (1 - r2) * (n - 1) / max(n - p - 1, 1))


class DispersionFeatures:
    """
    Compare pairwise distances of 'good' (top-q%) vs. remaining samples.
    Ratio < 1 → good solutions cluster (easy, local-search friendly).
    """

    def __init__(self, quantile: float = 0.1):
        self.quantile = quantile

    def compute(self, X: np.ndarray, y: np.ndarray) -> dict:
        threshold = np.quantile(y, self.quantile)
        good      = y <= threshold
        bad       = ~good

        nan_row = {
            "ela_disp.ratio_mean_02":   np.nan,
            "ela_disp.ratio_median_02": np.nan,
            "ela_disp.diff_mean_02":    np.nan,
            "ela_disp.dist_all_mean":   np.nan,
        }
        if good.sum() < 2 or bad.sum() < 2:
            return nan_row

        dg  = cdist(X[good], X[good])
        db  = cdist(X[bad],  X[bad])
        da  = cdist(X, X)

        mg  = np.mean(dg);   mb  = np.mean(db)
        mdg = np.median(dg); mdb = np.median(db)

        return {
            "ela_disp.ratio_mean_02":   float(mg  / (mb  + 1e-12)),
            "ela_disp.ratio_median_02": float(mdg / (mdb + 1e-12)),
            "ela_disp.diff_mean_02":    float(mg  -  mb),
            "ela_disp.dist_all_mean":   float(np.mean(da)),
        }


class InformationContentFeatures:
    """
    Shannon entropy of consecutive (up/flat/down) symbol pairs along the
    nearest-neighbour walk.  High entropy → rugged landscape.
    """

    def __init__(self, epsilons: list[float] | None = None):
        self.epsilons = epsilons or [0.0, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.5]

    def compute(self, X: np.ndarray, y: np.ndarray) -> dict:
        path   = _nearest_neighbour_path(X)
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
            "ela_ic.h_max":    float(max(H_vals)),
            "ela_ic.eps_s":    float(self.epsilons[best_i]),
            "ela_ic.eps_ratio":float(self.epsilons[-1] / (self.epsilons[best_i] + 1e-12)),
            "ela_ic.h_zero":   float(H_vals[0]),
            "ela_ic.m0":       float(np.mean(np.array(phi0) == 0)) if phi0 else np.nan,
        }

    @staticmethod
    def _phi(y_path: np.ndarray, eps: float) -> list[int]:
        return [
            1 if y_path[i+1] - y_path[i] > eps
            else (-1 if y_path[i+1] - y_path[i] < -eps else 0)
            for i in range(len(y_path) - 1)
        ]

    @staticmethod
    def _entropy(phi: list[int]) -> float:
        if len(phi) < 2:
            return 0.0
        pairs  = list(zip(phi[:-1], phi[1:]))
        total  = len(pairs)
        counts = {}
        for p in pairs:
            counts[p] = counts.get(p, 0) + 1
        return -sum((c / total) * np.log2(c / total + 1e-12) for c in counts.values())


class CellMappingFeatures:
    """
    Partition domain into a grid; analyse objective-value statistics per cell.
    High std across cell means → multi-modal landscape.
    """

    def __init__(self, grid_size: int = 5):
        self.g = grid_size

    def compute(self, X: np.ndarray, y: np.ndarray, bounds: list[tuple]) -> dict:
        g     = self.g
        edges = [np.linspace(lo, hi, g + 1) for lo, hi in bounds]
        cells: dict[tuple, list] = {}

        for xi, yi in zip(X, y):
            idx = tuple(
                min(int(np.digitize(xi[d], edges[d]) - 1), g - 1)
                for d in range(X.shape[1])
            )
            cells.setdefault(idx, []).append(yi)

        means = [np.mean(v) for v in cells.values() if v]
        return {
            "ela_cm.mean_ofs_10": float(np.std(means)),
            "ela_cm.sd_ofs_10":   float(np.mean(np.abs(np.diff(sorted(means))))),
            "ela_cm.n_cells":     float(len(means)),
            "ela_cm.coverage":    float(len(means) / g ** X.shape[1]),
        }


class LandscapeCorrelationFeatures:
    """
    Pearson correlations between f(x) and spatial structure indicators.
    Skewness and kurtosis of the objective distribution included.
    """

    def compute(self, X: np.ndarray, y: np.ndarray) -> dict:
        centroid = X.mean(axis=0)
        d_cent   = np.linalg.norm(X - centroid, axis=1)

        dist_mat = cdist(X, X)
        np.fill_diagonal(dist_mat, np.inf)
        nn_idx = np.argmin(dist_mat, axis=1)
        y_nn   = y[nn_idx]

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


# ═════════════════════════════════════════════════════════════════════════════
# Feature classes — Script 2
# ═════════════════════════════════════════════════════════════════════════════

class NearestBetterClusteringFeatures:
    """
    For each point, compare distance to its nearest neighbour vs. its nearest
    *better* neighbour.  Ratio ≈ 1 → funnel; large ratio → multi-modal.
    `nbc.n_candidates` is the fraction of points with no better neighbour
    (proxy for the number of local basins).
    """

    def compute(self, X: np.ndarray, y: np.ndarray) -> dict:
        n = len(y)
        D = cdist(X, X)
        np.fill_diagonal(D, np.inf)

        nn_dist  = np.full(n, np.nan)
        nb_dist  = np.full(n, np.nan)
        nb_ratio = np.full(n, np.nan)

        for i in range(n):
            nn_dist[i] = D[i].min()
            better     = y < y[i]
            if better.any():
                nb_idx = int(np.argmin(D[i] + (~better).astype(float) * 1e15))
                if y[nb_idx] < y[i]:
                    nb_dist[i]  = D[i, nb_idx]
                    nb_ratio[i] = y[nb_idx] / (abs(y[i]) + 1e-12)

        valid = ~np.isnan(nb_dist)
        ratio = nb_dist[valid] / (nn_dist[valid] + 1e-12)
        n_cand = int((~valid).sum())

        return {
            "nbc.nn_nb.dist.ratio.mean":  float(ratio.mean())            if valid.any() else np.nan,
            "nbc.nn_nb.dist.ratio.sd":    float(ratio.std())             if valid.any() else np.nan,
            "nbc.nb.fitness.ratio.mean":  float(nb_ratio[valid].mean())  if valid.any() else np.nan,
            "nbc.nb.fitness.ratio.sd":    float(nb_ratio[valid].std())   if valid.any() else np.nan,
            "nbc.n_candidates":           float(n_cand / n),
            "nbc.dist.ratio.coeff_var":   float(ratio.std() / (ratio.mean() + 1e-12)) if valid.any() else np.nan,
        }


class FitnessDistanceCorrelationFeatures:
    """
    Pearson correlation between f(x) and ||x − x*|| where x* is the best sample.
    fdc.r < −0.15 → "big valley" (easy);  fdc.r > 0.15 → deceptive landscape.
    """

    def compute(self, X: np.ndarray, y: np.ndarray) -> dict:
        best  = int(np.argmin(y))
        dists = np.linalg.norm(X - X[best], axis=1)
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


class GradientHomogeneityFeatures:
    """
    Estimate local gradients via WLS over k nearest neighbours; measure
    pairwise cosine similarity.  gh.mean → 1 means a clean funnel;
    gh.mean → 0 means chaotic, multi-directional gradients.
    """

    def __init__(self, k_neighbours: int = 6):
        self.k = k_neighbours

    def compute(self, X: np.ndarray, y: np.ndarray) -> dict:
        n = len(y)
        k = min(self.k, n - 1)
        D = cdist(X, X)
        np.fill_diagonal(D, np.inf)

        grads = np.zeros((n, X.shape[1]))
        for i in range(n):
            nb  = np.argsort(D[i])[:k]
            dX  = X[nb] - X[i]
            dy  = y[nb] - y[i]
            w   = 1.0 / (D[i, nb] + 1e-12)
            W   = np.diag(w)
            try:
                g = np.linalg.lstsq(dX.T @ W @ dX, dX.T @ W @ dy, rcond=None)[0]
            except np.linalg.LinAlgError:
                g = np.zeros(X.shape[1])
            grads[i] = g

        norms  = np.linalg.norm(grads, axis=1, keepdims=True) + 1e-12
        g_unit = grads / norms
        upper  = (g_unit @ g_unit.T)[np.triu_indices(n, k=1)]

        path   = _nearest_neighbour_path(X)
        angles = [
            np.degrees(np.arccos(float(np.clip(g_unit[a] @ g_unit[b], -1, 1))))
            for a, b in zip(path[:-1], path[1:])
        ]

        gdir = X[np.argmin(y)] - X.mean(axis=0)
        gdir /= (np.linalg.norm(gdir) + 1e-12)

        return {
            "gh.mean":       float(upper.mean()),
            "gh.sd":         float(upper.std()),
            "gh.norm.mean":  float(norms.mean()),
            "gh.norm.sd":    float(norms.std()),
            "gh.angle.mean": float(np.mean(angles)) if angles else np.nan,
            "gh.neg_ratio":  float(np.mean(grads @ gdir < 0)),
        }


class LengthScaleFeatures:
    """
    Fit a GP with an anisotropic Matérn-5/2 kernel; extract optimised length
    scales.  Large ls → smooth landscape; small ls → rugged.
    ls.ratio captures anisotropy between the two input dimensions.
    """

    def __init__(self, max_fit: int = 200, nu: float = 2.5):
        self.max_fit = max_fit
        self.nu      = nu

    def compute(self, X: np.ndarray, y: np.ndarray) -> dict:
        if len(y) > self.max_fit:
            idx = np.random.choice(len(y), self.max_fit, replace=False)
            X, y = X[idx], y[idx]

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=[1.0] * X.shape[1],
            length_scale_bounds=(1e-3, 1e3),
            nu=self.nu,
        )
        gp = GaussianProcessRegressor(kernel=kernel,
                                      n_restarts_optimizer=3,
                                      normalize_y=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X, y)

        ls = np.atleast_1d(gp.kernel_.k2.length_scale)
        return {
            "ls.mean":     float(ls.mean()),
            "ls.min":      float(ls.min()),
            "ls.max":      float(ls.max()),
            "ls.ratio":    float(ls.max() / (ls.min() + 1e-12)),
            "ls.log_mean": float(np.log10(ls.mean() + 1e-12)),
            "ls.lml":      float(gp.log_marginal_likelihood_value_),
        }

# ═════════════════════════════════════════════════════════════════════════════
# Core extractor
# ═════════════════════════════════════════════════════════════════════════════

class ELAExtractor:
    """
    Unified ELA extractor.  Accepts either pre-sampled arrays (X, y) or a
    callable together with bounds.  All feature groups from both scripts are
    included.

    Parameters
    ----------
    bounds          : list of (lo, hi) per dimension — used for CellMapping
                      Inferred from data when not provided.
    cm_grid_size    : grid resolution for CellMappingFeatures
    ic_quantile     : top-q fraction defining "good" samples in Dispersion
    gp_max_fit      : maximum points used to fit the GP (subsampled if larger)
    sobol_n_base    : Saltelli sample size for Sobol indices
    gh_k_neighbours : k for gradient WLS estimation
    """

    def __init__(
        self,
        bounds: list[tuple] | None = None,
        cm_grid_size: int           = 5,
        ic_quantile: float          = 0.1,
        gp_max_fit: int             = 200,
        sobol_n_base: int           = 512,
        gh_k_neighbours: int        = 6,
    ):
        self.bounds       = bounds
        self.cm_grid      = cm_grid_size
        self.ic_q         = ic_quantile
        self.gp_max       = gp_max_fit
        self.sobol_n      = sobol_n_base
        self.gh_k         = gh_k_neighbours

    def _resolve_bounds(self, X: np.ndarray) -> list[tuple]:
        if self.bounds is not None:
            return self.bounds
        return [(float(X[:, d].min()), float(X[:, d].max())) for d in range(X.shape[1])]

    def compute(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute all ELA features for a single landscape.

        Parameters
        ----------
        X    : (n, d) array of decision-space coordinates
        y    : (n,)   array of objective values
        """
        y      = np.asarray(y, dtype=float).ravel()
        y_norm = _normalise(y)
        bounds = self._resolve_bounds(X)

        features: dict[str, float] = {}

        features.update(MetaModelFeatures().compute(X, y_norm))
        features.update(DispersionFeatures(self.ic_q).compute(X, y_norm))
        features.update(InformationContentFeatures().compute(X, y_norm))
        features.update(CellMappingFeatures(self.cm_grid).compute(X, y_norm, bounds))
        features.update(LandscapeCorrelationFeatures().compute(X, y_norm))
        features.update(NearestBetterClusteringFeatures().compute(X, y_norm))
        features.update(FitnessDistanceCorrelationFeatures().compute(X, y_norm))
        features.update(GradientHomogeneityFeatures(self.gh_k).compute(X, y_norm))
        features.update(LengthScaleFeatures(self.gp_max).compute(X, y_norm))

        # ── Raw sample stats ───────────────────────────────────────────────
        features["sample.n"]      = float(len(y))
        features["sample.y_min"]  = float(y.min())
        features["sample.y_max"]  = float(y.max())
        features["sample.y_mean"] = float(y.mean())

        return features

# ═════════════════════════════════════════════════════════════════════════════
# Public API — DataFrame entry points
# ═════════════════════════════════════════════════════════════════════════════

def extract_from_dataframe(
    data: pd.DataFrame | list[pd.DataFrame],
    x_cols: list[str],
    y_col: str,
    datasets: list[str],
    bounds: list[tuple] | None = None,
    extractor_kwargs: dict | None = None,
    label_col: str | None = None,
) -> pd.DataFrame:
    """
    Compute ELA features from one or more pre-sampled DataFrames.

    Parameters
    ----------
    data            : a single DataFrame or a list of DataFrames.
                      Each must contain columns `x_cols` and `y_col`.
    x_cols          : column names for the decision-variable coordinates.
    y_col           : column name for the objective value.
    bounds          : optional [(lo, hi), …] passed to ELAExtractor.
                      Inferred from data when omitted.
    extractor_kwargs: extra keyword arguments forwarded to ELAExtractor.__init__.
    label_col       : if set, a column with this name in `data` (or a synthetic
                      integer index) is added as the first column of the result.

    Returns
    -------
    pd.DataFrame with one row per landscape and one column per feature.
    """
    extractor_kwargs = extractor_kwargs or {}
    extractor = ELAExtractor(bounds=bounds, **extractor_kwargs)

    # Normalise inputs to lists
    if isinstance(data, pd.DataFrame):
        data = [data]
    
    rows = []
    for (i, df) in enumerate(data):
        print("%d/%d - %s\n" % (i+1, len(datasets), datasets[i]))
        # print(df)

        X = df[x_cols].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)
        row = extractor.compute(X, y)

        if label_col is not None:
            label_val = df[label_col].iloc[0] if label_col in df.columns else i
            row = {label_col: label_val, **row}

        rows.append(row)

    return pd.DataFrame(rows)

# -----------------------------------------------
# -----------------------------------------------

def _worker_dataframe(args: tuple) -> tuple[int, dict]:
    """Process a single (index, DataFrame) job."""
    idx, df, x_cols, y_col, bounds, extractor_kwargs = args
    extractor = ELAExtractor(bounds=bounds, **extractor_kwargs)
    X = df[x_cols].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    features = extractor.compute(X, y)
    return idx, features

# ─────────────────────────────────────────────────────────────────────────────
# Public parallel API
# ─────────────────────────────────────────────────────────────────────────────
 
def parallel_extract_from_dataframe(
    data: pd.DataFrame | list[pd.DataFrame],
    x_cols: list[str],
    y_col: str,
    bounds: list[tuple] | None = None,
    extractor_kwargs: dict | None = None,
    label_col: str | None = None,
    n_workers: int | None = None,
) -> pd.DataFrame:
 
    extractor_kwargs = extractor_kwargs or {}
    n_workers        = 5 #n_workers or os.cpu_count()
 
    if isinstance(data, pd.DataFrame):
        data = [data]
  
    # Build job list
    jobs = [
        (i, df, x_cols, y_col, bounds, extractor_kwargs)
        for (i, df) in enumerate(data)
    ]
 
    # Run in parallel, collect ordered results
    results = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker_dataframe, job): job[0] for job in jobs}
        for future in as_completed(futures):
            idx, features = future.result()
            results[idx] = features
 
    # Attach labels
    if label_col is not None:
        for i, (df, row) in enumerate(zip(data, results)):
            label_val  = df[label_col].iloc[0] if label_col in df.columns else i
            results[i] = {label_col: label_val, **row}
 
    return pd.DataFrame(results)
 
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────