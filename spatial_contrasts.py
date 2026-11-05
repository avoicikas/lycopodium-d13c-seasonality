"""Paired contrasts under refitted Gaussian spatial covariance models (Tables S2-S7, Fig. S7)."""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.spatial.distance import cdist
from scipy.stats import skew
from statsmodels.stats.multitest import multipletests

from bootstrap import make_contrasts, site_panel
from config import PART_NAMES, RESPONSES, SEASON_NAMES, STATS
from prepare_data import load_coordinates, load_samples
from spatial_autocorrelation import moran_i, neighbour_weights

SEED = 451
RESAMPLES = 19999
RANGES = (5, 10, 20, 40, 80, 160, 320)
FRACTIONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
MODELS = ("iid", "exponential", "matern32", "exponential_10m", "exponential_100m")
FAMILIES = ("seasonal", "organ", "interaction")
KEYS = ["part", "versus", "later", "earlier"]


class CovarianceGrid:
    """Restricted eigensystems of a covariance family, for exact REML on a finite grid."""

    def __init__(self, distances: np.ndarray, model: str) -> None:
        self.n = len(distances)
        self.model = model
        q = helmert(self.n, full=False).T
        ranges = {
            "exponential": RANGES,
            "matern32": RANGES,
            "exponential_10m": (10,),
        }.get(model, (100,))
        fractions = (0.0,) if model == "iid" else FRACTIONS
        self.blocks = []
        matrices, mean_weights, se_gls, se_mean, parameters = [], [], [], [], []
        for range_m in ranges:
            x = distances / range_m
            if model == "matern32":
                kernel = (1 + np.sqrt(3) * x) * np.exp(-np.sqrt(3) * x)
            else:
                kernel = np.exp(-x)
            eigenvalues, eigenvectors = np.linalg.eigh(q.T @ kernel @ q)
            inverse_eigenvalues, logdets = [], []
            for fraction in fractions:
                # Independence is represented once; its range is unidentified.
                if fraction == 0 and matrices:
                    continue
                matrix = (1 - fraction) * np.eye(self.n) + fraction * kernel
                restricted = 1 - fraction + fraction * eigenvalues
                inverse_eigenvalues.append(1 / restricted)
                logdets.append(np.log(restricted).sum())
                inverse_one = np.linalg.solve(matrix, np.ones(self.n))
                mean_weights.append(inverse_one / inverse_one.sum())
                se_gls.append(1 / inverse_one.sum())
                se_mean.append(matrix.sum() / self.n**2)
                matrices.append(matrix)
                parameters.append((range_m if fraction else 0, fraction))
            self.blocks.append(
                (q @ eigenvectors, np.array(inverse_eigenvalues).T, np.array(logdets))
            )
        self.matrices = np.array(matrices)
        self.mean_weights = np.array(mean_weights)
        self.se_gls = np.array(se_gls)
        self.se_mean = np.array(se_mean)
        self.parameters = np.array(parameters)

    def fit(self, values: np.ndarray) -> dict[str, np.ndarray]:
        """Fit vectors in rows, profiling the total variance; REML minimum over the grid."""
        values = np.atleast_2d(values)
        centered = values - values.mean(axis=1, keepdims=True)
        quadratic, likelihood = [], []
        for transform, inverse_eigenvalues, logdets in self.blocks:
            sums = (centered @ transform) ** 2 @ inverse_eigenvalues
            quadratic.append(sums)
            likelihood.append((self.n - 1) * np.log(sums / (self.n - 1)) + logdets)
        quadratic = np.concatenate(quadratic, axis=1)
        likelihood = np.concatenate(likelihood, axis=1)
        index = likelihood.argmin(axis=1)
        variance = quadratic[np.arange(len(values)), index] / (self.n - 1)
        return {
            "index": index,
            "variance": variance,
            "gls": np.einsum("bi,bi->b", values, self.mean_weights[index]),
            "equal_weight": values.mean(axis=1),
            "se_gls": np.sqrt(variance * self.se_gls[index]),
            "se_equal_weight": np.sqrt(variance * self.se_mean[index]),
        }


def bootstrap_contrast(
    values: np.ndarray, grid: CovarianceGrid, normal_draws: np.ndarray, family_size: int
) -> list[dict]:
    """Parametric bootstrap under zero mean and the fitted covariance, refitting every draw."""
    observed = grid.fit(values)
    index = int(observed["index"][0])
    variance = observed["variance"][0]
    simulations = (
        normal_draws @ np.linalg.cholesky(grid.matrices[index]).T * np.sqrt(variance)
    )
    fitted = grid.fit(simulations)
    range_m, fraction = grid.parameters[index]
    rows = []
    for estimator in ("equal_weight", "gls"):
        estimate = float(observed[estimator][0])
        se = float(observed[f"se_{estimator}"][0])
        t_null = np.abs(fitted[estimator] / fitted[f"se_{estimator}"])
        point_q = np.quantile(t_null, 0.95, method="higher")
        family_q = np.quantile(t_null, 1 - 0.05 / family_size, method="higher")
        rows.append(
            {
                "model": grid.model,
                "estimator": estimator,
                "estimate_spatial": estimate,
                "se_spatial": se,
                "p_bootstrap": (1 + np.count_nonzero(t_null >= abs(estimate / se)))
                / (len(t_null) + 1),
                "pointwise_lower": estimate - point_q * se,
                "pointwise_upper": estimate + point_q * se,
                "bonferroni_lower": estimate - family_q * se,
                "bonferroni_upper": estimate + family_q * se,
                "range_m": range_m,
                "spatial_fraction": fraction,
                "total_variance": variance,
                "range_boundary": bool(
                    fraction > 0
                    and grid.model in ("exponential", "matern32")
                    and range_m in (RANGES[0], RANGES[-1])
                ),
                "fraction_upper_boundary": bool(fraction == FRACTIONS[-1]),
                "bootstrap_independence_fraction": float(
                    np.mean(grid.parameters[fitted["index"], 1] == 0)
                ),
                "bootstrap_range_upper_fraction": float(
                    np.mean(grid.parameters[fitted["index"], 0] == RANGES[-1])
                ),
                "resamples": len(normal_draws),
            }
        )
    return rows


def label(row: pd.Series) -> str:
    """Readable contrast label with the original sign convention."""
    organ = PART_NAMES[row["part"]]
    if row["family"] == "seasonal":
        return f"{organ}: {SEASON_NAMES[row['later']]} − {SEASON_NAMES[row['earlier']]}"
    pair = f"{organ} − {PART_NAMES[row['versus']]}"
    if row["family"] == "organ":
        when = (
            "date-averaged"
            if row["later"] == "all_seasons"
            else SEASON_NAMES[row["later"]]
        )
        return f"{pair} ({when})"
    return f"{pair}: {SEASON_NAMES[row['later']]} − {SEASON_NAMES[row['earlier']]}"


def run(response: str, family: str) -> pd.DataFrame:
    """Fit every contrast of one family under all five covariance models."""
    df = load_samples()
    metadata, contrast_weights = make_contrasts()
    coordinates = load_coordinates().sort_index()[["easting", "northing"]].to_numpy()
    values = site_panel(df, response) @ contrast_weights.T
    selected = metadata[metadata["family"] == family]
    distances = cdist(coordinates, coordinates)
    weights = neighbour_weights(coordinates, 2)
    grids = [CovarianceGrid(distances, model) for model in MODELS]
    original = pd.read_csv(STATS / "contrasts.csv", keep_default_na=False)
    original = original[
        (original["response"] == response) & (original["family"] == family)
    ]
    original = original.merge(
        selected.reset_index(names="contrast_id"), on=["family", *KEYS]
    )
    original["independent_holm_p"] = multipletests(
        original["bootstrap_p"], method="holm"
    )[1]
    rows = []
    for index, row in selected.iterrows():
        observations = values[:, index]
        previous = original[original["contrast_id"] == index].iloc[0].to_dict()
        # The same Gaussian draws are reused across covariance models.
        seed = SEED + index + (1000 if response == "c" else 0)
        normal_draws = np.random.default_rng(seed).standard_normal(
            (RESAMPLES, len(values))
        )
        for grid in grids:
            for result in bootstrap_contrast(
                observations, grid, normal_draws, len(selected)
            ):
                rows.append(
                    {
                        **previous,
                        **result,
                        "seed": seed,
                        "family_size": len(selected),
                        "moran_i_k2": moran_i(observations, weights),
                        "contrast_skewness": skew(observations, bias=False),
                        "label": label(row),
                    }
                )
    result = pd.DataFrame(rows)
    result["spatial_holm_p"] = result.groupby(["model", "estimator"])[
        "p_bootstrap"
    ].transform(lambda p: multipletests(p, method="holm")[1])
    return result


def main() -> None:
    jobs = [(response, family) for response in RESPONSES for family in FAMILIES]
    with ProcessPoolExecutor() as pool:
        frames = list(pool.map(run, *zip(*jobs)))
    STATS.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(
        STATS / "spatial_contrasts.csv", index=False
    )


if __name__ == "__main__":
    main()
