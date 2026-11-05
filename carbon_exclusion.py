"""Sensitivity of the carbon-concentration results to excluding the three C < 30% samples."""

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from statsmodels.stats.multitest import multipletests

from bootstrap import make_contrasts, site_panel
from config import RESAMPLES, SEED, STATS
from prepare_data import load_coordinates, load_samples
from spatial_contrasts import MODELS, SEED as SPATIAL_SEED
from spatial_contrasts import CovarianceGrid, bootstrap_contrast

THRESHOLD = 30.0
KEYS = ["family", "part", "versus", "later", "earlier"]


def paired_values(panel: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Contrast values per site; NaN when a cell with nonzero weight is missing."""
    values = np.nan_to_num(panel) @ weights.T
    missing = (~np.isfinite(panel)).astype(int) @ (weights != 0).T.astype(int)
    return np.where(missing, np.nan, values)


def incomplete_bootstrap(
    values: np.ndarray, draws: np.ndarray
) -> tuple[pd.DataFrame, dict]:
    """Whole-site bootstrap-t where each contrast uses only its complete sites."""
    n = np.isfinite(values).sum(axis=0)
    estimate = np.nanmean(values, axis=0)
    se = np.nanstd(values, axis=0, ddof=1) / np.sqrt(n)
    samples = values[draws]
    sample_n = np.isfinite(samples).sum(axis=1)
    sample_se = np.nanstd(samples, axis=1, ddof=1) / np.sqrt(sample_n)
    boot_t = (np.nanmean(samples, axis=1) - estimate) / sample_se
    maximum = np.abs(boot_t).max(axis=1)
    critical = np.quantile(maximum, 0.95)
    observed_t = estimate / se
    interval = np.quantile(boot_t, [0.025, 0.975], axis=0)
    count = len(draws)
    result = pd.DataFrame(
        {
            "estimate": estimate,
            "standard_error": se,
            "n_sites": n,
            "ci95_lower": estimate - interval[1] * se,
            "ci95_upper": estimate - interval[0] * se,
            "simultaneous_lower": estimate - critical * se,
            "simultaneous_upper": estimate + critical * se,
            "bootstrap_p": (1 + (np.abs(boot_t) >= np.abs(observed_t)).sum(axis=0))
            / (count + 1),
            "familywise_p": (1 + (maximum[:, None] >= np.abs(observed_t)).sum(axis=0))
            / (count + 1),
        }
    )
    return result, {
        "global_bootstrap_p": (1 + (maximum >= np.abs(observed_t).max()).sum())
        / (count + 1),
        "critical_max_t_95": critical,
    }


def independent_results(
    panels: dict[str, np.ndarray], metadata: pd.DataFrame, weights: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All 140 carbon contrasts per analysis, with the six-global-test Holm gate."""
    n = len(panels["full"])
    draws = np.random.default_rng(SEED).integers(0, n, size=(RESAMPLES, n))
    isotope = pd.read_csv(STATS / "global_tests.csv")
    isotope = isotope[isotope["response"] == "d13c"]
    contrasts, globals_ = [], []
    for analysis, panel in panels.items():
        values = paired_values(panel, weights)
        carbon = []
        for family, group in metadata.groupby("family", sort=False):
            result, global_result = incomplete_bootstrap(values[:, group.index], draws)
            result = pd.concat([group.reset_index(drop=True), result], axis=1)
            result["analysis"] = analysis
            contrasts.append(result)
            carbon.append({"response": "c", "family": family, **global_result})
        combined = pd.concat([isotope, pd.DataFrame(carbon)], ignore_index=True)
        combined["holm_p_six_global_tests"] = multipletests(
            combined["global_bootstrap_p"], method="holm"
        )[1]
        combined["analysis"] = analysis
        globals_.append(combined)
    contrast_df = pd.concat(contrasts, ignore_index=True)
    global_df = pd.concat(globals_, ignore_index=True)
    gate = global_df.loc[
        global_df["response"] == "c", ["analysis", "family", "holm_p_six_global_tests"]
    ]
    contrast_df = contrast_df.merge(gate, on=["analysis", "family"])
    contrast_df["supported_after_gate"] = (contrast_df["familywise_p"] <= 0.05) & (
        contrast_df["holm_p_six_global_tests"] <= 0.05
    )
    return contrast_df, global_df


def spatial_results(
    panel: np.ndarray, metadata: pd.DataFrame, weights: np.ndarray
) -> pd.DataFrame:
    """Refit affected contrasts under the five spatial models; reuse the unaffected fits."""
    coordinates = load_coordinates().sort_index()[["easting", "northing"]].to_numpy()
    distances = cdist(coordinates, coordinates)
    original = pd.read_csv(STATS / "spatial_contrasts.csv", keep_default_na=False)
    original = original[original["response"] == "c"].reset_index(drop=True)
    values = paired_values(panel, weights)

    @lru_cache(maxsize=None)
    def grid_for(mask: tuple[bool, ...], model: str) -> CovarianceGrid:
        indices = np.flatnonzero(mask)
        return CovarianceGrid(distances[np.ix_(indices, indices)], model)

    rows = []
    for index, row in metadata.iterrows():
        mask = np.isfinite(values[:, index])
        previous = original[(original[KEYS] == row[KEYS]).all(axis=1)]
        seed = SPATIAL_SEED + index + 1000
        normal_draws = np.random.default_rng(seed).standard_normal(
            (RESAMPLES, mask.sum())
        )
        family_size = int((metadata["family"] == row["family"]).sum())
        for model in MODELS:
            if mask.all():
                results = previous[previous["model"] == model].to_dict("records")
            else:
                results = bootstrap_contrast(
                    values[mask, index],
                    grid_for(tuple(mask), model),
                    normal_draws,
                    family_size,
                )
            for result in results:
                rows.append(
                    {
                        **result,
                        **row.to_dict(),
                        "n_sites": int(mask.sum()),
                        "seed": seed,
                        "analysis": "excluded",
                    }
                )
    excluded = pd.DataFrame(rows)
    excluded["spatial_holm_p"] = excluded.groupby(["family", "model", "estimator"])[
        "p_bootstrap"
    ].transform(lambda p: multipletests(p, method="holm")[1])
    original["analysis"] = "full"
    original["n_sites"] = 25
    columns = [
        *KEYS,
        "analysis",
        "model",
        "estimator",
        "n_sites",
        "estimate_spatial",
        "se_spatial",
        "p_bootstrap",
        "spatial_holm_p",
        "pointwise_lower",
        "pointwise_upper",
        "bonferroni_lower",
        "bonferroni_upper",
        "range_m",
        "spatial_fraction",
        "seed",
        "resamples",
    ]
    return pd.concat([original[columns], excluded[columns]], ignore_index=True)


def main() -> None:
    panel = site_panel(load_samples(), "c")
    panels = {"full": panel, "excluded": np.where(panel < THRESHOLD, np.nan, panel)}
    metadata, weights = make_contrasts()
    contrasts, globals_df = independent_results(panels, metadata, weights)
    STATS.mkdir(parents=True, exist_ok=True)
    globals_df.to_csv(STATS / "carbon_exclusion_global_tests.csv", index=False)
    contrasts.to_csv(STATS / "carbon_exclusion_contrasts.csv", index=False)
    spatial = spatial_results(panels["excluded"], metadata, weights)
    spatial.to_csv(STATS / "carbon_exclusion_spatial.csv", index=False)


if __name__ == "__main__":
    main()
