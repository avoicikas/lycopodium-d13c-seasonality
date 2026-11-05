"""Equal-site means and paired contrasts with a whole-site bootstrap (cell means, Table S8)."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from config import PARTS, RESAMPLES, RESPONSES, SEASONS, SEED, STATS
from prepare_data import load_samples

CELLS = [(season, part) for season in SEASONS for part in PARTS]


def site_panel(df: pd.DataFrame, response: str) -> np.ndarray:
    """Sites (sorted) by the 20 season x part cells."""
    panel = df.pivot(index="location", columns=["season", "part"], values=response)
    return (
        panel.reindex(columns=pd.MultiIndex.from_tuples(CELLS)).sort_index().to_numpy()
    )


def make_contrasts() -> tuple[pd.DataFrame, np.ndarray]:
    """30 seasonal, 50 organ and 60 interaction contrasts as weights over CELLS."""
    rows, weights = [], []

    def add(family, part, later, earlier, versus, terms):
        vector = np.zeros(len(CELLS))
        for season, organ, weight in terms:
            vector[CELLS.index((season, organ))] += weight
        rows.append(
            {
                "family": family,
                "part": part,
                "versus": versus,
                "later": later,
                "earlier": earlier,
            }
        )
        weights.append(vector)

    for earlier, later in combinations(SEASONS, 2):
        for part in PARTS:
            add(
                "seasonal",
                part,
                later,
                earlier,
                "",
                [(later, part, 1), (earlier, part, -1)],
            )
    for part, other in combinations(PARTS, 2):
        for season in SEASONS:
            add(
                "organ",
                part,
                season,
                "",
                other,
                [(season, part, 1), (season, other, -1)],
            )
        add(
            "organ",
            part,
            "all_seasons",
            "",
            other,
            [(s, o, w) for s in SEASONS for o, w in [(part, 0.25), (other, -0.25)]],
        )
        for earlier, later in combinations(SEASONS, 2):
            add(
                "interaction",
                part,
                later,
                earlier,
                other,
                [
                    (later, part, 1),
                    (later, other, -1),
                    (earlier, part, -1),
                    (earlier, other, 1),
                ],
            )
    return pd.DataFrame(rows), np.asarray(weights)


def bootstrap_family(
    values: np.ndarray, draws: np.ndarray
) -> tuple[pd.DataFrame, dict]:
    """Bootstrap-t intervals and max-|t| familywise p-values; rows are sites."""
    n = len(values)
    estimate = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / np.sqrt(n)
    samples = values[draws]
    boot_t = (samples.mean(axis=1) - estimate) / (
        samples.std(axis=1, ddof=1) / np.sqrt(n)
    )
    maximum = np.abs(boot_t).max(axis=1)
    critical = np.quantile(maximum, 0.95)
    observed_t = estimate / se
    count = len(draws)
    interval = np.quantile(boot_t, [0.025, 0.975], axis=0)
    leave_one_out = (values.sum(axis=0) - values) / (n - 1)
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
            "paired_t_p_sensitivity": 2 * stats.t.sf(np.abs(observed_t), df=n - 1),
            "median_site_difference": np.median(values, axis=0),
            "leave_one_site_out_min": leave_one_out.min(axis=0),
            "leave_one_site_out_max": leave_one_out.max(axis=0),
        }
    )
    global_result = {
        "contrasts": values.shape[1],
        "max_abs_t": np.abs(observed_t).max(),
        "critical_max_t_95": critical,
        "global_bootstrap_p": (1 + (maximum >= np.abs(observed_t).max()).sum())
        / (count + 1),
    }
    return result, global_result


def main() -> None:
    STATS.mkdir(parents=True, exist_ok=True)
    df = load_samples()
    n = df["location"].nunique()
    # One set of site draws shared by every cell, contrast and response.
    draws = np.random.default_rng(SEED).integers(0, n, size=(RESAMPLES, n))
    metadata, weights = make_contrasts()
    means, contrasts, globals_ = [], [], []
    for response in RESPONSES:
        values = site_panel(df, response)
        low, high = stats.bootstrap(
            (values,),
            np.mean,
            axis=0,
            n_resamples=RESAMPLES,
            method="percentile",
            rng=np.random.default_rng(SEED),
        ).confidence_interval
        for i, (season, part) in enumerate(CELLS):
            means.append(
                {
                    "response": response,
                    "season": season,
                    "part": part,
                    "mean": values[:, i].mean(),
                    "ci95_lower": low[i],
                    "ci95_upper": high[i],
                    "n_sites": n,
                    "sd_between_sites": values[:, i].std(ddof=1),
                }
            )
        contrast_values = values @ weights.T
        for family, rows in metadata.groupby("family", sort=False):
            result, global_result = bootstrap_family(
                contrast_values[:, rows.index], draws
            )
            result = pd.concat([rows.reset_index(drop=True), result], axis=1)
            result.insert(0, "response", response)
            contrasts.append(result)
            globals_.append({"response": response, "family": family, **global_result})
    global_df = pd.DataFrame(globals_)
    global_df["holm_p_six_global_tests"] = multipletests(
        global_df["global_bootstrap_p"], method="holm"
    )[1]
    pd.DataFrame(means).to_csv(STATS / "cell_means.csv", index=False)
    pd.concat(contrasts, ignore_index=True).to_csv(STATS / "contrasts.csv", index=False)
    global_df.to_csv(STATS / "global_tests.csv", index=False)


if __name__ == "__main__":
    main()
