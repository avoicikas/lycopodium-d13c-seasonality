"""Spearman permutation tests of site means against environmental descriptors (Table 2, S5)."""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from config import PREDICTORS, RESAMPLES, RESPONSES, SEED, STATS
from prepare_data import load_samples


def standardized_ranks(values: np.ndarray) -> np.ndarray:
    ranks = stats.rankdata(values, axis=0)
    ranks -= ranks.mean(axis=0)
    return ranks / np.linalg.norm(ranks, axis=0)


def main() -> None:
    STATS.mkdir(parents=True, exist_ok=True)
    df = load_samples()
    sites = df.groupby("location", as_index=False)[RESPONSES].mean()
    descriptors = df[["location", "releve_id", *PREDICTORS]].drop_duplicates()
    sites = sites.merge(descriptors, on="location").sort_values("location")
    x = standardized_ranks(sites[PREDICTORS].to_numpy())
    y = standardized_ranks(sites[RESPONSES].to_numpy())
    observed = x.T @ y
    rng = np.random.default_rng(SEED + 1)
    orders = rng.permuted(np.tile(np.arange(len(sites)), (RESAMPLES, 1)), axis=1)
    permuted = np.einsum("nk,bnl->bkl", x, y[orders])
    p = (1 + (np.abs(permuted) >= np.abs(observed) - 1e-14).sum(axis=0)) / (
        RESAMPLES + 1
    )
    result = pd.DataFrame(
        [
            {
                "predictor": predictor,
                "response": response,
                "n_sites": len(sites),
                "spearman_rho": observed[i, j],
                "permutation_p": p[i, j],
            }
            for i, predictor in enumerate(PREDICTORS)
            for j, response in enumerate(RESPONSES)
        ]
    )
    result["holm_p_ten_tests"] = multipletests(result["permutation_p"], method="holm")[
        1
    ]
    result.to_csv(STATS / "environmental_correlations.csv", index=False)
    sites.to_csv(STATS / "site_means.csv", index=False)


if __name__ == "__main__":
    main()
