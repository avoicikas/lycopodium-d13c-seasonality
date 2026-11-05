"""Moran's I of site-level residuals after removing season x part cell means (Table S1)."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from statsmodels.stats.multitest import multipletests

from config import RESPONSES, STATS
from bootstrap import site_panel
from prepare_data import load_coordinates, load_samples

PERMUTATIONS = 99999
SEED = 44


def adjusted_site_means(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over the 20 cells of the residuals from the season x part cell means."""
    means = {}
    for response in RESPONSES:
        panel = site_panel(df, response)
        means[response] = (panel - panel.mean(axis=0)).mean(axis=1)
    return pd.DataFrame(means, index=sorted(df["location"].unique()))


def neighbour_weights(coordinates: np.ndarray, k: int) -> np.ndarray:
    """Binary k-nearest links (kth-distance ties included), symmetrized by union, row-scaled."""
    distances = cdist(coordinates, coordinates)
    np.fill_diagonal(distances, np.inf)
    thresholds = np.sort(distances, axis=1)[:, k - 1]
    adjacency = distances <= thresholds[:, None]
    adjacency |= adjacency.T
    return adjacency / adjacency.sum(axis=1, keepdims=True)


def moran_i(values: np.ndarray, weights: np.ndarray) -> float:
    """Global Moran's I."""
    z = values - values.mean()
    return float(len(values) / weights.sum() * (z @ weights @ z) / (z @ z))


def main() -> None:
    df = load_samples()
    site_means = adjusted_site_means(df)
    coordinates = load_coordinates().sort_index()[["easting", "northing"]].to_numpy()
    n = len(site_means)
    rng = np.random.default_rng(SEED)
    # The same site permutations are used for both responses and both networks.
    permutations = np.array([rng.permutation(n) for _ in range(PERMUTATIONS)])
    expected = -1 / (n - 1)
    rows = []
    for k in (2, 4):
        weights = neighbour_weights(coordinates, k)
        for response in RESPONSES:
            values = site_means[response].to_numpy()
            observed = moran_i(values, weights)
            z = values - values.mean()
            randomized = z[permutations]
            null = (
                n
                / weights.sum()
                * np.einsum(
                    "bi,ij,bj->b", randomized, weights, randomized, optimize=True
                )
                / (z @ z)
            )
            # Two-sided about E[I] with a plus-one correction; 1e-12 absorbs rounding ties.
            extreme = np.abs(null - expected) >= abs(observed - expected) - 1e-12
            p_value = (1 + np.count_nonzero(extreme)) / (PERMUTATIONS + 1)
            rows.append(
                {
                    "k": k,
                    "response": response,
                    "moran_i": observed,
                    "expected_i": expected,
                    "permutation_p": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["holm_p"] = result.groupby("k")["permutation_p"].transform(
        lambda p: multipletests(p, method="holm")[1]
    )
    STATS.mkdir(parents=True, exist_ok=True)
    result.to_csv(STATS / "moran.csv", index=False)


if __name__ == "__main__":
    main()
