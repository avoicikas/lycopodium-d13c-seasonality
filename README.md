# Organ-specific δ¹³C of *Lycopodium annotinum* across an annual cycle — analysis code

Reproduces the statistics, tables and figures of the article and its supplement from the
dataset at https://doi.org/10.5281/zenodo.23452117 (downloaded automatically into `data/`).

```
pixi run analysis        # or: python run_all.py (Python 3.14 with the packages in pixi.toml)
```

Outputs are written to `results/`: `stats/` (all computed results), `tables/` (Tables 1–2,
S1–S9 and in-text values, formatted as printed) and `figures/` (Figures 1–2, S1–S7).
Figure 3 is a photograph. Figure S6 needs internet access for OpenStreetMap tiles.
`measurement_dates.csv` gives the EA-IRMS measurement date of each sample (Section 4.4).

| Step | Output |
|---|---|
| `download.py` | Zenodo files |
| `prepare_data.py` | sample table with vegetation descriptors, site coordinates |
| `bootstrap.py` | cell means, 140 paired contrasts per response, global tests |
| `mixed_models.py` | REML mixed models, variance components, residuals |
| `environment.py` | Spearman permutation tests |
| `spatial_autocorrelation.py` | Moran's I |
| `spatial_contrasts.py` | contrasts under spatial covariance models |
| `carbon_exclusion.py` | sensitivity to the three C < 30% samples |
| `tables.py`, `figures.py`, `sampling_map.py` | tables and figures |

Code: MIT License (`LICENSE`). `measurement_dates.csv`: CC BY 4.0, like the Zenodo dataset.
