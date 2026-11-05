"""Run the complete analysis: download data, compute statistics, write tables and figures."""

import bootstrap
import carbon_exclusion
import download
import environment
import figures
import mixed_models
import sampling_map
import spatial_autocorrelation
import spatial_contrasts
import tables

STEPS = [
    download,
    bootstrap,
    mixed_models,
    environment,
    spatial_autocorrelation,
    spatial_contrasts,
    carbon_exclusion,
    tables,
    figures,
    sampling_map,
]

if __name__ == "__main__":
    for step in STEPS:
        print(f"Running {step.__name__}", flush=True)
        step.main()
