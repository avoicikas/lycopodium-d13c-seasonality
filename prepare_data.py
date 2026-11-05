"""Load samples, derive vegetation descriptors per relevé, and load site coordinates."""

import numpy as np
import pandas as pd

from config import (
    COORDINATES_FILE,
    MEASUREMENT_DATES_FILE,
    PREDICTORS,
    SAMPLES_FILE,
    VEGETATION_FILE,
)


def vegetation_descriptors() -> pd.DataFrame:
    veg = pd.read_excel(VEGETATION_FILE)
    veg["Species"] = veg["Species"].str.strip()
    rows = []
    for releve, plot in veg.groupby("Location"):
        # A taxon recorded in several layers counts once, at its largest cover.
        p = plot.groupby("Species")["Coverage_percent"].max()
        p = p / p.sum()
        rows.append(
            {
                "releve_id": releve,
                "shannon": -(p * np.log(p)).sum(),
                "simpson": 1 - (p**2).sum(),
                "species_richness": plot["Species"].nunique(),
                "tree_species_cover_sum": plot.loc[
                    plot["Layer"].isin(["M1", "M2"]), "Coverage_percent"
                ].sum(),
            }
        )
    return pd.DataFrame(rows)


def load_samples() -> pd.DataFrame:
    df = pd.read_excel(SAMPLES_FILE, sheet_name="Updated_data")
    df = df[
        [
            "code",
            "location",
            "season",
            "part",
            "d13c",
            "c",
            "releve_id",
            "soil_moisture",
        ]
    ]
    df = df.merge(vegetation_descriptors(), on="releve_id", how="left")
    df = df.merge(pd.read_csv(MEASUREMENT_DATES_FILE), on=["code", "part"], how="left")
    # Removes floating-point noise so that sites sharing a relevé tie exactly.
    df[PREDICTORS] = df[PREDICTORS].round(10)
    return df


def load_coordinates() -> pd.DataFrame:
    xy = pd.read_excel(COORDINATES_FILE)
    # In the source workbook x holds northing and y holds easting (LKS-94).
    return (
        pd.DataFrame(
            {"location": xy.iloc[:, 0], "easting": xy["y"], "northing": xy["x"]}
        )
        .set_index("location")
        .sort_index()
    )
