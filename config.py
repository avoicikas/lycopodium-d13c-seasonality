"""Paths and analysis settings shared by all steps."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
STATS = RESULTS / "stats"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

ZENODO = "https://zenodo.org/records/23452117/files"
SAMPLES_FILE = DATA / "lycopodium_seasonality_metadata_carbon.xlsx"
VEGETATION_FILE = DATA / "vegetation_releves_lycop_seasonality.xlsx"
COORDINATES_FILE = DATA / "site_coordinates_lycop_LKS.xlsx"
MEASUREMENT_DATES_FILE = HERE / "measurement_dates.csv"

SEED = 42
RESAMPLES = 19999

SEASONS = ["summer", "autumn", "winter", "spring"]
SEASON_NAMES = dict(zip(SEASONS, ["Summer", "Autumn", "Winter", "Spring"]))
PARTS = ["S", "PS", "PSL", "OS", "OSL"]
PART_NAMES = {
    "S": "Roots",
    "PS": "Plagiotropic stems",
    "PSL": "Plagiotropic leaves",
    "OS": "Orthotropic stems",
    "OSL": "Orthotropic leaves",
}
RESPONSES = ["d13c", "c"]
PREDICTORS = [
    "soil_moisture",
    "shannon",
    "simpson",
    "tree_species_cover_sum",
    "species_richness",
]
