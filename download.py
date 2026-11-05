"""Download the published dataset from Zenodo (doi:10.5281/zenodo.23452117)."""

from urllib.request import urlretrieve

from config import COORDINATES_FILE, DATA, SAMPLES_FILE, VEGETATION_FILE, ZENODO


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for path in (SAMPLES_FILE, VEGETATION_FILE, COORDINATES_FILE):
        if not path.exists():
            urlretrieve(f"{ZENODO}/{path.name}?download=1", path)


if __name__ == "__main__":
    main()
