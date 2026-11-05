"""Figure S6: the 25 collection sites on OpenStreetMap tiles (downloaded once, cached in data/)."""

import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from pyproj import Transformer

from config import DATA, FIGURES
from prepare_data import load_coordinates

CRS = "EPSG:3346"
COLOR = "#a22132"


def basemap(ax: Axes, bounds: tuple[float, ...], name: str, zoom: int) -> None:
    """Show OSM tiles covering bounds (LKS-94), fetching them on the first run."""
    west, south, east, north = bounds
    raster = DATA / f"osm_{name}_z{zoom}.tif"
    if not raster.exists():
        tile_bounds = Transformer.from_crs(
            CRS, "EPSG:3857", always_xy=True
        ).transform_bounds(west, south, east, north)
        ctx.bounds2raster(
            *tile_bounds,
            path=str(raster),
            zoom=zoom,
            source=ctx.providers.OpenStreetMap.Mapnik,
            headers={
                "User-Agent": "LycopodiumSamplingMap/1.0 (scientific figure; contextily)"
            },
            n_connections=1,
            wait=1,
        )
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ctx.add_basemap(ax, source=str(raster), crs=CRS, attribution=False, alpha=0.85)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#666666")


def decorations(ax: Axes, length: int) -> None:
    """Metric scale bar and a true-north arrow (the grid is not aligned with north)."""
    west, east = ax.get_xlim()
    south, north = ax.get_ylim()
    x = west + 0.07 * (east - west)
    y = south + 0.065 * (north - south)
    ax.plot([x, x + length], [y, y], color="white", linewidth=7, zorder=8)
    ax.plot([x, x + length], [y, y], color="#222222", linewidth=3, zorder=9)
    ax.text(
        x + length / 2,
        y + 0.022 * (north - south),
        f"{length // 1000} km" if length >= 1000 else f"{length} m",
        ha="center",
        fontsize=9,
        zorder=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )
    x, y = west + 0.9 * (east - west), south + 0.82 * (north - south)
    lon, lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(x, y)
    nx, ny = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(
        lon, lat + 0.001
    )
    size = 0.09 * (north - south) / np.hypot(nx - x, ny - y)
    tip = (x + size * (nx - x), y + size * (ny - y))
    ax.annotate(
        "",
        xy=tip,
        xytext=(x, y),
        arrowprops={"facecolor": "#222222", "edgecolor": "white", "width": 3},
        zorder=10,
    )
    ax.text(
        tip[0],
        tip[1] + 0.018 * (north - south),
        "N",
        ha="center",
        fontsize=11,
        fontweight="bold",
        zorder=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
    )


def main() -> None:
    sites = load_coordinates()
    cx = (sites["easting"].min() + sites["easting"].max()) / 2
    cy = (sites["northing"].min() + sites["northing"].max()) / 2
    detail = (cx - 160, cy - 160, cx + 160, cy + 160)
    context = (cx - 1400, cy - 1000, cx + 600, cy + 1000)
    national = (290000, 5950000, 710000, 6290000)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(12, 15.5))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(1.55, 1),
        left=0.045,
        right=0.975,
        bottom=0.075,
        top=0.94,
        wspace=0.12,
        hspace=0.12,
    )
    locator, local, stand = (
        fig.add_subplot(grid[0, :]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    )
    for ax, bounds, name, zoom in zip(
        (locator, local, stand),
        (national, context, detail),
        ("lithuania", "context", "detail"),
        (7, 15, 18),
    ):
        basemap(ax, bounds, name, zoom)
    locator.set_title("A  Location within Lithuania", loc="left", fontsize=12, pad=10)
    locator.scatter(
        [cx],
        [cy],
        s=220,
        marker="*",
        c=COLOR,
        edgecolor="white",
        linewidth=0.8,
        zorder=8,
    )
    locator.annotate(
        "Gėla Lake\nSampling location (B–C)",
        xy=(cx, cy),
        xytext=(-28, -42),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=COLOR,
        zorder=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 3},
        arrowprops={"arrowstyle": "-", "color": COLOR, "lw": 1},
    )
    decorations(locator, 100000)
    local.set_title("B  Gėla Lake and the study area", loc="left", fontsize=12, pad=10)
    stand.set_title(
        "C  Collection sites within the stand", loc="left", fontsize=12, pad=10
    )
    local.scatter(sites["easting"], sites["northing"], s=10, c=COLOR, zorder=5)
    local.add_patch(
        Rectangle(
            detail[:2], 320, 320, fill=False, edgecolor=COLOR, linewidth=1.5, zorder=6
        )
    )
    local.annotate(
        "Panel C",
        xy=(detail[2], detail[1]),
        xytext=(6, -12),
        textcoords="offset points",
        color=COLOR,
        fontweight="bold",
        fontsize=9,
    )
    stand.scatter(
        sites["easting"],
        sites["northing"],
        s=30,
        c=COLOR,
        edgecolor="white",
        linewidth=0.7,
        zorder=6,
    )
    # Label offsets (points) keep neighbouring site numbers apart.
    custom = {14: (12, 9), 16: (-14, 4), 15: (16, -2), 17: (12, -8), 18: (-14, -4)}
    for site, row in sites.iterrows():
        dx, dy = custom.get(site, (12 if site % 2 else -14, 0))
        stand.annotate(
            str(site),
            xy=(row["easting"], row["northing"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else "right",
            va="center",
            fontsize=9,
            fontweight="bold",
            zorder=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1},
            arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.5},
        )
    decorations(local, 500)
    decorations(stand, 50)
    fig.text(
        0.025,
        0.99,
        "Sampling locations near Gėla Lake, Lithuania",
        fontsize=15,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.025,
        0.047,
        "Basemap © OpenStreetMap contributors",
        fontsize=8,
        color="#444444",
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "Figure_S6.png", dpi=300, facecolor="white")


if __name__ == "__main__":
    main()
