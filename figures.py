"""Figures 1, 2, S1-S5 and S7A-D from the saved statistics tables."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy import stats

from bootstrap import CELLS, site_panel
from config import (
    FIGURES,
    PART_NAMES,
    PARTS,
    RESPONSES,
    SEASON_NAMES,
    SEASONS,
    SEED,
    STATS,
)
from prepare_data import load_samples

MM = 1 / 25.4
COLORS = {
    "S": "#D55E00",
    "PS": "#E69F00",
    "PSL": "#009E73",
    "OS": "#555555",
    "OSL": "#0072B2",
}
MARKERS = {"S": "o", "PS": "s", "PSL": "^", "OS": "D", "OSL": "v"}
AXIS_LABELS = {"d13c": r"$\delta^{13}$C (‰)", "c": "Carbon concentration (% dry mass)"}
PREDICTOR_LABELS = {
    "soil_moisture": "Soil moisture\n(g g⁻¹)",
    "shannon": "Shannon diversity",
    "simpson": "Simpson diversity (1 − D)",
}
SPATIAL_MODELS = [
    "original",
    "iid",
    "exponential",
    "matern32",
    "exponential_10m",
    "exponential_100m",
]
SPATIAL_LABELS = [
    "Original",
    "IID Gaussian",
    "Exponential",
    "Matérn 3/2",
    "Exp. 10 m",
    "Exp. 100 m",
]
SPATIAL_COLORS = ["#333333", "#888888", "#0072B2", "#009E73", "#D55E00", "#CC79A7"]


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(STATS / f"{name}.csv")


def finish_axis(ax: plt.Axes) -> None:
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)
    ax.tick_params(length=3, width=0.6)


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def figure_1() -> None:
    """Seasonal equal-site means with whole-site bootstrap intervals."""
    means = read("cell_means")
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 120 * MM))
    for response, ax, heading in zip(
        RESPONSES, axes, ["A  Carbon isotope composition", "B  Carbon concentration"]
    ):
        for index, part in enumerate(PARTS):
            cells = means[(means["response"] == response) & (means["part"] == part)]
            cells = cells.set_index("season").loc[SEASONS]
            estimate = cells["mean"].to_numpy()
            errors = np.vstack(
                [estimate - cells["ci95_lower"], cells["ci95_upper"] - estimate]
            )
            ax.errorbar(
                np.arange(4) + (index - 2) * 0.045,
                estimate,
                yerr=errors,
                color=COLORS[part],
                marker=MARKERS[part],
                markersize=4,
                capsize=2,
                elinewidth=0.8,
                label=PART_NAMES[part],
            )
        ax.set_xticks(
            np.arange(4),
            ["Summer\n2021", "Autumn\n2021", "Winter\n2022", "Spring\n2022"],
        )
        ax.set_xlim(-0.25, 3.25)
        ax.set_ylabel(AXIS_LABELS[response])
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.set_title(heading, loc="left", fontweight="bold", pad=10)
        finish_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.54, 0.01),
        columnspacing=1.7,
    )
    fig.subplots_adjust(left=0.1, right=0.99, top=0.9, bottom=0.28, wspace=0.34)
    save(fig, "Figure_1")


def figure_2() -> None:
    """Autumn, winter and spring minus summer, with simultaneous 95% intervals."""
    contrasts = read("contrasts")
    contrasts = contrasts[
        (contrasts["family"] == "seasonal") & (contrasts["earlier"] == "summer")
    ]
    fig, axes = plt.subplots(
        2, 3, figsize=(180 * MM, 142 * MM), sharex="row", sharey=True
    )
    for row, response in enumerate(RESPONSES):
        for col, season in enumerate(SEASONS[1:]):
            ax = axes[row, col]
            cells = contrasts[
                (contrasts["response"] == response) & (contrasts["later"] == season)
            ]
            cells = cells.set_index("part").loc[PARTS]
            ax.axvline(0, color="#888888", linestyle="--", linewidth=0.8)
            for y, part in enumerate(PARTS):
                value = cells.loc[part]
                lower, upper = value["simultaneous_lower"], value["simultaneous_upper"]
                ax.hlines(y, lower, upper, color=COLORS[part], linewidth=1.1)
                ax.vlines(
                    [lower, upper],
                    y - 0.065,
                    y + 0.065,
                    color=COLORS[part],
                    linewidth=0.8,
                )
                ax.plot(
                    value["estimate"],
                    y,
                    marker=MARKERS[part],
                    markersize=5,
                    markeredgecolor=COLORS[part],
                    markeredgewidth=1,
                    markerfacecolor=COLORS[part]
                    if value["familywise_p"] <= 0.05
                    else "white",
                )
            ax.set_yticks(
                range(5), [PART_NAMES[p].replace(" ", "\n", 1) for p in PARTS]
            )
            ax.set_title(
                f"{'ABCDEF'[row * 3 + col]}  {SEASON_NAMES[season]} − summer",
                loc="left",
                fontsize=8,
                fontweight="bold",
                pad=10,
            )
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.set_xlabel(
                r"$\Delta\delta^{13}$C (‰)"
                if response == "d13c"
                else "Carbon (percentage points)",
                fontsize=7,
            )
            finish_axis(ax)
    axes[0, 0].set_ylim(4.6, -0.6)
    legend = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color="#444444",
            markersize=4,
            label="Familywise P ≤ 0.05",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color="#444444",
            markerfacecolor="white",
            markersize=4,
            label="Familywise P > 0.05",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.56, 0.005),
    )
    fig.subplots_adjust(
        left=0.2, right=0.99, bottom=0.14, top=0.94, wspace=0.18, hspace=0.52
    )
    save(fig, "Figure_2")


def figure_s1(samples: pd.DataFrame) -> None:
    """Every measurement with descriptive boxplots."""
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, 5, figsize=(180 * MM, 150 * MM), sharey="row")
    for row, response in enumerate(RESPONSES):
        for col, part in enumerate(PARTS):
            ax = axes[row, col]
            cells = samples[samples["part"] == part]
            sns.boxplot(
                data=cells,
                x="season",
                y=response,
                order=SEASONS,
                color=COLORS[part],
                saturation=0.5,
                width=0.58,
                linewidth=0.6,
                showfliers=False,
                ax=ax,
                boxprops={"alpha": 0.25},
            )
            for index, season in enumerate(SEASONS):
                values = cells.loc[cells["season"] == season, response]
                ax.scatter(
                    index + rng.uniform(-0.17, 0.17, len(values)),
                    values,
                    s=7,
                    color=COLORS[part],
                    alpha=0.65,
                    linewidths=0.15,
                    edgecolors="white",
                    zorder=3,
                )
            ax.set_title(
                f"{'ABCDEFGHIJ'[row * 5 + col]}\n{PART_NAMES[part].replace(' ', chr(10), 1)}",
                fontsize=7.5,
                fontweight="bold",
                loc="left",
                pad=8,
            )
            ax.set_xticks(range(4), ["Su", "Au", "Wi", "Sp"])
            ax.set_xlabel("")
            ax.set_ylabel(AXIS_LABELS[response] if col == 0 else "")
            ax.yaxis.set_major_locator(MaxNLocator(5))
            finish_axis(ax)
    fig.subplots_adjust(
        left=0.1, right=0.995, top=0.88, bottom=0.08, wspace=0.15, hspace=0.46
    )
    save(fig, "Figure_S1")


def figure_s2() -> None:
    """Conditional residuals against fitted values and normal quantile plots."""
    residuals = read("model_residuals")
    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 150 * MM))
    for row, response in enumerate(RESPONSES):
        selected = residuals[residuals["response"] == response]
        left, right = axes[row]
        for part in PARTS:
            organ = selected[selected["part"] == part]
            left.scatter(
                organ["fitted"],
                organ["standardized_residual"],
                s=7,
                alpha=0.6,
                color=COLORS[part],
                linewidths=0,
                label=PART_NAMES[part],
            )
        left.axhline(0, color="#555555", linewidth=0.7, linestyle="--")
        left.set_xlabel(
            "Fitted " + (r"$\delta^{13}$C (‰)" if response == "d13c" else "carbon (%)")
        )
        left.set_ylabel("Standardized conditional residual")
        (theoretical, ordered), _ = stats.probplot(selected["standardized_residual"])
        right.scatter(
            theoretical, ordered, s=7, color="#487898", linewidths=0, alpha=0.65
        )
        # Reference line through the quartiles, as in a standard Q-Q plot.
        observed = np.quantile(ordered, [0.25, 0.75])
        normal = stats.norm.ppf([0.25, 0.75])
        slope = np.diff(observed).item() / np.diff(normal).item()
        right.plot(
            theoretical,
            observed[0] + slope * (theoretical - normal[0]),
            color="#555555",
            linewidth=0.8,
            linestyle="--",
        )
        right.set_xlabel("Theoretical normal quantile")
        right.set_ylabel("Ordered standardized residual")
        for col, ax in enumerate([left, right]):
            label = "Isotopes" if response == "d13c" else "Carbon concentration"
            ax.set_title(
                f"{'ABCD'[row * 2 + col]}  {label}", loc="left", fontweight="bold"
            )
            finish_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.55, 0.01),
        fontsize=7,
    )
    fig.subplots_adjust(
        left=0.12, right=0.99, top=0.94, bottom=0.2, wspace=0.37, hspace=0.46
    )
    save(fig, "Figure_S2")


def figure_s3() -> None:
    """Shares of conditional unexplained variance."""
    variance = read("variance_components")
    colors = ["#0072B2", "#E69F00", "#BBBBBB"]
    labels = ["Collection site", "Sampling event within site", "Residual"]
    fig, ax = plt.subplots(figsize=(160 * MM, 82 * MM))
    left = np.zeros(2)
    for component, color, label in zip(["site", "event", "residual"], colors, labels):
        shares = variance[variance["component"] == component].set_index("response")
        values = shares.loc[RESPONSES, "share"].to_numpy() * 100
        ax.barh([1, 0], values, left=left, height=0.48, color=color, label=label)
        for y, origin, value in zip([1, 0], left, values):
            if value > 3:
                ax.text(
                    origin + value / 2,
                    y,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if component == "site" else "#222222",
                )
        left += values
    ax.set_yticks([1, 0], [r"$\delta^{13}$C", "Carbon concentration"])
    ax.set_xlabel("Share of conditional unexplained variance (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 1.6)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True)
    fig.legend(
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.58, 0.01),
        fontsize=7,
    )
    fig.subplots_adjust(left=0.24, right=0.99, bottom=0.34, top=0.94)
    save(fig, "Figure_S3")


def figure_s4(samples: pd.DataFrame) -> None:
    """Organ-pair d13C gaps per site, for pairs with a supported date change."""
    contrasts = read("contrasts")
    contrasts = contrasts[contrasts["response"] == "d13c"]
    supported = contrasts[
        (contrasts["family"] == "interaction") & (contrasts["familywise_p"] <= 0.05)
    ]
    pairs = list(dict.fromkeys(zip(supported["part"], supported["versus"])))
    sites = pd.DataFrame(
        site_panel(samples, "d13c"), columns=pd.MultiIndex.from_tuples(CELLS)
    )
    short = {season: SEASON_NAMES[season][:2] for season in SEASONS}
    fig, axes = plt.subplots(1, len(pairs), figsize=(180 * MM, 70 * MM), sharey=True)
    for index, (ax, (part, versus)) in enumerate(zip(axes, pairs)):
        for values in (
            sites.xs(part, axis=1, level=1) - sites.xs(versus, axis=1, level=1)
        )[SEASONS].to_numpy():
            ax.plot(range(4), values, color="#BBBBBB", lw=0.5, zorder=1)
        organ = contrasts[
            (contrasts["family"] == "organ")
            & (contrasts["part"] == part)
            & (contrasts["versus"] == versus)
        ].set_index("later")
        estimate = organ.loc[SEASONS, "estimate"].to_numpy()
        ax.errorbar(
            range(4),
            estimate,
            yerr=[
                estimate - organ.loc[SEASONS, "ci95_lower"].to_numpy(),
                organ.loc[SEASONS, "ci95_upper"].to_numpy() - estimate,
            ],
            color="black",
            marker="o",
            ms=4,
            capsize=2.5,
            zorder=3,
        )
        ax.axhline(0, color="#555555", lw=0.6, ls="--", zorder=0)
        changes = supported[
            (supported["part"] == part) & (supported["versus"] == versus)
        ]
        note = "\n".join(
            f"{short[row.later]} vs {short[row.earlier]}: "
            f"{row.estimate:+.2f}‰, P = {row.familywise_p:.5f}".replace("-", "−", 1)
            for row in changes.itertuples()
        )
        ax.text(
            0.97,
            0.97,
            note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
            zorder=4,
        )
        ax.set_xticks(range(4), [SEASON_NAMES[season] for season in SEASONS])
        ax.set_title(
            f"{chr(65 + index)}  {PART_NAMES[part]} − {PART_NAMES[versus].lower()}",
            loc="left",
            fontsize=8,
            fontweight="bold",
        )
        finish_axis(ax)
    axes[0].set_ylabel(r"Difference in $\delta^{13}$C (‰)")
    fig.tight_layout()
    save(fig, "Figure_S4")


def figure_s5() -> None:
    """Site means against vegetation and soil predictors, with Holm-adjusted tests."""
    means = read("site_means")
    correlations = read("environmental_correlations")
    fig, axes = plt.subplots(2, 3, figsize=(180 * MM, 140 * MM), sharey="row")
    for row, response in enumerate(RESPONSES):
        for col, predictor in enumerate(PREDICTOR_LABELS):
            ax = axes[row, col]
            ax.scatter(
                means[predictor],
                means[response],
                s=20,
                color="#487898",
                edgecolors="white",
                linewidths=0.4,
                alpha=0.9,
            )
            result = correlations[
                (correlations["predictor"] == predictor)
                & (correlations["response"] == response)
            ].iloc[0]
            ax.text(
                0.04,
                0.98,
                f"ρ = {result['spearman_rho']:.2f}\nHolm P = {result['holm_p_ten_tests']:.3f}",
                transform=ax.transAxes,
                va="top",
                fontsize=7,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.8,
                    "pad": 1,
                },
            )
            ax.set_title("ABCDEF"[row * 3 + col], loc="left", fontweight="bold")
            ax.set_xlabel(PREDICTOR_LABELS[predictor])
            ax.set_ylabel(AXIS_LABELS[response] if col == 0 else "")
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.margins(x=0.09, y=0.23)
            finish_axis(ax)
    fig.subplots_adjust(
        left=0.11, right=0.99, top=0.95, bottom=0.12, wspace=0.15, hspace=0.43
    )
    save(fig, "Figure_S5")


def spatial_title(row: pd.Series) -> str:
    part = PART_NAMES[row.part]
    if row.family == "seasonal":
        return f"{part}\n{SEASON_NAMES[row.later]} − {SEASON_NAMES[row.earlier]}"
    pair = f"{part} −\n{PART_NAMES[row.versus]}"
    if row.family == "organ":
        return pair + " (date-averaged)"
    return pair + f"\n{SEASON_NAMES[row.later]} − {SEASON_NAMES[row.earlier]}"


def forest(
    selected: pd.DataFrame, frame: pd.DataFrame, letter: str, heading: str
) -> None:
    """Pointwise intervals and adjusted p-values for the original and spatial models."""
    rows = (len(selected) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(11.8, rows * 3.5 + 1.1), squeeze=False)
    fig.subplots_adjust(
        left=0.17, right=0.88, top=0.88, bottom=0.20, hspace=1.05, wspace=1.5
    )
    for ax, (_, item) in zip(axes.flat, selected.iterrows()):
        group = frame.loc[frame["contrast_id"].eq(item["contrast_id"])].set_index(
            "model"
        )
        first = group.iloc[0]
        low, high = [], []
        for number, model in enumerate(SPATIAL_MODELS):
            row = first if model == "original" else group.loc[model]
            estimate = row.estimate if model == "original" else row.estimate_spatial
            lower = row.ci95_lower if model == "original" else row.pointwise_lower
            upper = row.ci95_upper if model == "original" else row.pointwise_upper
            p_value = row.familywise_p if model == "original" else row.spatial_holm_p
            y = 5 - number
            color = SPATIAL_COLORS[number]
            ax.plot([lower, upper], [y, y], color=color, linewidth=1.5)
            ax.plot(
                estimate,
                y,
                "o",
                ms=5.5,
                mec=color,
                mfc=color if p_value <= 0.05 else "white",
            )
            ax.text(
                1.04,
                y,
                f"{p_value:.4f}" if p_value >= 0.0001 else "<0.0001",
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=13,
            )
            low.append(lower)
            high.append(upper)
        span = max(high) - min(low)
        ax.set_xlim(min(min(low), 0) - 0.07 * span, max(max(high), 0) + 0.07 * span)
        ax.axvline(0, color="#777777", lw=0.8, ls="--", zorder=0)
        ax.set_yticks(range(6), SPATIAL_LABELS[::-1], fontsize=13)
        ax.set_ylim(-0.6, 5.6)
        ax.set_title(spatial_title(item), fontsize=13, loc="left", pad=22)
        ax.text(1.04, 1.03, "Adj. p", transform=ax.transAxes, fontsize=13)
        ax.set_xlabel("δ¹³C difference (‰)", fontsize=13)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=12)
    for ax in list(axes.flat)[len(selected) :]:
        ax.set_visible(False)
    fig.suptitle(
        f"Figure S7{letter}. {heading}", fontsize=17, x=0.025, ha="left", y=0.99
    )
    fig.text(
        0.025,
        0.045,
        "Bars: pointwise 95% intervals, not simultaneous intervals. Axis ranges vary.\n"
        "Filled points: family-adjusted p ≤ 0.05. Original: independent-site max-t.\n"
        "Other scenarios: refitted Gaussian bootstrap with Holm adjustment.",
        fontsize=13,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"Figure_S7{letter}.png", dpi=300, facecolor="white")
    plt.close(fig)


def figures_s7() -> None:
    """Spatial-sensitivity forest plots for the originally supported d13C contrasts."""
    frame = pd.read_csv(STATS / "spatial_contrasts.csv", keep_default_na=False)
    equal = frame.loc[
        frame["estimator"].eq("equal_weight") & frame["response"].eq("d13c")
    ]
    originals = equal.drop_duplicates("contrast_id").sort_values("contrast_id")
    supported = originals["familywise_p"].le(0.05)
    dates = originals.loc[originals["family"].eq("seasonal") & supported]
    interactions = originals.loc[originals["family"].eq("interaction") & supported]
    organs = originals.loc[
        originals["family"].eq("organ") & originals["later"].eq("all_seasons")
    ]
    forest(dates, equal, "A", "Originally supported within-organ date contrasts")
    forest(interactions, equal, "B", "Originally supported organ-by-date contrasts")
    forest(organs.iloc[:6], equal, "C", "Date-averaged organ contrasts (1–6)")
    forest(organs.iloc[6:], equal, "D", "Date-averaged organ contrasts (7–10)")


def main() -> None:
    sns.set_theme(
        style="whitegrid", context="paper", font="DejaVu Sans", font_scale=0.9
    )
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.45,
            "lines.linewidth": 1.1,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    samples = load_samples()
    figure_1()
    figure_2()
    figure_s1(samples)
    figure_s2()
    figure_s3()
    figure_s4(samples)
    figure_s5()
    with plt.style.context("default"):
        figures_s7()


if __name__ == "__main__":
    main()
