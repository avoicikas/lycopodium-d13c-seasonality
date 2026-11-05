"""Write one CSV per published table, and the in-text statistics, formatted as printed."""

import pandas as pd

from config import PART_NAMES, PARTS, STATS, TABLES
from prepare_data import load_samples

TITLES = {"d13c": "δ¹³C", "c": "Carbon (%)"}
FAMILIES = {
    "seasonal": "Collection date",
    "organ": "Organ",
    "interaction": "Organ × date",
}
DATES = {"summer": "Su", "autumn": "Au", "winter": "Wi", "spring": "Sp"}
MODELS = ["iid", "exponential", "matern32", "exponential_10m", "exponential_100m"]


def minus(text: str) -> str:
    return text.replace("-", "−")


def signed(value: float) -> str:
    return minus(f"{value:+.2f}")


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(STATS / name, keep_default_na=False)


def table_1() -> pd.DataFrame:
    c = read("contrasts.csv")
    c = c[(c.response == "d13c") & (c.family == "organ") & (c.later == "all_seasons")]
    return pd.DataFrame(
        {
            "Organ pair": [
                f"{PART_NAMES[p]} − {PART_NAMES[v].lower()}"
                for p, v in zip(c.part, c.versus)
            ],
            "Difference (‰)": c.estimate.map(signed),
            "Simultaneous 95% interval (‰)": c.simultaneous_lower.map(signed)
            + " to "
            + c.simultaneous_upper.map(signed),
            "Familywise p": c.familywise_p.map("{:.5f}".format),
        }
    )


def table_2() -> pd.DataFrame:
    e = read("environmental_correlations.csv").set_index(["predictor", "response"])
    labels = {
        "soil_moisture": "Soil moisture (g g⁻¹)",
        "shannon": "Shannon diversity",
        "simpson": "Simpson diversity (1 − D)",
        "tree_species_cover_sum": "Summed tree-species cover",
        "species_richness": "Taxon richness",
    }
    rows = [
        [label]
        + [
            f(e.loc[(predictor, r), col])
            for r in ("d13c", "c")
            for col, f in (
                ("spearman_rho", lambda x: minus(f"{x:.3f}")),
                ("holm_p_ten_tests", "{:.5f}".format),
            )
        ]
        for predictor, label in labels.items()
    ]
    columns = ["Predictor", "δ¹³C: ρ", "Adjusted p", "Carbon: ρ", "Adjusted p"]
    return pd.DataFrame(rows, columns=columns)


def table_s1() -> pd.DataFrame:
    m = read("moran.csv")
    return pd.DataFrame(
        {
            "Network": "k=" + m.k.astype(str),
            "Response": m.response.map({"d13c": "δ¹³C", "c": "Carbon (%)"}),
            "Moran’s I": m.moran_i.map("{:.3f}".format),
            "Permutation p": m.permutation_p.map("{:.5f}".format),
            "Holm p": m.holm_p.map("{:.5f}".format),
        }
    )


def short_label(row: pd.Series) -> str:
    if row.family == "seasonal":
        return f"{row.part}: {DATES[row.later]}−{DATES[row.earlier]}"
    pair = f"{row.part}−{row.versus}"
    if row.family == "organ":
        return pair + (
            ": mean" if row.later == "all_seasons" else f": {DATES[row.later]}"
        )
    return f"{pair}: {DATES[row.later]}−{DATES[row.earlier]}"


def tables_s2_s7() -> dict[str, pd.DataFrame]:
    s = read("spatial_contrasts.csv")
    s = s[s.estimator == "equal_weight"]
    out = {}
    for number, (response, family) in enumerate(
        [(r, f) for r in ("d13c", "c") for f in FAMILIES], start=2
    ):
        sel = s[(s.response == response) & (s.family == family)]
        rows = []
        for identifier, g in sel.groupby("contrast_id"):
            first = g.iloc[0]
            p = g.set_index("model").spatial_holm_p
            rows.append(
                [
                    str(identifier),
                    short_label(first),
                    f"{first.estimate:+.3f}",
                    f"{first.familywise_p:.5f}",
                ]
                + [f"{p[m]:.5f}" for m in MODELS]
            )
        out[f"S{number}"] = pd.DataFrame(
            rows,
            columns=[
                "ID",
                "Contrast",
                "Mean difference",
                "Orig. p",
                "IID p",
                "Exp. p",
                "Mat. p",
                "10 m p",
                "100 m p",
            ],
        )
    return out


def table_s8() -> pd.DataFrame:
    g = read("global_tests.csv")
    return pd.DataFrame(
        {
            "Response": g.response.map(TITLES),
            "Comparison family": g.family.map(FAMILIES),
            "Contrasts": g.contrasts.astype(str),
            "Global p": g.global_bootstrap_p.map("{:.5f}".format),
            "Holm-adjusted p": g.holm_p_six_global_tests.map("{:.5f}".format),
        }
    )


def tables_s9() -> tuple[pd.DataFrame, pd.DataFrame]:
    g = read("carbon_exclusion_global_tests.csv")
    g = g[g.response == "c"].set_index(["family", "analysis"]).holm_p_six_global_tests
    effects = [
        ("seasonal", "Collection date"),
        ("organ", "Organ"),
        ("interaction", "Organ × collection date"),
    ]
    a = pd.DataFrame(
        [
            [label] + [f"{g[(f, an)]:.4f}" for an in ("full", "excluded")]
            for f, label in effects
        ],
        columns=[
            "Effect",
            "Full data\nHolm-adjusted p",
            "Three observations excluded\nHolm-adjusted p",
        ],
    )
    c = read("carbon_exclusion_contrasts.csv")

    def cell(row: pd.Series, reverse: bool) -> str:
        est, lo, hi = row.estimate, row.simultaneous_lower, row.simultaneous_upper
        if reverse:
            est, lo, hi = -est, -hi, -lo
        return minus(
            f"{est:+.2f} ({lo:+.2f} to {hi:+.2f})\nn = {int(row.n_sites)} sites"
        )

    winter = c.family.eq("seasonal") & c.part.eq("OSL")
    winter &= c.later.eq("winter") & c.earlier.eq("summer")
    organ = (
        c.family.eq("organ")
        & c.part.eq("OS")
        & c.versus.eq("OSL")
        & c.later.eq("all_seasons")
    )
    analyses = ("full", "excluded")
    winter = [cell(c[winter & c.analysis.eq(an)].iloc[0], False) for an in analyses]
    organ = [cell(c[organ & c.analysis.eq(an)].iloc[0], True) for an in analyses]
    b = pd.DataFrame(
        [
            ["Orthotropic leaves:\nwinter − summer"] + winter,
            ["Orthotropic leaves − orthotropic stems, averaged across dates"] + organ,
        ],
        columns=[
            "Contrast",
            "Full data\nDifference (simultaneous 95% interval)",
            "Three observations excluded\nDifference (simultaneous 95% interval)",
        ],
    )
    return a, b


def text_values() -> pd.DataFrame:
    df = load_samples()
    rows = []

    def add(section: str, quantity: str, value: object) -> None:
        rows.append((section, quantity, value))

    for part in PARTS:
        minimum = df.c[df.part == part].min()
        add("2.2", f"minimum carbon, {PART_NAMES[part].lower()} (%)", f"{minimum:.2f}")
    low = df.c[df.c < 30].sort_values()
    add("2.2", "samples below 30% carbon", len(low))
    for v in low:
        add("2.2", "carbon below 30% (%)", f"{v:.2f}")
    veg = df.drop_duplicates("releve_id")
    add("2.3", "vegetation relevés", len(veg))
    for col, fmt in (
        ("species_richness", "{:.0f}"),
        ("shannon", "{:.2f}"),
        ("simpson", "{:.3f}"),
    ):
        add("2.3", f"{col} minimum", fmt.format(veg[col].min()))
        add("2.3", f"{col} maximum", fmt.format(veg[col].max()))
    v = read("variance_components.csv")
    for r in v.itertuples():
        add(
            "2.4",
            f"variance share, {r.response}, {r.component} (%)",
            f"{100 * r.share:.1f}",
        )
    cells = read("cell_means.csv")
    for season in ("summer", "winter"):
        m = cells[
            (cells.response == "c") & (cells.part == "OSL") & (cells.season == season)
        ]
        add(
            "3.2",
            f"mean carbon, orthotropic leaves, {season} (%)",
            f"{m['mean'].iloc[0]:.2f}",
        )
    c = read("contrasts.csv")
    d = c[c.response == "d13c"]
    within_date = d[(d.family == "organ") & (d.later != "all_seasons")]
    within_organ = d[d.family == "seasonal"]
    add("3.3", "within-date organ contrasts (n)", len(within_date))
    add(
        "3.3",
        "median SE, within-date organ contrasts (‰)",
        f"{within_date.standard_error.median():.2f}",
    )
    add("3.3", "within-organ date contrasts (n)", len(within_organ))
    add(
        "3.3",
        "median SE, within-organ date contrasts (‰)",
        f"{within_organ.standard_error.median():.2f}",
    )
    dates = df.groupby("measurement_date").season.nunique()
    events = df.groupby(["location", "season"]).measurement_date.nunique()
    add("4.4", "calendar dates", len(dates))
    add("4.4", "dates with more than one collection date", int((dates > 1).sum()))
    add("4.4", "events with all organs on one date", int((events == 1).sum()))
    add("4.4", "events", len(events))
    return pd.DataFrame(rows, columns=["section", "quantity", "value"])


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    tables = {
        "1": table_1(),
        "2": table_2(),
        "S1": table_s1(),
        **tables_s2_s7(),
        "S8": table_s8(),
    }
    tables["S9A"], tables["S9B"] = tables_s9()
    for name, frame in tables.items():
        frame.to_csv(TABLES / f"Table_{name}.csv", index=False)
    text_values().to_csv(TABLES / "text_values.csv", index=False)


if __name__ == "__main__":
    main()
