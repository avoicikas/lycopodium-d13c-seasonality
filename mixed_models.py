"""REML mixed models with site and event-within-site intercepts (Figures S2, S3)."""

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config import RESPONSES, SEASONS, STATS
from prepare_data import load_samples


def fit(df: pd.DataFrame, response: str):
    model = smf.mixedlm(
        f"{response} ~ C(organ) * C(season)",
        df,
        groups=df["location"],
        re_formula="1",
        vc_formula={"event": "0 + C(code)"},
    )
    fits = []
    for method in ["lbfgs", "powell", "nm"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(method=method, maxiter=3000, reml=True)
        if result.converged:
            fits.append(result)
    return model, max(fits, key=lambda r: r.llf)


def main() -> None:
    STATS.mkdir(parents=True, exist_ok=True)
    df = load_samples()
    # Reference levels: orthotropic stems and summer.
    df["organ"] = pd.Categorical(df["part"], categories=["OS", "S", "PS", "PSL", "OSL"])
    df["season"] = pd.Categorical(df["season"], categories=SEASONS, ordered=True)
    variance, coefficients, wald, residuals = [], [], [], []
    for response in RESPONSES:
        model, result = fit(df, response)
        components = {
            "site": result.cov_re.iloc[0, 0],
            **dict(zip(model.exog_vc.names, result.vcomp)),
            "residual": result.scale,
        }
        total = sum(components.values())
        for name, value in components.items():
            variance.append(
                {
                    "response": response,
                    "component": name,
                    "variance": value,
                    "share": value / total,
                }
            )
        ci = result.conf_int()
        for term in result.fe_params.index:
            coefficients.append(
                {
                    "response": response,
                    "term": term,
                    "estimate": result.fe_params[term],
                    "standard_error": result.bse[term],
                    "wald_z_p_unadjusted": result.pvalues[term],
                    "ci95_lower": ci.loc[term, 0],
                    "ci95_upper": ci.loc[term, 1],
                }
            )
        interaction = result.fe_params.index.str.contains(":")
        test = result.wald_test(
            np.eye(len(result.params))[: len(result.fe_params)][interaction],
            scalar=True,
        )
        wald.append(
            {
                "response": response,
                "term": "organ_by_season_interaction",
                "chi_square": float(test.statistic),
                "df": int(interaction.sum()),
                "asymptotic_p": float(test.pvalue),
            }
        )
        residuals.append(
            df[["code", "location", "season", "part"]].assign(
                response=response,
                observed=df[response],
                fitted=result.fittedvalues,
                residual=result.resid,
                standardized_residual=result.resid / np.sqrt(result.scale),
            )
        )
    pd.DataFrame(variance).to_csv(STATS / "variance_components.csv", index=False)
    pd.DataFrame(coefficients).to_csv(
        STATS / "mixed_model_coefficients.csv", index=False
    )
    pd.DataFrame(wald).to_csv(STATS / "mixed_model_wald_tests.csv", index=False)
    pd.concat(residuals, ignore_index=True).to_csv(
        STATS / "model_residuals.csv", index=False
    )


if __name__ == "__main__":
    main()
