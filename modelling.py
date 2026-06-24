# ============================================================
# STAGE 7 — OLS REGRESSION MODELLING
# modelling.ipynb  |  Cells 16–23
#
# Input : data/thesis_modelling.csv  (40 rows × 17 cols)
# Output: results/regression_main.csv
#         results/regression_standardized.csv
#         results/vif_table.csv
#         results/robustness_summary.csv
#         figures/interaction_IV1.png
#         figures/interaction_IV2.png
#         figures/interaction_IV3.png
#         figures/assumption_checks.png
# ============================================================


# ============================================================
# CELL 16 — Imports & Setup
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

np.random.seed(42)

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Plotting style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.size":        10,
})
MED_BLUE = "#4C72B0"
MED_PINK = "#C44E52"

df = pd.read_csv(DATA_DIR / "thesis_modelling.csv")
log.info("Loaded modelling dataset: %d rows x %d cols", *df.shape)


# ============================================================
# CELL 17 — Main OLS Regression
#
# Formula uses MEAN-CENTERED IVs (_c suffix) for interaction
# terms to reduce multicollinearity (per thesis spec).
# Fixed effects: C(feature), C(brand)
# Controls: pre_launch_volume, post_review_count
# ============================================================

formula = """
post_sentiment_DV ~
    sentiment_IV1_c +
    pre_expect_score_IV2_c +
    competitor_freq_IV3_c +
    product_type +
    sentiment_IV1_c:product_type +
    pre_expect_score_IV2_c:product_type +
    competitor_freq_IV3_c:product_type +
    C(feature) +
    C(brand) +
    pre_launch_volume +
    post_review_count
"""

model_main = smf.ols(formula=formula, data=df).fit()
print(model_main.summary())


# ── Clean coefficient table ──────────────────────────────────
def build_coef_table(result, label="main"):
    coef  = result.params
    se    = result.bse
    tstat = result.tvalues
    pval  = result.pvalues
    ci    = result.conf_int()

    table = pd.DataFrame({
        "coefficient": coef.round(4),
        "std_error":   se.round(4),
        "t_stat":      tstat.round(3),
        "p_value":     pval.round(4),
        "ci_lower":    ci[0].round(4),
        "ci_upper":    ci[1].round(4),
        "sig":         pval.apply(
            lambda p: "***" if p < .001
                      else ("**" if p < .01
                            else ("*" if p < .05
                                  else ("." if p < .10 else "")))
        ),
    })
    table.index.name = "term"
    out_path = RESULTS_DIR / f"regression_{label}.csv"
    table.to_csv(out_path)
    log.info("Coefficient table saved -> %s", out_path)
    return table

coef_table = build_coef_table(model_main, "main")

# Print hypothesis-relevant terms only
hyp_terms = [
    "sentiment_IV1_c",
    "pre_expect_score_IV2_c",
    "competitor_freq_IV3_c",
    "product_type",
    "sentiment_IV1_c:product_type",
    "pre_expect_score_IV2_c:product_type",
    "competitor_freq_IV3_c:product_type",
]

print("\n── Hypothesis terms ─────────────────────────────────────")
print(coef_table.loc[
    coef_table.index.intersection(hyp_terms)
].to_string())

print(f"\nR-squared      : {model_main.rsquared:.4f}")
print(f"Adj. R-squared : {model_main.rsquared_adj:.4f}")
print(f"F-statistic    : {model_main.fvalue:.3f}  (p = {model_main.f_pvalue:.4f})")
print(f"N              : {int(model_main.nobs)}")


# ============================================================
# CELL 18 — Standardized Beta Coefficients
# Needed for cross-variable effect size comparison
# ============================================================

# Z-score all continuous variables
std_cols = [
    "post_sentiment_DV",
    "sentiment_IV1_c",
    "pre_expect_score_IV2_c",
    "competitor_freq_IV3_c",
    "pre_launch_volume",
    "post_review_count",
]

df_std = df.copy()
for col in std_cols:
    df_std[col] = (df[col] - df[col].mean()) / df[col].std()

model_std = smf.ols(formula=formula, data=df_std).fit()
coef_std  = build_coef_table(model_std, "standardized")

print("\n── Standardized betas (hypothesis terms) ────────────────")
print(coef_std.loc[
    coef_std.index.intersection(hyp_terms),
    ["coefficient", "std_error", "p_value", "sig"]
].rename(columns={"coefficient": "std_beta"}).to_string())


# ============================================================
# CELL 19 — Assumption Checks
# (1) Normality of residuals — Q-Q plot + Shapiro-Wilk
# (2) Homoscedasticity  — fitted vs residuals
# (3) Multicollinearity — VIF on IVs + controls
# ============================================================

residuals = model_main.resid
fitted    = model_main.fittedvalues

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Q-Q plot
sm.qqplot(residuals, line="s", ax=axes[0], alpha=0.7)
axes[0].set_title("Normal Q-Q Plot of Residuals")

# Fitted vs residuals
axes[1].scatter(fitted, residuals, color=MED_BLUE, alpha=0.7, edgecolors="white")
axes[1].axhline(0, color="grey", linestyle="--", linewidth=1)
axes[1].set_xlabel("Fitted values")
axes[1].set_ylabel("Residuals")
axes[1].set_title("Residuals vs Fitted")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "assumption_checks.png", dpi=150, bbox_inches="tight")
plt.show()
log.info("Assumption plots saved")

# Shapiro-Wilk normality test
stat, p_sw = stats.shapiro(residuals)
print(f"\nShapiro-Wilk: W = {stat:.4f}, p = {p_sw:.4f}",
      "-> normality OK" if p_sw > .05 else "-> normality violated (small n, check Q-Q)")

# Breusch-Pagan homoscedasticity test
from statsmodels.stats.diagnostic import het_breuschpagan
bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(residuals, model_main.model.exog)
print(f"Breusch-Pagan: LM = {bp_lm:.4f}, p = {bp_p:.4f}",
      "-> homoscedasticity OK" if bp_p > .05 else "-> heteroscedasticity detected")

# VIF — compute on design matrix (excluding fixed effect dummies for clarity)
vif_cols = [
    "sentiment_IV1_c",
    "pre_expect_score_IV2_c",
    "competitor_freq_IV3_c",
    "product_type",
    "pre_launch_volume",
    "post_review_count",
]
X_vif = sm.add_constant(df[vif_cols])
vif_df = pd.DataFrame({
    "variable": vif_cols,
    "VIF": [variance_inflation_factor(X_vif.values, i + 1)
            for i in range(len(vif_cols))],
}).round(3)
vif_df["flag"] = vif_df["VIF"].apply(
    lambda v: "HIGH (>10)" if v > 10 else ("MODERATE (5-10)" if v > 5 else "OK")
)
print("\n── VIF table ────────────────────────────────────────────")
print(vif_df.to_string(index=False))
vif_df.to_csv(RESULTS_DIR / "vif_table.csv", index=False)
log.info("VIF table saved")


# ============================================================
# CELL 20 — Interaction Plots (H4a, H4b, H4c)
# Simple slopes: separate regression lines for
# hedonic (product_type=1) vs utilitarian (product_type=0)
# ============================================================

def interaction_plot(iv_col, iv_label, hyp_label, ax):
    """Plot DV ~ IV separately for hedonic and utilitarian."""
    for ptype, color, linestyle, label in [
        (1, MED_BLUE,  "-",  "Hedonic"),
        (0, MED_PINK, "--", "Utilitarian"),
    ]:
        subset = df[df["product_type"] == ptype]
        x = subset[iv_col]
        y = subset["post_sentiment_DV"]

        # Fit simple slope
        slope, intercept, r, p, se = stats.linregress(x, y)

        x_range = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_range, intercept + slope * x_range,
                color=color, linestyle=linestyle,
                linewidth=2, label=f"{label} (b={slope:.3f}, p={p:.3f})")
        ax.scatter(x, y, color=color, alpha=0.5, s=40, edgecolors="white")

    ax.set_xlabel(iv_label, fontsize=9)
    ax.set_ylabel("Post-Launch Sentiment (DV)", fontsize=9)
    ax.set_title(f"{hyp_label}: {iv_label}", fontsize=10)
    ax.legend(fontsize=8)
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

interaction_plot("sentiment_IV1_c",        "IV1 - Anticipatory Sentiment",        "H4a", axes[0])
interaction_plot("pre_expect_score_IV2_c", "IV2 - Feature Expectation Intensity", "H4b", axes[1])
interaction_plot("competitor_freq_IV3_c",  "IV3 - Competitor Mention Frequency",  "H4c", axes[2])

fig.suptitle("Interaction Plots: Product Type as Moderator", fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "interaction_plots.png", dpi=150, bbox_inches="tight")
plt.show()
log.info("Interaction plots saved")


# ============================================================
# CELL 21 — Robustness Checks
#
# RC1: Exclude events with < 300 pre-launch posts
# RC2: Star rating as alternative DV
# RC3: IV3b (in-brand frequency) as alternative IV3
# RC4: Drop brand fixed effects (simpler spec)
# ============================================================

robustness_results = {}

# RC1 — Volume sensitivity: exclude low-volume events
df_rc1 = df[df["pre_launch_volume"] >= 300].copy()
if len(df_rc1) >= 20:
    m_rc1 = smf.ols(formula=formula, data=df_rc1).fit()
    robustness_results["RC1_volume_filter"] = {
        "n": int(m_rc1.nobs),
        "r2": round(m_rc1.rsquared, 4),
        "r2_adj": round(m_rc1.rsquared_adj, 4),
        "b_IV1": round(m_rc1.params.get("sentiment_IV1_c", np.nan), 4),
        "b_IV2": round(m_rc1.params.get("pre_expect_score_IV2_c", np.nan), 4),
        "b_IV3": round(m_rc1.params.get("competitor_freq_IV3_c", np.nan), 4),
        "p_IV1": round(m_rc1.pvalues.get("sentiment_IV1_c", np.nan), 4),
        "p_IV2": round(m_rc1.pvalues.get("pre_expect_score_IV2_c", np.nan), 4),
        "p_IV3": round(m_rc1.pvalues.get("competitor_freq_IV3_c", np.nan), 4),
    }
    log.info("RC1 complete: n=%d", int(m_rc1.nobs))
else:
    log.warning("RC1 skipped: fewer than 20 rows after volume filter (n=%d)", len(df_rc1))
    robustness_results["RC1_volume_filter"] = {"note": "skipped — insufficient rows"}


# RC2 — Alternative DV: star rating instead of LLM sentiment
# Requires pulling mean star rating per event x feature from BQ gold_post_launch
# If not yet available, load from a local CSV: data/star_ratings.csv
star_path = DATA_DIR / "star_ratings.csv"
if star_path.exists():
    df_stars = pd.read_csv(star_path)
    df_rc2   = df.merge(df_stars[["event", "feature", "mean_star_rating"]],
                         on=["event", "feature"], how="left")
    formula_rc2 = formula.replace("post_sentiment_DV", "mean_star_rating")
    m_rc2 = smf.ols(formula=formula_rc2, data=df_rc2.dropna(subset=["mean_star_rating"])).fit()
    robustness_results["RC2_star_rating_DV"] = {
        "n": int(m_rc2.nobs),
        "r2": round(m_rc2.rsquared, 4),
        "r2_adj": round(m_rc2.rsquared_adj, 4),
        "b_IV1": round(m_rc2.params.get("sentiment_IV1_c", np.nan), 4),
        "b_IV2": round(m_rc2.params.get("pre_expect_score_IV2_c", np.nan), 4),
        "b_IV3": round(m_rc2.params.get("competitor_freq_IV3_c", np.nan), 4),
        "p_IV1": round(m_rc2.pvalues.get("sentiment_IV1_c", np.nan), 4),
        "p_IV2": round(m_rc2.pvalues.get("pre_expect_score_IV2_c", np.nan), 4),
        "p_IV3": round(m_rc2.pvalues.get("competitor_freq_IV3_c", np.nan), 4),
    }
    log.info("RC2 complete: star rating DV")
else:
    log.warning("RC2 skipped: %s not found. Generate it from gold_post_launch.", star_path)
    robustness_results["RC2_star_rating_DV"] = {"note": "skipped — star_ratings.csv not found"}


# RC3 — Alternative IV3: in-brand frequency (IV3b)
formula_rc3 = formula.replace("competitor_freq_IV3_c", "inbrand_freq_IV3b_c")
m_rc3 = smf.ols(formula=formula_rc3, data=df).fit()
robustness_results["RC3_IV3b_inbrand"] = {
    "n": int(m_rc3.nobs),
    "r2": round(m_rc3.rsquared, 4),
    "r2_adj": round(m_rc3.rsquared_adj, 4),
    "b_IV1": round(m_rc3.params.get("sentiment_IV1_c", np.nan), 4),
    "b_IV2": round(m_rc3.params.get("pre_expect_score_IV2_c", np.nan), 4),
    "b_IV3b": round(m_rc3.params.get("inbrand_freq_IV3b_c", np.nan), 4),
    "p_IV1": round(m_rc3.pvalues.get("sentiment_IV1_c", np.nan), 4),
    "p_IV2": round(m_rc3.pvalues.get("pre_expect_score_IV2_c", np.nan), 4),
    "p_IV3b": round(m_rc3.pvalues.get("inbrand_freq_IV3b_c", np.nan), 4),
}
log.info("RC3 complete: IV3b in-brand alternative")


# RC4 — Drop brand fixed effects (simpler specification)
formula_rc4 = formula.replace("C(brand) +", "")
m_rc4 = smf.ols(formula=formula_rc4, data=df).fit()
robustness_results["RC4_no_brand_FE"] = {
    "n": int(m_rc4.nobs),
    "r2": round(m_rc4.rsquared, 4),
    "r2_adj": round(m_rc4.rsquared_adj, 4),
    "b_IV1": round(m_rc4.params.get("sentiment_IV1_c", np.nan), 4),
    "b_IV2": round(m_rc4.params.get("pre_expect_score_IV2_c", np.nan), 4),
    "b_IV3": round(m_rc4.params.get("competitor_freq_IV3_c", np.nan), 4),
    "p_IV1": round(m_rc4.pvalues.get("sentiment_IV1_c", np.nan), 4),
    "p_IV2": round(m_rc4.pvalues.get("pre_expect_score_IV2_c", np.nan), 4),
    "p_IV3": round(m_rc4.pvalues.get("competitor_freq_IV3_c", np.nan), 4),
}
log.info("RC4 complete: no brand fixed effects")


# ── Summary table ────────────────────────────────────────────
rc_df = pd.DataFrame(robustness_results).T
rc_df.index.name = "specification"
print("\n── Robustness check summary ─────────────────────────────")
print(rc_df.to_string())
rc_df.to_csv(RESULTS_DIR / "robustness_summary.csv")
log.info("Robustness summary saved")


# ============================================================
# CELL 22 — Hypothesis Summary Table
# Quick-read table mapping each H to its coefficient,
# direction, significance, and supported/not supported.
# ============================================================

def get_term(params, pvalues, key):
    b = params.get(key, np.nan)
    p = pvalues.get(key, np.nan)
    sig = ("***" if p < .001 else "**" if p < .01
           else "*" if p < .05 else "n.s.")
    direction = "positive" if b > 0 else "negative" if b < 0 else "—"
    supported = (
        "Supported" if (p < .05)
        else "Not supported"
    )
    return round(b, 4), round(p, 4), sig, direction, supported

p = model_main.params
pv = model_main.pvalues

hyp_summary = pd.DataFrame([
    ["H1", "IV1 -> DV (overall sentiment)",
     *get_term(p, pv, "sentiment_IV1_c")],
    ["H2", "IV2 -> DV (feature expectation intensity)",
     *get_term(p, pv, "pre_expect_score_IV2_c")],
    ["H3", "IV3 -> DV (competitor frequency)",
     *get_term(p, pv, "competitor_freq_IV3_c")],
    ["H4a", "IV1 x ProductType interaction",
     *get_term(p, pv, "sentiment_IV1_c:product_type")],
    ["H4b", "IV2 x ProductType interaction",
     *get_term(p, pv, "pre_expect_score_IV2_c:product_type")],
    ["H4c", "IV3 x ProductType interaction",
     *get_term(p, pv, "competitor_freq_IV3_c:product_type")],
], columns=["Hypothesis", "Description", "b", "p", "sig",
            "Direction", "Verdict"])

print("\n── Hypothesis summary ───────────────────────────────────")
print(hyp_summary.to_string(index=False))
hyp_summary.to_csv(RESULTS_DIR / "hypothesis_summary.csv", index=False)
log.info("Hypothesis summary saved")


# ============================================================
# CELL 23 — Save Full Results Bundle
# ============================================================

log.info("Stage 7 complete. Output files:")
for f in sorted(RESULTS_DIR.iterdir()):
    log.info("  results/%s", f.name)
for f in sorted(FIGURES_DIR.iterdir()):
    log.info("  figures/%s", f.name)

print("\n== Stage 7 done ==")
print(f"Main model R2     : {model_main.rsquared:.4f}")
print(f"Adj R2            : {model_main.rsquared_adj:.4f}")
print(f"N                 : {int(model_main.nobs)}")
print(f"Robustness checks : {len(robustness_results)} completed")
