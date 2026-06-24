"""
Stage 5 — Inter-Rater Reliability Validation
=============================================
Draws a stratified random sample (n=300) from the Gold tables,
runs LLM extraction on each document, and computes Cohen's Kappa
between LLM labels and your manual annotations.

Workflow
--------
1. Run this script once to generate the sample CSV:
       python validation.py --mode sample

2. Open data/validation/validation_sample.csv in Excel or VSCode.
   Fill in the four manual annotation columns for all 300 rows:
       manual_overall_sentiment    (-1 / 0 / 1)
       manual_feat_mentioned       (1 / 0)
       manual_feat_sentiment       (-1 / 0 / 1)
       manual_has_competitor       (1 / 0)

3. Save the file, then run:
       python validation.py --mode evaluate

   Cohen's Kappa is reported per task. Target: κ ≥ 0.70 on all tasks.

Notes
-----
- Continuous LLM scores are discretised to (-1 / 0 / 1) for Kappa
  using thresholds: score < -0.2 -> -1, score > 0.2 -> 1, else 0.
- Stratified sampling: 150 pre-launch posts, 150 post-launch reviews,
  proportional across events.
- RANDOM_SEED = 42 ensures reproducible sampling.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from brand_taxonomy import KNOWN_BRANDS, FOCAL_BRAND_KEYS

import numpy as np
import pandas as pd
import ollama
from google.cloud import bigquery
from sklearn.metrics import cohen_kappa_score

# ── Config (must match llm_extraction.ipynb) ──────────────
PROJECT_ID      = "vuthesis-llm-buzz"
DATASET_ID      = "thesis_dataset"
EVENTS_CSV      = Path("events.csv")
VALIDATION_DIR  = Path("data/validation")
CACHE_DIR       = Path("data/cache")
LOG_DIR         = Path("logs")
MODEL           = "llama3.1"
RANDOM_SEED     = 42
SAMPLE_N        = 300          # 150 pre-launch + 150 post-launch
TEXT_CHAR_LIMIT = 1200
SENTIMENT_THRESHOLD = 0.2      # |score| ≤ threshold -> neutral (0)

# Unified feature set — same four features across all product types
# Required for OLS: hedonic and utilitarian rows must be comparable
FEATURES = ["battery", "price", "performance", "display"]
FEATURE_SETS = {
    "hedonic":     FEATURES,
    "utilitarian": FEATURES,
}

VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "stage5_validation.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Helpers ───────────────────────────────────────────────

def discretise(score: float, threshold: float = SENTIMENT_THRESHOLD) -> int:
    """Map continuous [-1, 1] score to ternary label: -1, 0, or 1."""
    if score > threshold:
        return 1
    elif score < -threshold:
        return -1
    return 0


SYSTEM_PROMPT = (
    "You are a sentiment analysis assistant. Extract structured sentiment "
    "information from consumer text. Respond with valid JSON only — "
    "no explanation, no preamble, no markdown fences."
)


def classify_mentions_val(brand_mentions: list, focal_brand: str) -> tuple:
    """
    Rule-based brand taxonomy classifier for validation script.
    Mirrors the logic in Cell 7 of llm_extraction.ipynb.
    """
    if not focal_brand:
        return 0, 0   # Cannot classify without a focal brand — skip safely

    focal_keys = FOCAL_BRAND_KEYS.get(focal_brand, set())
    # Resolve focal brand to its canonical form too
    focal_canonical = KNOWN_BRANDS.get(focal_brand, focal_brand)

    cross_brand = 0
    in_brand    = 0
    for mention in brand_mentions:
        text = str(mention).lower().strip()
        if not text:
            continue
        # Resolve to canonical brand — longest key match first
        canonical = None
        for key in sorted(KNOWN_BRANDS.keys(), key=len, reverse=True):
            if key in text:
                canonical = KNOWN_BRANDS[key]
                break
        if canonical is None:
            continue
        # In-brand: same canonical brand OR text contains a focal-brand keyword
        # Use non-empty focal_keys only to avoid "" matching everything
        is_inbrand = (canonical == focal_canonical)
        if focal_keys:
            is_inbrand = is_inbrand or any(k and k in text for k in focal_keys)
        if is_inbrand:
            in_brand = 1
        else:
            cross_brand = 1
    return cross_brand, in_brand


def build_prompt(text: str, product_name: str, product_type: str,
                 context_type: str = "review") -> str:
    features = FEATURE_SETS[product_type]
    feature_lines = "\n".join(
        f'        "{f}": {{"mentioned": <bool>, "sentiment": <float -1.0 to 1.0>}}'
        for f in features
    )
    text_clipped = text[:TEXT_CHAR_LIMIT].replace('"', "'")
    return (
        f"Analyse this consumer {context_type} about the {product_name}.\n\n"
        f"Text:\n---\n{text_clipped}\n---\n\n"
        "Task: Extract the following and return JSON only.\n\n"
        "1. overall_sentiment  Float -1.0 to 1.0.\n"
        "2. features  For each attribute: mentioned (bool) + sentiment (-1.0 to 1.0).\n"
        "3. brand_mentions\n"
        "   List ALL smartphone, tablet, or device brand or product names mentioned\n"
        "   anywhere in the text — regardless of context or intent.\n"
        "   Include brand names, product lines, and specific model names.\n"
        "   Exclude: accessory brands (OtterBox), retailers (Amazon), services\n"
        "   (Spotify, Google Maps), and chipset names (Snapdragon, Exynos).\n"
        "   Return [] if no device brand or product names appear.\n\n"
        "Required JSON:\n"
        "{\n"
        '  "overall_sentiment": <float>,\n'
        '  "features": {\n'
        f"{feature_lines}\n"
        "  },\n"
        '  "brand_mentions": ["<brand_name_1>", "<brand_name_2>"]\n'
        "}"
    )


def extract_single(text, product_name, product_type, context_type, max_retries=3):
    prompt = build_prompt(text, product_name, product_type, context_type)
    for attempt in range(max_retries):
        try:
            resp = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                options={
                    "temperature": 0.0,
                    "num_predict": 200,   # JSON response never exceeds ~100 tokens
                    "num_ctx":     1024,  # Prompt + response fits well within 1024
                },
                format="json",
            )
            raw = resp["message"]["content"].strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw.strip())
            assert "overall_sentiment" in result
            assert "features"           in result
            assert "brand_mentions"     in result
            # Replace null values with safe defaults before clipping
            if result["overall_sentiment"] is None:
                result["overall_sentiment"] = 0.0
            result["overall_sentiment"] = float(np.clip(result["overall_sentiment"], -1, 1))
            for feat in result["features"].values():
                if feat.get("mentioned") is None:
                    feat["mentioned"] = False
                # Failsafe: if feature not mentioned, force sentiment to 0
                if not feat["mentioned"]:
                    feat["sentiment"] = 0.0
                else:
                    if feat.get("sentiment") is None:
                        feat["sentiment"] = 0.0
                    feat["sentiment"] = float(np.clip(feat["sentiment"], -1, 1))
            # Normalise brand_mentions to list of lowercase strings
            result["brand_mentions"] = [
                str(b).lower().strip()
                for b in result.get("brand_mentions", [])
                if b
            ]
            return result
        except Exception as e:
            wait = 2 ** attempt
            log.warning("Attempt %d/%d failed: %s — retry in %ds", attempt + 1, max_retries, e, wait)
            import time; time.sleep(wait)
    return None


# ── Mode: sample ──────────────────────────────────────────

def run_sample():
    log.info("Drawing stratified validation sample (n=%d) ...", SAMPLE_N)

    events_df = pd.read_csv(EVENTS_CSV)
    # Support both integer (1/0) and string ("hedonic"/"utilitarian") encodings
    if events_df["product_type"].dtype != object:
        events_df["product_type"] = events_df["product_type"].map({1: "hedonic", 0: "utilitarian"})
    else:
        events_df["product_type"] = events_df["product_type"].str.lower().str.strip()
    type_lookup = events_df.set_index("product_name")["product_type"].to_dict()

    bq = bigquery.Client(project=PROJECT_ID)

    def load(table):
        cache = CACHE_DIR / f"{table}.parquet"
        if cache.exists():
            return pd.read_parquet(cache)
        df = bq.query(f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{table}`").to_dataframe()
        df.to_parquet(cache, index=False)
        return df

    df_pre  = load("gold_pre_launch")
    df_post = load("gold_post_launch")

    # Print actual column names so we can align the script
    print("gold_pre_launch columns :", df_pre.columns.tolist())
    print("gold_post_launch columns:", df_post.columns.tolist())

    # Stratified sample: proportional per event, 150 pre + 150 post
    half = SAMPLE_N // 2

    def stratified_sample(df, id_col, text_col, n):
        groups = df.groupby("product_event")
        per_group = max(1, n // len(groups))
        sampled = (
            pd.concat([
                g.sample(min(len(g), per_group), random_state=RANDOM_SEED)
                for _, g in groups
            ])
            .reset_index(drop=True)
        )
        if len(sampled) > n:
            sampled = sampled.sample(n, random_state=RANDOM_SEED).reset_index(drop=True)
        return sampled[[id_col, "product_event", "brand", text_col]].rename(
            columns={id_col: "doc_id", text_col: "text"}
        )

    pre_sample  = stratified_sample(df_pre,  "post_id",   "text",   half)
    post_sample = stratified_sample(df_post, "review_id", "text", half)

    pre_sample["corpus"]  = "pre_launch"
    post_sample["corpus"] = "post_launch"
    sample = pd.concat([pre_sample, post_sample], ignore_index=True)

    # Run LLM extraction on each sampled document
    log.info("Running LLM extraction on %d documents ...", len(sample))

    llm_overall, llm_mentioned, llm_feat_sent, llm_competitor = [], [], [], []
    first_feature = []

    for _, row in sample.iterrows():
        ptype  = type_lookup.get(row["product_event"], "hedonic")
        ctx    = "pre-launch discussion post" if row["corpus"] == "pre_launch" else "post-launch customer review"
        result = extract_single(str(row["text"]), row["product_event"], ptype, ctx)

        if result is None:
            llm_overall.append(None)
            llm_mentioned.append(None)
            llm_feat_sent.append(None)
            llm_competitor.append(None)
            first_feature.append(None)
            continue

        llm_overall.append(discretise(result["overall_sentiment"]))
        # Rule-based classification using brand taxonomy
        brand_mentions = result.get("brand_mentions", [])
        # Use brand column from Gold table row directly — avoids events_df lookup mismatch
        focal_brand_str = str(row.get("brand", "")).lower().strip()
        cross_brand, _ = classify_mentions_val(brand_mentions, focal_brand_str)
        llm_competitor.append(cross_brand)

        # Rotate across all four features so IRR covers battery/price/performance/display
        feats = FEATURE_SETS[ptype]
        feat0 = feats[len(first_feature) % len(feats)]
        first_feature.append(feat0)
        feat_data = result["features"].get(feat0, {})
        llm_mentioned.append(int(bool(feat_data.get("mentioned", False))))
        llm_feat_sent.append(discretise(feat_data.get("sentiment", 0.0)))

    sample["llm_overall_sentiment"] = llm_overall
    sample["llm_feat_name"]         = first_feature
    sample["llm_feat_mentioned"]    = llm_mentioned
    sample["llm_feat_sentiment"]    = llm_feat_sent
    sample["llm_has_competitor"]    = llm_competitor

    # Add empty manual annotation columns
    sample["manual_overall_sentiment"] = ""
    sample["manual_feat_mentioned"]    = ""
    sample["manual_feat_sentiment"]    = ""
    sample["manual_has_competitor"]    = ""

    out = VALIDATION_DIR / "validation_sample.csv"
    sample.to_csv(out, index=False)
    log.info("Validation sample saved -> %s", out)
    print(f"\nNext step: open {out} and fill in the four 'manual_*' columns for all {len(sample)} rows.")
    print("Column guide:")
    print("  manual_overall_sentiment : -1 (negative) / 0 (neutral) / 1 (positive)")
    print("  manual_feat_mentioned    :  1 (yes) / 0 (no)")
    print("  manual_feat_sentiment    : -1 / 0 / 1")
    print("  manual_has_competitor    :  1 (yes) / 0 (no)")
    print("\nThen run:  python validation.py --mode evaluate")


# ── Mode: evaluate ────────────────────────────────────────

def run_evaluate():
    sample_path = VALIDATION_DIR / "validation_sample.csv"
    if not sample_path.exists():
        sys.exit(f"ERROR: {sample_path} not found. Run --mode sample first.")

    df = pd.read_csv(sample_path)

    required = [
        "manual_overall_sentiment", "manual_feat_mentioned",
        "manual_feat_sentiment", "manual_has_competitor",
    ]
    for col in required:
        if df[col].isnull().all() or (df[col] == "").all():
            sys.exit(f"ERROR: Column '{col}' is empty — fill in manual annotations first.")

    # Drop rows where either label is missing (annotator skipped)
    df = df.dropna(subset=required + [
        "llm_overall_sentiment", "llm_feat_mentioned",
        "llm_feat_sentiment", "llm_has_competitor",
    ])

    df[required] = df[required].astype(int)

    tasks = {
        "Overall sentiment (IV1 / DV)": (
            "llm_overall_sentiment", "manual_overall_sentiment"
        ),
        "Feature mentioned (IV2 / DV)": (
            "llm_feat_mentioned", "manual_feat_mentioned"
        ),
        "Feature sentiment (IV2 / DV)": (
            "llm_feat_sentiment", "manual_feat_sentiment"
        ),
        "Competitor mention (IV3)": (
            "llm_has_competitor", "manual_has_competitor"
        ),
    }

    print("\n" + "=" * 55)
    print("  Inter-Rater Reliability — Cohen's Kappa")
    print("  LLM (Llama 3.1 8B via Ollama) vs. Manual")
    print("=" * 55)

    all_pass = True
    for task_name, (llm_col, man_col) in tasks.items():
        llm_vals = df[llm_col].tolist()
        man_vals = df[man_col].tolist()
        kappa    = cohen_kappa_score(man_vals, llm_vals)
        status   = "PASS ✓" if kappa >= 0.70 else "BELOW THRESHOLD ✗"
        if kappa < 0.70:
            all_pass = False
        print(f"  {task_name:<35} κ = {kappa:.3f}   {status}")

    print("=" * 55)
    print(f"  Documents evaluated: {len(df)}")
    print(f"  Threshold: κ ≥ 0.70")

    if all_pass:
        print("\n  All tasks pass. Safe to proceed with full extraction (Cells 5–6).")
    else:
        print("\n  One or more tasks below threshold.")
        print("  Review the prompt in Cell 4 of llm_extraction.ipynb,")
        print("  adjust few-shot examples or instruction phrasing, then re-run.")

    # Save results table
    results_rows = []
    for task_name, (llm_col, man_col) in tasks.items():
        kappa = cohen_kappa_score(df[man_col].tolist(), df[llm_col].tolist())
        results_rows.append({"task": task_name, "kappa": round(kappa, 4),
                              "n": len(df), "threshold": 0.70,
                              "pass": kappa >= 0.70})
    results_df = pd.DataFrame(results_rows)
    results_path = VALIDATION_DIR / "kappa_results.csv"
    results_df.to_csv(results_path, index=False)
    log.info("Kappa results saved -> %s", results_path)


# ── Entry point ───────────────────────────────────────────



# ── Lightweight brand-only extraction for revalidation ───────────────────────

BRAND_ONLY_SYSTEM = (
    "You are a brand name extractor. Extract device brand and product names "
    "from text. Respond with valid JSON only — no explanation, no markdown."
)

def extract_brands_only(text: str, max_retries: int = 3) -> list:
    """
    Lightweight single-purpose extraction: returns only brand_mentions list.
    Much faster than extract_single — skips sentiment and feature extraction.
    """
    text_clipped = text[:TEXT_CHAR_LIMIT].replace('"', "'")
    prompt = (
        f'Text: "{text_clipped}"'
        "List all smartphone, tablet, or device brand names and product model "
        "names mentioned in the text."
        "Exclude: accessories, retailers, services, chipset names."
        'Return JSON: {"brand_mentions": ["<name1>", "<name2>"]}'
    )
    for attempt in range(max_retries):
        try:
            resp = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": BRAND_ONLY_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                options={
                    "temperature": 0.0,
                    "num_predict": 200,   # JSON response never exceeds ~100 tokens
                    "num_ctx":     1024,  # Prompt + response fits well within 1024
                },
                format="json",
            )
            raw = resp["message"]["content"].strip()
            result = json.loads(raw)
            brands = result.get("brand_mentions", [])
            if not isinstance(brands, list):
                return []
            return [str(b).lower().strip() for b in brands if b]
        except Exception as e:
            import time
            wait = 2 ** attempt
            log.warning("Brand extraction attempt %d/%d failed: %s", attempt + 1, max_retries, e)
            time.sleep(wait)
    return []


def run_revalidate_competitor():
    """
    Re-run competitor extraction only on the existing validation sample.
    Overwrites only llm_has_competitor. Manual annotations preserved.

    Patches the brand column from the Gold table cache if missing,
    so the existing sample CSV can be reused without re-drawing.
    """
    sample_path = VALIDATION_DIR / "validation_sample.csv"
    if not sample_path.exists():
        sys.exit(f"ERROR: {sample_path} not found. Run --mode sample first.")

    df = pd.read_csv(sample_path)

    # ── Patch brand column if missing from old sample CSV ────────────────────
    # The brand column was added to sampling after some CSVs were already drawn.
    # Recover brand from cached Gold tables to avoid re-drawing the sample
    # and losing existing manual annotations.
    if "brand" not in df.columns or df["brand"].isna().all() or (df["brand"] == "").all():
        log.info("Brand column missing from sample — patching from Gold table cache ...")
        brand_patched = False
        for cache_name in ["gold_pre_launch", "gold_post_launch"]:
            cache_path = CACHE_DIR / f"{cache_name}.parquet"
            if cache_path.exists():
                # NEW
                import pyarrow.parquet as pq
                table = pq.read_table(str(cache_path))
                gold  = table.to_pandas(timestamp_as_object=True, date_as_object=True)
                id_col = "post_id" if "post_id" in gold.columns else "review_id"
                if "brand" in gold.columns and id_col in gold.columns:
                    brand_map = gold.set_index(id_col)["brand"].to_dict()
                    df["brand"] = df["doc_id"].map(brand_map).fillna(df.get("brand", ""))
                    brand_patched = True
                    log.info("Patched brand from %s (%d matches)",
                             cache_name, df["brand"].notna().sum())
        # Fallback: derive brand from events.csv via product_event
        if not brand_patched or df["brand"].isna().any():
            log.info("Falling back to events.csv brand lookup ...")
            evt = pd.read_csv(EVENTS_CSV)
            # Try both common column names for the event identifier
            evt_id_col = "product_name" if "product_name" in evt.columns else "product_event"
            if "brand" in evt.columns:
                evt_map = evt.set_index(evt_id_col)["brand"].to_dict()
                mask = df["brand"].isna() | (df["brand"] == "")
                df.loc[mask, "brand"] = df.loc[mask, "product_event"].map(evt_map)

    # Save patched brand column back so future runs don't need to re-patch
    df.to_csv(sample_path, index=False)
    log.info("Brand column ready. Unique brands: %s",
             df["brand"].dropna().unique().tolist())

    log.info("Re-running competitor extraction on %d documents ...", len(df))

    events_df = pd.read_csv(EVENTS_CSV)
    if events_df["product_type"].dtype != object:
        events_df["product_type"] = events_df["product_type"].map({1: "hedonic", 0: "utilitarian"})
    else:
        events_df["product_type"] = events_df["product_type"].str.lower().str.strip()
    type_lookup = events_df.set_index(
        "product_name" if "product_name" in events_df.columns else "product_event"
    )["product_type"].to_dict()

    updated = 0
    for idx, row in df.iterrows():
        ptype  = type_lookup.get(row["product_event"], "hedonic")
        ctx    = ("pre-launch discussion post" if row["corpus"] == "pre_launch"
                  else "post-launch customer review")
        result = extract_single(str(row["text"]), row["product_event"], ptype, ctx)

        if result is None:
            log.warning("Extraction failed for doc_id=%s — keeping existing value", row["doc_id"])
            continue

        focal_b = str(row.get("brand", "")).lower().strip()
        bm      = result.get("brand_mentions", [])
        cross, _ = classify_mentions_val(bm, focal_b)
        df.at[idx, "llm_has_competitor"] = cross
        updated += 1

        if updated % 50 == 0:
            log.info("Progress: %d/%d", updated, len(df))

    df.to_csv(sample_path, index=False)
    log.info("Revalidation complete. Updated %d rows.", updated)
    print(f"Done. Updated llm_has_competitor for {updated} rows.")
    print("Now run:  python validation.py --mode evaluate")


def run_rerun():
    """
    Re-run LLM extraction only on rows where llm_overall_sentiment is blank.
    Overwrites those rows in-place in the existing validation_sample.csv.
    Uses format='json' (already applied to extract_single).
    """
    sample_path = VALIDATION_DIR / "validation_sample.csv"
    if not sample_path.exists():
        sys.exit(f"ERROR: {sample_path} not found. Run --mode sample first.")

    df = pd.read_csv(sample_path)
    failed_mask = df["llm_overall_sentiment"].isna() | (df["llm_overall_sentiment"] == "")
    failed = df[failed_mask].copy()
    log.info("Re-running extraction on %d failed rows ...", len(failed))

    events_df = pd.read_csv(EVENTS_CSV)
    if events_df["product_type"].dtype != object:
        events_df["product_type"] = events_df["product_type"].map({1: "hedonic", 0: "utilitarian"})
    else:
        events_df["product_type"] = events_df["product_type"].str.lower().str.strip()
    type_lookup = events_df.set_index("product_name")["product_type"].to_dict()

    for idx, row in failed.iterrows():
        ptype  = type_lookup.get(row["product_event"], "hedonic")
        ctx    = "pre-launch discussion post" if row["corpus"] == "pre_launch" else "post-launch customer review"
        result = extract_single(str(row["text"]), row["product_event"], ptype, ctx)

        if result is None:
            log.warning("Still failed after rerun: doc_id=%s", row["doc_id"])
            continue

        df.at[idx, "llm_overall_sentiment"] = discretise(result["overall_sentiment"])
        focal_b = str(row.get("brand", "")).lower().strip()
        bm = result.get("brand_mentions", [])
        cross, _ = classify_mentions_val(bm, focal_b)
        df.at[idx, "llm_has_competitor"] = cross

        feats    = FEATURE_SETS[ptype]
        feat0    = feats[0]
        feat_data = result["features"].get(feat0, {})
        df.at[idx, "llm_feat_name"]      = feat0
        df.at[idx, "llm_feat_mentioned"] = int(bool(feat_data.get("mentioned", False)))
        df.at[idx, "llm_feat_sentiment"] = discretise(feat_data.get("sentiment", 0.0))

    df.to_csv(sample_path, index=False)
    still_blank = df["llm_overall_sentiment"].isna().sum()
    log.info("Rerun complete. Remaining blanks: %d", still_blank)
    print(f"Saved updated file. Remaining blank rows: {still_blank}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5 inter-rater reliability validation")
    parser.add_argument(
        "--mode",
        choices=["sample", "rerun", "revalidate", "evaluate"],
        required=True,
        help=(
            "'sample': draw and extract; 'rerun': fix blank rows; "
            "'revalidate': re-run competitor extraction with updated prompt; "
            "'evaluate': compute Kappa"
        ),
    )
    args = parser.parse_args()

    if args.mode == "sample":
        run_sample()
    elif args.mode == "rerun":
        run_rerun()
    elif args.mode == "revalidate":
        run_revalidate_competitor()
    else:
        run_evaluate()
