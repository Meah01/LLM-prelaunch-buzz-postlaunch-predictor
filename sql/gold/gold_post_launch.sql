-- ============================================================
-- gold_post_launch.sql
-- Stage 4a — Gold corpus for post-launch Amazon reviews
--
-- Source:      thesis_dataset.silver_amazon
-- Destination: thesis_dataset.gold_post_launch
-- ============================================================
--
-- PURPOSE:
--   Produce a clean, LLM-ready corpus for extraction of the
--   dependent variable:
--
--     DV  post-launch review sentiment
--         (overall + feature-level, extracted by ABSA in Stage 5)
--
-- KEY DECISIONS:
--
--   review_text → `text` field (no concatenation needed)
--     Amazon reviews are self-contained — there is no separate title
--     field in the UCSD 2023 dataset for the products in scope.
--     The text field maps 1:1 to review_text.
--
--   `rating` retained alongside `text`
--     Star rating (1–5) serves two purposes in Stage 7:
--       (1) Alternative DV in robustness checks — OLS re-run with
--           mean star rating instead of LLM-extracted sentiment score
--       (2) Validation signal — high correlation between LLM-extracted
--           sentiment and star rating validates the extraction quality
--
--   `verified_purchase` retained as control variable
--     Verified purchase status is associated with more credible and
--     less extreme reviews (Hu et al., 2009). Included as a control
--     in the Stage 7 OLS regression.
--
-- ============================================================

CREATE OR REPLACE TABLE `vuthesis-llm-buzz.thesis_dataset.gold_post_launch` AS

SELECT

  -- ── Identifiers ───────────────────────────────────────────────────────────
  review_id,
  asin,
  product_name,
  product_event,
  brand,

  -- Moderator variable (1 = hedonic, 0 = utilitarian)
  product_type,
  launch_date,

  -- ── LLM input field ───────────────────────────────────────────────────────
  -- TRIM removes leading/trailing whitespace introduced by the UCSD dataset
  -- export. No other transformation applied here — text cleaning
  -- (URL stripping, HTML tag removal) is handled in Stage 5 before
  -- the LLM call, not in SQL, to preserve the original text in Gold.
  TRIM(review_text) AS text,

  -- ── Numeric ground truth ──────────────────────────────────────────────────
  -- Star rating 1–5. Used as alternative DV and extraction quality check.
  -- NULL ratings are retained here — Stage 5 handles them.
  rating,

  -- ── Control variables for Stage 7 OLS ────────────────────────────────────
  verified_purchase,
  days_since_launch,  -- positive integer, e.g. 14 = 14 days after launch
  review_date

FROM `vuthesis-llm-buzz.thesis_dataset.silver_amazon`
ORDER BY product_event, days_since_launch
