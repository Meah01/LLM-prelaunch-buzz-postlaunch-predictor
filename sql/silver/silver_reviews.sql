-- ============================================================
-- silver_reviews.sql
-- Stage 4a — Silver transform for post-launch Amazon reviews
--
-- Source:      thesis_dataset.bronze_amazon
-- Destination: thesis_dataset.silver_amazon
-- ============================================================
--
-- PREREQUISITE: thesis_dataset.events must exist in BigQuery.
-- See silver_reddit.sql header for the one-time load command.
--
-- ============================================================
-- TRANSFORMS (applied in order):
--
--   1. Deduplication     — keep latest version of each review_id
--   2. Date window       — keep only days_since_launch IN [1, 90]
--   3. Short text        — remove reviews with < 10 word tokens
--   4. days_since_launch — recompute from review_date + launch_date
-- ============================================================

CREATE OR REPLACE TABLE `vuthesis-llm-buzz.thesis_dataset.silver_amazon` AS

WITH

-- ── Step 1: Deduplication ──────────────────────────────────────────────────
--
-- Amazon review IDs from the UCSD dataset are unique within the raw file,
-- but WRITE_APPEND in the BQ load step means re-running the orchestrator
-- accumulates duplicate rows. Partition by review_id, keep most recent
-- review_date (handles the rare case of a review being updated).

deduped AS (
  SELECT *
  FROM (
    SELECT
      *,
      ROW_NUMBER() OVER (
        PARTITION BY review_id
        ORDER BY review_date DESC
      ) AS _rn
    FROM `vuthesis-llm-buzz.thesis_dataset.bronze_amazon`
    WHERE review_id IS NOT NULL
  )
  WHERE _rn = 1
),

-- ── Step 2: Join events reference table ────────────────────────────────────
--
-- Same logic as silver_reddit.sql. INNER JOIN drops reviews for any
-- product_event not in the analytical sample (should not occur in practice,
-- but acts as a safety net against ASIN cross-contamination).

joined AS (
  SELECT
    r.*,
    e.launch_date,
    e.product_type,
    e.brand
  FROM deduped r
  INNER JOIN `vuthesis-llm-buzz.thesis_dataset.events` e
    ON r.product_event = e.product_event
),

-- ── Step 3: Date window enforcement ────────────────────────────────────────
--
-- Retain only reviews within the 60-day post-launch window.
-- Lower bound is 1, not 0: day 0 reviews were written before the product
-- was physically available in most cases (early access, pre-order shipments)
-- and represent a systematically different reviewer population.
-- Upper bound is 60: beyond this window, review sentiment reflects the
-- installed base rather than launch-period reaction — outside the
-- theoretical scope of the DV construct.

windowed AS (
  SELECT *
  FROM joined
  WHERE days_since_launch BETWEEN 1 AND 90
),

-- ── Step 4: Short text filter ──────────────────────────────────────────────
--
-- Reviews with fewer than 10 word tokens are star-only ratings with no
-- substantive text ("Love it", "Doesn't work", "Great product!!!!").
-- These cannot contribute meaningful feature-level sentiment signal for
-- the ABSA extraction in Stage 5.
--
-- Note: the 10-word threshold is applied to review_text only, not to the
-- review title. Amazon review titles are not collected in this dataset.

filtered AS (
  SELECT *
  FROM windowed
  WHERE
    review_text IS NOT NULL
    AND ARRAY_LENGTH(REGEXP_EXTRACT_ALL(review_text, r'\S+')) >= 3
)

-- ── Final output: silver_amazon ────────────────────────────────────────────

SELECT
  review_id,
  asin,
  product_name,
  product_event,
  brand,
  product_type,
  launch_date,
  rating,            -- 1–5 stars; alternative DV in robustness checks (Stage 7)
  review_text,
  verified_purchase, -- control variable in Stage 7 OLS
  review_date,
  days_since_launch
FROM filtered
ORDER BY product_event, days_since_launch
