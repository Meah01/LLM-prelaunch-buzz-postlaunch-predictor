-- ============================================================
-- silver_reddit.sql
-- Stage 4a — Silver transform for pre-launch Reddit data
--
-- Source:      thesis_dataset.bronze_reddit
-- Destination: thesis_dataset.silver_reddit
-- ============================================================
--
-- PREREQUISITE: thesis_dataset.events must exist in BigQuery.
-- Load it once from events.csv before running this script:
--
--   bq load \
--     --source_format=CSV \
--     --autodetect \
--     --replace \
--     vuthesis-llm-buzz:thesis_dataset.events \
--     events.csv
--
-- ============================================================
-- TRANSFORMS (applied in order):
--
--   1. Deduplication   — keep latest version of each post_id
--   2. Date window     — keep only days_to_launch IN [-90, -1]
--   3. Short text      — remove posts with < 10 word tokens
--   4. days_to_launch  — recompute from created_utc + launch_date
--                        as cross-check against scraper value
-- ============================================================

CREATE OR REPLACE TABLE `vuthesis-llm-buzz.thesis_dataset.silver_reddit` AS

WITH

-- ── Step 1: Deduplication ──────────────────────────────────────────────────
--
-- The Reddit scraper streams daily JSONL batches to GCS, and WRITE_APPEND
-- in the BQ load step means re-runs accumulate duplicate rows. A post can
-- also appear across overlapping search windows for the same product_event.
--
-- Strategy: partition by post_id, keep the row with the latest created_utc.
-- This preserves the most up-to-date version of an edited post.

deduped AS (
  SELECT *
  FROM (
    SELECT
      *,
      ROW_NUMBER() OVER (
        PARTITION BY post_id
        ORDER BY created_utc DESC
      ) AS _rn
    FROM `vuthesis-llm-buzz.thesis_dataset.bronze_reddit`
    WHERE post_id IS NOT NULL  -- rows without an ID cannot be deduplicated safely
  )
  WHERE _rn = 1
),

-- ── Step 2: Join events reference table ────────────────────────────────────
--
-- Brings in confirmed launch_date, product_type (hedonic=1 / utilitarian=0),
-- and brand for each product_event. INNER JOIN intentionally drops any posts
-- whose product_event has no matching entry in events — those are orphaned
-- records from a scraper run against a product not in the analytical sample.

joined AS (
  SELECT
    r.post_id,
    r.product_event,
    r.subreddit,
    r.title,
    r.body_text,
    r.comments,
    r.upvotes,
    r.created_utc,
    r.days_to_launch,      -- trust Bronze value directly
    e.launch_date,
    e.product_type,
    e.brand
  FROM deduped r
  INNER JOIN `vuthesis-llm-buzz.thesis_dataset.events` e
    ON r.product_event = e.product_event
),

-- ── Step 3: Date window enforcement ────────────────────────────────────────
--
-- Retain only posts strictly within the 90-day pre-launch window.
-- Upper bound is -1, not 0: launch day itself is post-launch context
-- (product is publicly available), so it does not belong in the
-- pre-launch corpus used to construct the IVs.

windowed AS (
  SELECT *
  FROM joined
  WHERE (days_to_launch BETWEEN -90 AND -1)   -- original scraper (negative)
   OR (days_to_launch BETWEEN  1  AND 90)   -- newer scraper (positive)
),

-- ── Step 4: Short text filter ──────────────────────────────────────────────
--
-- Posts with fewer than 10 word tokens in body_text carry insufficient
-- textual content for reliable sentiment or feature extraction.
-- Typical examples: "[deleted]", "This.", link-only posts.
--
-- REGEXP_EXTRACT_ALL(text, r'\S+') counts whitespace-delimited tokens.
-- This is more robust than SPLIT(text, ' ') which is sensitive to
-- consecutive spaces and returns empty strings between them.
--
-- Posts where body_text is NULL (link posts, removed posts) are also
-- excluded here — they contribute no extractable signal.

filtered AS (
  SELECT *
  FROM windowed
  WHERE
    body_text IS NOT NULL
    AND ARRAY_LENGTH(REGEXP_EXTRACT_ALL(body_text, r'\S+')) >= 3
)

-- ── Final output: silver_reddit ────────────────────────────────────────────

SELECT
  post_id,
  product_event,
  brand,
  product_type,
  launch_date,
  subreddit,
  title,
  body_text,
  comments,         -- JSON string; unpacked in Stage 5
  upvotes,
  created_utc,
  days_to_launch
FROM filtered
ORDER BY product_event, days_to_launch DESC
