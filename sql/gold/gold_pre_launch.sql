-- ============================================================
-- gold_pre_launch.sql
-- Stage 4a — Gold corpus for pre-launch Reddit data
--
-- Source:      thesis_dataset.silver_reddit
-- Destination: thesis_dataset.gold_pre_launch
-- ============================================================
--
-- PURPOSE:
--   Produce a clean, LLM-ready corpus for extraction of the
--   three independent variables:
--
--     IV1  anticipatory sentiment         (overall post sentiment)
--     IV2  feature expectation intensity  (feature-level mentions)
--     IV3  competitor comparison freq.    (competitor co-mentions)
--
-- KEY DECISIONS:
--
--   title + body_text → single `text` field
--     The LLM receives one input string per post. Title is prepended
--     because Reddit titles carry the most concentrated signal —
--     the key opinion or claim that the body elaborates.
--     Separator ' ||| ' is used instead of a newline character to
--     avoid JSONL line-break issues when the Gold table is exported
--     to GCS for Stage 5 batch processing.
--
--   `comments` kept as a separate JSON string column
--     Top-level comments are not folded into `text` here.
--     Stage 5 unpacks the JSON array and processes comments
--     independently (each comment as a separate extraction call),
--     allowing comment-level sentiment to be aggregated separately
--     from post-level sentiment.
--
--   `upvotes` retained and floored at 0
--     Upvote count is available as an optional weighting signal:
--     high-upvote posts represent community-endorsed sentiment
--     and can be used to produce engagement-weighted IV aggregates
--     in Stage 6 as a robustness check.
--
-- ============================================================

CREATE OR REPLACE TABLE `vuthesis-llm-buzz.thesis_dataset.gold_pre_launch` AS

SELECT

  -- ── Identifiers ───────────────────────────────────────────────────────────
  post_id,
  product_event,
  brand,

  -- Moderator variable (1 = hedonic, 0 = utilitarian)
  -- Carried through to Gold so Stage 5 output can be labelled
  -- without a separate join during modelling.
  product_type,
  launch_date,
  subreddit,

  -- ── LLM input field ───────────────────────────────────────────────────────
  -- Concatenate title and body_text into a single text string.
  -- Three cases:
  --   (a) both present  → "Title ||| Body"
  --   (b) title only    → "Title"           (link posts with short body removed in Silver)
  --   (c) body only     → "Body"            (edge case: missing title)
  CASE
    WHEN title     IS NOT NULL AND body_text IS NOT NULL
      THEN CONCAT(TRIM(title), ' ||| ', TRIM(body_text))
    WHEN title     IS NOT NULL AND body_text IS NULL
      THEN TRIM(title)
    ELSE TRIM(body_text)
  END AS text,

  -- ── Comments field ────────────────────────────────────────────────────────
  -- Raw JSON string from the scraper. Format: ["comment1 text", "comment2 text", ...]
  -- NULL where no comments were collected. Stage 5 handles NULL gracefully.
  comments,

  -- ── Engagement ────────────────────────────────────────────────────────────
  -- Floor at 0: Reddit can show negative net vote counts for heavily downvoted
  -- posts, but negative upvotes have no meaningful weighting interpretation.
  GREATEST(COALESCE(upvotes, 0), 0) AS upvotes,

  -- ── Time context ──────────────────────────────────────────────────────────
  created_utc,
  days_to_launch  -- negative integer, e.g. -45 = 45 days before launch

FROM `vuthesis-llm-buzz.thesis_dataset.silver_reddit`
ORDER BY product_event, days_to_launch DESC
