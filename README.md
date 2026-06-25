# LLM Pre-Launch Buzz → Post-Launch Predictor

**MSc Marketing Thesis · Digital Marketing & Analytics · VU Amsterdam · 2025–2026**

> *Do LLM-extracted pre-launch eWOM signals predict post-launch review sentiment — and does product type moderate this relationship?*

---

## Research Overview

This repository contains the full data pipeline, LLM extraction code, and modelling notebook for an MSc thesis investigating whether pre-launch consumer discourse on Reddit can predict post-launch feature-level sentiment in Amazon reviews, using large language models as the extraction instrument.

**Unit of analysis:** Event × Feature (long format) — 13 product launch events × 4 features = 52 observations

**Sample:** Consumer electronics launches (Apple, Samsung, Google, LG, Motorola), 2015–2022

### Variables

| Role | Variable | Description |
|------|----------|-------------|
| IV1 | Anticipatory Sentiment | Mean overall sentiment of pre-launch Reddit posts (event-level) |
| IV2 | Feature Expectation Intensity | Mention rate × conditional sentiment per feature (feature-level) |
| IV3 | Competitor Comparison Frequency | Proportion of pre-launch posts co-mentioning a rival brand (event-level) |
| DV | Post-Launch Feature Sentiment | Mean LLM-extracted feature sentiment from Amazon reviews (feature-level) |
| Moderator | Product Type | Binary: 1 = hedonic, 0 = utilitarian (event-level) |

**Model:** OLS regression with HC3 heteroscedasticity-consistent standard errors; moderated regression with mean-centered IVs.

---

## Repository Structure

```
LLM-prelaunch-buzz-postlaunch-predictor/
│
├── cloud/
│   └── gcp_config.py                    # GCP project ID, bucket name, dataset constants
│
├── sql/
│   ├── silver/
│   │   ├── silver_reddit.sql            # Dedup, date windowing, days_to_launch
│   │   └── silver_reviews.sql           # Dedup, date windowing, days_since_launch
│   └── gold/
│       ├── gold_pre_launch.sql          # Pre-launch analytics corpus (IV inputs)
│       └── gold_post_launch.sql         # Post-launch analytics corpus (DV input)
│
├── prompts/
│   ├── sentiment_extraction.txt         # LLM prompt — IV1 / DV overall sentiment
│   ├── feature_extraction.txt           # LLM prompt — IV2 / DV feature-level sentiment
│   └── competitor_detection.txt         # LLM prompt — IV3 competitor mention detection
│
├── Pre-launch Reddit Scrapper/
│   └── reddit_scraper.ipynb             # Collect Reddit posts → GCS Bronze
│
├── Post-launch Amazon reviews/
│   ├── amazon_data_loader.ipynb         # Filter UCSD Amazon dataset by ASIN → GCS Bronze
│   └── amazon_eda_clean.ipynb           # EDA of Amazon Silver table
│
├── Post-launch BestBuy Scrapper/
│   └── bestbuy_scraper.ipynb            # Scrape BestBuy reviews → GCS Bronze
│
├── orchestrate.py                       # ETL: GCS Bronze → BigQuery Silver → Gold
├── events.csv                           # Reference table: events, ASINs, launch dates, product type
├── pre_launch_buzz_eda.ipynb            # EDA of Reddit Silver table
├── post_launch_reviews_eda.ipynb        # EDA of Amazon Silver table
├── llm_extraction.ipynb                 # Stages 1–5: LLM extraction + IRR validation
└── modelling.ipynb                      # Stages 6–7: variable construction + OLS regression
```

---

## Pipeline Stages

```
events.csv
    │
    ├──► Reddit (Arctic Shift API) ──► GCS Bronze: bronze/reddit/
    └──► UCSD Amazon Reviews 2023  ──► GCS Bronze: bronze/amazon/
                   │
                   ▼
           orchestrate.py
           ├── Language filter (langdetect)
           ├── Schema validation
           ├── bq load → silver_reddit / silver_amazon
           ├── Silver SQL transforms (dedup, date window, days columns)
           └── Gold SQL aggregation
                   │
          ┌────────┴────────┐
          ▼                 ▼
  gold_pre_launch    gold_post_launch
          │                 │
          └────────┬────────┘
                   ▼
          llm_extraction.ipynb
          (Syn-Chain prompting via Ollama → LLaMA 3.1 8B)
                   │
                   ▼
          modelling.ipynb
          (Variable construction → OLS → results)
```

**Stage 0 — Event selection** (manual): Define product events in `events.csv`; confirm ≥500 pre-launch posts and ≥200 post-launch reviews per event.

**Stage 1 — Pre-launch collection**: Reddit posts from product-specific subreddits in the 90-day pre-launch window, collected via Arctic Shift API, streamed to GCS Bronze.

**Stage 2 — Post-launch collection**: Amazon reviews from UCSD McAuley Amazon Reviews 2023 dataset, filtered by ASIN and 90-day post-launch window.

**Stage 3 — ETL orchestration** (`orchestrate.py`): Language filtering → schema validation → BigQuery Silver load → Silver SQL transforms → Gold aggregation.

**Stage 4 — EDA**: Volume, sentiment distribution, and threshold checks per event.

**Stage 5 — LLM extraction** (`llm_extraction.ipynb`): Two-step Syn-Chain prompting (opinion extraction → sentiment scoring) using LLaMA 3.1 8B via Ollama. Inter-rater reliability validated against a stratified human-coded benchmark (Cohen's κ: .712–.917).

**Stage 6–7 — Modelling** (`modelling.ipynb`): Variable construction at event × feature level, mean-centering, OLS regression with HC3 standard errors across seven progressive specifications.

---

## Setup

### Prerequisites

- Python 3.10+ via **Anaconda** (avoid Windows Store Python — module installation fails)
- [Ollama](https://ollama.com/) installed locally with `llama3.1:8b` pulled
- GCP project with BigQuery and GCS enabled
- `gcloud` CLI authenticated via ADC: `gcloud auth application-default login`

### Installation

```bash
# Clone the repository
git clone https://github.com/<Meah01>/LLM-prelaunch-buzz-postlaunch-predictor.git
cd LLM-prelaunch-buzz-postlaunch-predictor

# Create and activate virtual environment (Anaconda)
conda create -n thesis python=3.10
conda activate thesis

# Install dependencies
pip install -r requirements.txt
```

### Cloud Configuration

Set your GCP credentials before running any cloud-connected notebook:

```bash
set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\<you>\.gcp\thesis-sa-key.json
```

GCS bucket structure:

```
gs://thesis-bucket/
├── config/events.csv
├── bronze/reddit/{product_event}/{date}.jsonl
├── bronze/amazon/{product_event}.csv
└── logs/orchestrate_{timestamp}.log
```

BigQuery dataset: `vuthesis-llm-buzz.thesis_dataset`

```
thesis_dataset/
├── silver_reddit
├── silver_amazon
├── gold_pre_launch
└── gold_post_launch
```

> **Note:** Add `*.json`, `.gcp/`, and `data/cache/` to `.gitignore`. Never commit credentials.

### LLM Inference

LLM extraction runs **entirely locally** via Ollama. No cloud compute or API key is required for inference.

```bash
# Pull the model before running extraction
ollama pull llama3.1:8b
```

Hardware used during development: Lenovo Legion Y540-17IRH, GTX 1660 Ti 6GB VRAM. GPU inference is enabled through layer offloading to the GPU via Ollama's default configuration.

---

## Key Files

| File | Description |
|------|-------------|
| `events.csv` | Defines the analytical sample: product names, brands, launch dates, ASINs, product type classification |
| `llm_extraction.ipynb` | Core extraction notebook (Stages 1–5): Syn-Chain prompting, IRR validation, score aggregation |
| `modelling.ipynb` | Core modelling notebook (Stages 6–7): dataset construction, OLS specifications, robustness checks |
| `thesis_extracted.csv` | 52-row extracted dataset (pre-aggregation) |
| `thesis_modelling.csv` | 52-row modelling-ready dataset with mean-centered IVs (`_c` suffix) |
| `orchestrate.py` | ETL glue layer: Bronze → Silver → Gold |
| `prompts/` | All LLM prompts stored as `.txt` for full reproducibility |

---

## Modelling Dataset Schema

The final modelling dataset (`thesis_modelling.csv`) is structured in long format, one row per product event × feature:

| Column | Type | Level | Description |
|--------|------|-------|-------------|
| `event` | str | Event | Product launch identifier |
| `brand` | str | Event | Brand name |
| `feature` | str | Feature | battery / display / performance / price |
| `product_type` | int | Event | 1 = hedonic, 0 = utilitarian |
| `sentiment_IV1` | float | Event | Mean anticipatory sentiment (−1 to 1) |
| `pre_expect_score_IV2` | float | Event × Feature | Feature expectation intensity composite |
| `competitor_freq_IV3` | float | Event | Cross-brand competitor mention frequency |
| `inbrand_freq_IV3b` | float | Event | In-brand self-reference frequency (robustness) |
| `post_sentiment_DV` | float | Event × Feature | Mean post-launch feature sentiment (−1 to 1) |
| `sentiment_IV1_c` | float | Event | IV1 mean-centered |
| `pre_expect_score_IV2_c` | float | Event × Feature | IV2 mean-centered |
| `competitor_freq_IV3_c` | float | Event | IV3 mean-centered |
| `pre_launch_volume` | int | Event | Total pre-launch Reddit posts (control) |
| `post_review_count` | int | Event | Total post-launch Amazon reviews (control) |

---

## Reproducibility Notes

- All random operations seeded at `42` (`random.seed(42)`, `np.random.seed(42)`)
- All file paths use `pathlib.Path` for cross-platform compatibility
- GCS paths remain as strings (`gs://...`) — do not wrap in `pathlib`
- BigQuery client requires explicit location: `bigquery.Client(project="vuthesis-llm-buzz", location="europe-west4")`
- EDA notebooks cache BigQuery pulls locally as Parquet (`data/cache/`). Set `REFRESH_CACHE = True` at the top of each notebook to re-pull from BigQuery
- LLM prompts in `/prompts/` are part of the extraction method and must be versioned alongside the code

---

## Citation

If you use this pipeline or build on this work, please cite the thesis:

> Constantinescu, A. (2026). *LLM Pre-Launch Buzz as a Post-Launch Predictor: Feature-Level eWOM Signals and the Moderating Role of Product Type*. MSc Thesis, VU Amsterdam.

---

## License

This repository is made available for academic reference. Code may be reused with attribution. The extracted datasets are derived from Reddit (via Arctic Shift API) and the UCSD McAuley Amazon Reviews 2023 dataset — please consult their respective terms before reuse.
