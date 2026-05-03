# LLM-prelaunch-buzz-postlaunch-predictor
(in development)

## Core idea
Before a launch, consumers talk online about the upcoming product on Reddit and online forums. They're trying to predict features, compare them to competititors, which builds up expectations. All of that is a signal.

My thesis asks: can we capture that pre-launch noise, run it through an LLM and actually predict how a product will land competitively once it hits the market?

## Mechanism (tech-wise)
The system collects pre-launch Reddit discussions and post-launch reviews from major electronics launches (f.e. iPhone, Samsung smartphones), extracting structured signals using LLM, and tests whether what people said before launch predicts the competitive reception after it, and if the gap between what the consumers expected vs what they got changes the competitive landscape.

### Stack
- Data collection: scrapping, APIs
- Data pre-processing & storage: Pandas, NumPy, DuckDB (SQL)
- Data processing: Syn-Chain prompting, ETL pipelines, Ollama 

### Data
The system uses scrapped Reddit discussions and Amazon reviews (meta + reviews) of phones and electronics (made available by [McAuley lab, 2023](https://amazon-reviews-2023.github.io)). 

### LLM & prompting
- The system uses an open-source LLM (still to be decided) used via Ollama to process the reviews. 
- For prompting, the system uses Syn-Chain, developed by [Fan et al. (2025)](https://aclanthology.org/2025.coling-main.210/) for ABSA.
- The sentiment of individual features and overall products are collected and stored for regression.

### Regression
- For the regression, an OLS is used due to the low number of products that are being examined.
