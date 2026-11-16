# InsightX — Business & Customer Analytics

InsightX is a reproducible analytics project for understanding online shopping behaviour, predicting purchase intent, and classifying customer review sentiment.

## What is included

- **Customer-session conversion modelling:** an XGBoost classifier trained on the UCI Online Shoppers Purchasing Intention dataset.
- **Review sentiment modelling:** fine-tune a BERT sequence classifier on customer-written clothing reviews. Labels are derived transparently from star ratings: 1–2 negative, 3 neutral, 4–5 positive.
- **Explainability:** SHAP TreeExplainer explanations for the XGBoost model, including global feature importance and individual-session explanations.
- **Dashboard:** Streamlit overview of session purchase behaviour and review trends.

The session dataset and review dataset are analysed independently. They do not share customer identifiers, so this repository does not join them or claim a customer-level link between a purchase session and a review.

## Data sources

1. **UCI Online Shoppers Purchasing Intention** — 12,330 sessions and 18 features; target is `Revenue`. The data is imbalanced (about 15.5% purchase sessions).  
   https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
2. **Women's E-Commerce Clothing Reviews** — 23,486 anonymised customer reviews. Downloaded via the Hugging Face mirror; upstream source is the Kaggle dataset and the mirror documents CC0-1.0.  
   https://huggingface.co/datasets/chibifire/kaggle-womens-ecom-clothing-reviews  
   Upstream: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews

Dataset files are downloaded locally and are not committed to Git.

## Quick start

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python data/download_data.py
```

### Train the conversion model

```bash
python src/train_conversion.py
```

The script saves a fitted preprocessing + XGBoost pipeline to `models/conversion_pipeline.joblib`, metrics to `reports/conversion_metrics.json`, and SHAP plots to `reports/`.

### Fine-tune BERT for sentiment

```bash
python src/train_sentiment.py --epochs 2 --max-samples 0
```

This step is compute-intensive and may need a GPU. Set `--max-samples` to a smaller number for a smoke test. The fine-tuned model is saved under `models/bert_sentiment/`. It is not run automatically by the dashboard.

### Launch the dashboard

```bash
streamlit run app.py
```

The dashboard can run after data download; model training is optional for its descriptive views.

## Project structure

```text
.
├── app.py
├── data/
│   └── download_data.py
├── models/                 # generated locally; gitignored
├── reports/                # generated locally; gitignored
├── src/
│   ├── prepare.py
│   ├── train_conversion.py
│   └── train_sentiment.py
├── requirements.txt
└── README.md
```

## Methodology and responsible interpretation

- Use a stratified train/test split for purchase-intent modelling and report precision, recall, F1, ROC-AUC, and the confusion matrix; accuracy alone is misleading for an imbalanced target.
- All preprocessing is fitted inside a scikit-learn pipeline on training data to reduce leakage.
- SHAP values describe model behaviour, not causal effects.
- Sentiment classes are rating-derived weak labels; they are not independently human-annotated sentiment ground truth.
- The UCI session data is a tabular session snapshot, not a raw event stream. It supports session-level purchase-intent analysis, not user journey reconstruction.
- Results should be generated locally and reported only after running the scripts. No benchmark scores are hard-coded in this repository.

## Reproducibility

The download script stores the original dataset files in `data/raw/`. Training outputs are written to `models/` and `reports/`. Keep raw data, model binaries, and credentials out of version control.

## License and attribution

Project code is provided under the MIT License. Dataset rights remain with their respective providers; follow each dataset page's terms and citation guidance.