"""Download the two source datasets used by InsightX.

The session and review sources are intentionally stored separately. There are no
shared customer identifiers, so they must not be joined at customer level.
"""
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def download_sessions() -> None:
    dataset = fetch_ucirepo(id=468)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    if "Revenue" not in targets.columns:
        targets.columns = ["Revenue"]
    df = pd.concat([features.reset_index(drop=True), targets.reset_index(drop=True)], axis=1)
    path = RAW / "online_shoppers_intention.csv"
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} shopping sessions to {path}")


def download_reviews() -> None:
    ds = load_dataset("chibifire/kaggle-womens-ecom-clothing-reviews", split="train")
    df = ds.to_pandas()
    path = RAW / "womens_ecommerce_reviews.csv"
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} reviews to {path}")


if __name__ == "__main__":
    download_sessions()
    download_reviews()
