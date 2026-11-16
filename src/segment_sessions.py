"""Cluster online shopping sessions with KMeans for behavioural segmentation.

Revenue is deliberately excluded from the clustering features and is used only
after clustering to describe observed conversion by segment.
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "online_shoppers_intention.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def main():
    if not DATA.exists():
        raise FileNotFoundError("Run python data/download_data.py first.")
    df = pd.read_csv(DATA)
    y = df.pop("Revenue") if "Revenue" in df else None
    categorical = df.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    numeric = [c for c in df.columns if c not in categorical]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    matrix = ColumnTransformer(transformers).fit_transform(df)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    clusterer = KMeans(n_clusters=4, n_init=10, random_state=42)
    labels = clusterer.fit_predict(matrix)

    output = df.copy()
    output["SessionSegment"] = labels
    if y is not None:
        output["Revenue"] = y.values
    output.to_csv(REPORTS / "session_segments.csv", index=False)

    profiles = output.groupby("SessionSegment").agg(
        Sessions=("SessionSegment", "size"),
        **({"PurchaseRate": ("Revenue", "mean")} if "Revenue" in output else {})
    )
    if numeric:
        means = output.groupby("SessionSegment")[numeric].mean().add_prefix("Mean_")
        profiles = profiles.join(means)
    profiles.to_csv(REPORTS / "session_segment_profiles.csv")
    print(profiles.to_string())
    (REPORTS / "segmentation_metadata.json").write_text(json.dumps({
        "algorithm": "KMeans",
        "clusters": 4,
        "random_state": 42,
        "feature_columns": df.columns.tolist(),
        "target_excluded_from_clustering": "Revenue",
        "interpretation": "Clusters describe session behaviour; they are not stable person-level customer identities."
    }, indent=2))


if __name__ == "__main__":
    main()
