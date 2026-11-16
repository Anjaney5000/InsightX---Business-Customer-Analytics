"""Train and explain an XGBoost model for online session purchase intent."""
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "online_shoppers_intention.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError("Run python data/download_data.py first.")
    df = pd.read_csv(DATA)
    if "Revenue" not in df:
        raise ValueError("Expected target column 'Revenue' in UCI session data.")
    raw_target = df.pop("Revenue")
    if raw_target.dtype == bool:
        y = raw_target.astype(int)
    elif pd.api.types.is_numeric_dtype(raw_target):
        y = raw_target.astype(int)
    else:
        mapped = raw_target.astype(str).str.strip().str.lower().map(
            {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}
        )
        if mapped.isna().any():
            raise ValueError("Revenue target must contain booleans or Yes/No values.")
        y = mapped.astype(int)
    X = df.copy()

    categorical = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    model = XGBClassifier(
        n_estimators=350, max_depth=4, learning_rate=0.04,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=2,
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=-1
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "rows": int(len(df)),
        "positive_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "threshold": 0.5,
    }
    joblib.dump(pipeline, MODELS / "conversion_pipeline.joblib")
    (REPORTS / "conversion_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))

    # Explain the fitted tree model on held-out sessions after preprocessing.
    transformed = pipeline.named_steps["preprocess"].transform(X_test)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    sample = pd.DataFrame(transformed[: min(500, len(X_test))], columns=names)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    shap_values = explainer(sample)
    plt.figure()
    shap.summary_plot(shap_values.values, sample, show=False, max_display=18)
    plt.tight_layout()
    plt.savefig(REPORTS / "conversion_shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved model and reports under {MODELS} and {REPORTS}")


if __name__ == "__main__":
    main()
