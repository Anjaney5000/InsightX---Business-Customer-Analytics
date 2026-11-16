"""Fine-tune BERT on rating-derived review sentiment labels.

The labels are weak labels derived from star ratings (1-2 negative, 3 neutral,
4-5 positive); this is not a manually annotated sentiment benchmark.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "womens_ecommerce_reviews.csv"
OUT = ROOT / "models" / "bert_sentiment"
REPORTS = ROOT / "reports"
LABELS = {"negative": 0, "neutral": 1, "positive": 2}


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=192):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        enc = self.tokenizer(
            str(self.texts[index]), truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return item


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128,
                        help="Maximum token length for BERT inputs.")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Optional cap for quick experiments; 0 uses all rows.")
    parser.add_argument("--model-name", default="bert-base-uncased")
    args = parser.parse_args()

    if not DATA.exists():
        raise FileNotFoundError("Run python data/download_data.py first.")
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["rating"]).copy()
    df["text"] = (df["title"].fillna("").astype(str) + " " +
                  df["review_text"].fillna("").astype(str)).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df[df["rating"].between(1, 5)]
    df["label"] = np.select(
        [df["rating"] <= 2, df["rating"] == 3, df["rating"] >= 4],
        [0, 1, 2]
    )
    if args.max_samples and args.max_samples < len(df):
        df = df.groupby("label", group_keys=False).apply(
            lambda part: part.sample(
                n=max(1, round(args.max_samples * len(part) / len(df))),
                random_state=42
            )
        ).reset_index(drop=True)
    train_text, val_text, train_y, val_y = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42,
        stratify=df["label"]
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=3,
        id2label={v: k for k, v in LABELS.items()},
        label2id=LABELS
    )
    train_ds = ReviewDataset(train_text, train_y, tokenizer, max_length=args.max_length)
    val_ds = ReviewDataset(val_text, val_y, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {running_loss / max(1, len(train_loader)):.4f}")

    y_true, y_pred = evaluate(model, val_loader, device)
    REPORTS.mkdir(exist_ok=True)
    metrics = {
        "source_rows_used": int(len(df)),
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "train_rows": int(len(train_ds)),
        "validation_rows": int(len(val_ds)),
        "label_mapping": LABELS,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(
            y_true, y_pred, target_names=list(LABELS), output_dict=True,
            zero_division=0
        ),
        "label_note": "Weak labels derived from star rating, not human sentiment annotation.",
        "device": str(device),
    }
    (REPORTS / "sentiment_metrics.json").write_text(json.dumps(metrics, indent=2))
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
