"""
Climate Sentiment Classifier — Simple Version
===============================================
Fine-tunes DistilBERT on tweet sentiment using a plain PyTorch loop.
No Trainer class, no accelerate — just basic PyTorch you can explain line by line.

Steps:
  1. Load dataset from HuggingFace
  2. Tokenize the text
  3. Train with a basic PyTorch loop
  4. Track experiments with MLflow
  5. Save the best model

Usage:
  pip install torch transformers datasets scikit-learn mlflow
  python train.py
"""

import torch
import mlflow
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_NAME = "distilbert-base-uncased"
EPOCHS     = 2
BATCH_SIZE = 16
LR         = 2e-5
MAX_LEN    = 64        # shorter = faster, fine for tweets
TRAIN_SIZE = 800       # small so it runs on CPU in ~10 mins
VAL_SIZE   = 200

LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── 1. Load Data ─────────────────────────────────────────────────────────────

print("\nLoading dataset...")
dataset   = load_dataset("cardiffnlp/tweet_eval", "sentiment")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_data = dataset["train"].shuffle(seed=42).select(range(TRAIN_SIZE))
val_data   = dataset["validation"].select(range(VAL_SIZE))

# ─── 2. Tokenize ──────────────────────────────────────────────────────────────

def tokenize(batch):
    tokens = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )
    tokens["labels"] = batch["label"]
    return tokens

print("Tokenizing...")
train_data = train_data.map(tokenize, batched=True)
val_data   = val_data.map(tokenize, batched=True)

# Set format so PyTorch DataLoader can read them
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_data.set_format(type="torch",   columns=["input_ids", "attention_mask", "labels"])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE)

# ─── 3. Model ─────────────────────────────────────────────────────────────────

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)

# ─── 4. Train ─────────────────────────────────────────────────────────────────

def evaluate(loader):
    """Run model on a DataLoader, return accuracy and macro-F1."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    return round(acc, 4), round(f1, 4)


mlflow.set_experiment("climate-sentiment")

with mlflow.start_run():

    # Log hyperparameters
    mlflow.log_params({
        "model":      MODEL_NAME,
        "epochs":     EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr":         LR,
        "train_size": TRAIN_SIZE,
    })

    best_f1 = 0

    for epoch in range(EPOCHS):
        # ── Training loop ──
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss    = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 10 == 0:
                print(f"  Epoch {epoch+1} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        # ── Validation ──
        val_acc, val_f1 = evaluate(val_loader)
        print(f"\nEpoch {epoch+1} done — Loss: {avg_loss:.4f} | Val Acc: {val_acc} | Val F1: {val_f1}\n")

        mlflow.log_metrics({"loss": avg_loss, "val_acc": val_acc, "val_f1": val_f1}, step=epoch)

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained("./best_model")
            tokenizer.save_pretrained("./best_model")
            print(f"  ✅ Best model saved (F1: {best_f1})\n")

    mlflow.log_metric("best_val_f1", best_f1)
    print(f"Training complete. Best Val F1: {best_f1}")
    print("Run `mlflow ui` to view experiment dashboard.")
