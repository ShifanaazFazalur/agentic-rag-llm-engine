"""
Climate Sentiment — Inference Script
======================================
Run predictions on new text using the fine-tuned DistilBERT model.

Usage:
    python predict.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR  = "./best_model"
MAX_LENGTH = 128
LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}

def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

def predict(text: str, tokenizer, model) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs     = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred_id   = int(torch.argmax(logits))
    return {
        "text":       text,
        "prediction": LABEL_NAMES[pred_id],
        "confidence": round(probs[pred_id], 4),
        "scores":     {LABEL_NAMES[i]: round(probs[i], 4) for i in range(3)},
    }

if __name__ == "__main__":
    print("Loading model...")
    tokenizer, model = load_model(MODEL_DIR)

    samples = [
        "Climate change is the biggest threat to our planet and we need to act now.",
        "Global warming is a hoax invented by politicians to control people.",
        "Scientists released a new report on Arctic ice levels this week.",
        "Renewable energy jobs grew 12% last year according to new data.",
        "Climate policy is destroying the economy and hurting working families.",
    ]

    print("\nPredictions:\n" + "─" * 60)
    for text in samples:
        result = predict(text, tokenizer, model)
        print(f"Text       : {result['text'][:70]}...")
        print(f"Prediction : {result['prediction']} (confidence: {result['confidence']})")
        print(f"Scores     : {result['scores']}")
        print()
