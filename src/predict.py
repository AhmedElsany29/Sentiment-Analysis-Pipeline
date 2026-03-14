import json
import os
import joblib

import pandas as pd
import torch

from src.train import SentimentModel
from src.preprocessing import preprocess_text
from src.feature_engineering import pad_sequences

MODELS_DIR = "models"


def load_artifacts(models_dir: str = MODELS_DIR):
    tokenizer = joblib.load(os.path.join(models_dir, "tokenizer.pkl"))
    label_encoder = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))
    max_len = joblib.load(os.path.join(models_dir, "max_len.pkl"))
    with open(os.path.join(models_dir, "config.json"), "r") as f:
        config = json.load(f)

    model = SentimentModel(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        num_layers=config.get("num_layers", 1),
        bidirectional=config.get("bidirectional", True),
        dropout=config.get("dropout", 0.5),
    )

    state_dict = torch.load(
        os.path.join(models_dir, "sentiment_model.pth"),
        map_location="cpu",
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer, label_encoder, max_len


def clean_text_for_model(text: str):
    frame = pd.DataFrame({"text": [text]})
    return preprocess_text(frame).iloc[0]


def predict_text(text: str, model, tokenizer, label_encoder, max_len: int):
    cleaned = clean_text_for_model(text)
    tokens = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(tokens, maxlen=max_len, padding="pre")
    inputs = torch.tensor(padded, dtype=torch.long)

    with torch.no_grad():
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        label = label_encoder.inverse_transform([pred_idx])[0]

    confidence = probs[0, pred_idx].item()
    return label, confidence


def interactive_predict():
    model, tokenizer, label_encoder, max_len = load_artifacts()
    print("Model and tokenizer loaded. Enter a sentence to predict (or 'q' to quit).")

    while True:
        text = input("> ").strip()
        if text.lower() in ("q", "quit", "exit"):
            break

        label, confidence = predict_text(text, model, tokenizer, label_encoder, max_len)
        print(f"Prediction: {label} (confidence {confidence:.2f})")


if __name__ == "__main__":
    interactive_predict()
