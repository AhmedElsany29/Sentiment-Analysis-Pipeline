import json
import os
import joblib
import torch
import numpy as np

from src.data_ingestion import ingest_data
from src.preprocessing import preprocess_text
from src.feature_engineering import (
    split_data,
    encode_labels,
    create_tokenizer,
    convert_to_sequences
)
from src.train import (
    SentimentModel,
    create_dataloaders,
    train_model,
    save_model
)
from src.evaluation import classification_metrics
from src.predict import *


DATA_PATH = "data/raw/sentiment_data.csv"
MODELS_DIR = "models"

VOCAB_SIZE = 5000
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001


def main():
    print("Step 1: Loading data...")
    df = ingest_data(DATA_PATH)

    print("Step 2: Preprocessing data...")
    df["cleaned_text"] = preprocess_text(df["text"])

    print("Step 3: Extracting text and labels...")
    X = df["cleaned_text"]
    y = df["sentiment"]

    print("Step 4: Splitting data...")
    X_train_text, X_test_text, y_train, y_test = split_data(X, y, test_size=0.2)

    print("Step 5: Encoding labels...")
    y_train_enc, y_test_enc, label_encoder = encode_labels(y_train, y_test)

    print("Step 6: Tokenizing text...")
    tokenizer = create_tokenizer(X_train_text, num_words=VOCAB_SIZE)

    print("Step 7: Converting text to padded sequences...")
    X_train_pad, X_test_pad, max_len = convert_to_sequences(
        tokenizer,
        X_train_text,
        X_test_text
    )

    print("Step 8: Creating dataloaders...")
    train_loader, test_loader = create_dataloaders(
        X_train_pad,
        y_train_enc,
        X_test_pad,
        y_test_enc,
        batch_size=BATCH_SIZE
    )

    print("Step 9: Building model...")
    output_dim = len(np.unique(y_train_enc))

    model = SentimentModel(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=output_dim
    )

    print("Step 10: Training model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, history = train_model(
        model,
        train_loader,
        test_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        device=device
    )

    print("Step 11: Final evaluation...")
    classification_metrics(model, test_loader, device)

    print("Step 12: Saving model and artifacts...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    save_model(model, path=os.path.join(MODELS_DIR, "sentiment_model.pth"))
    joblib.dump(tokenizer, os.path.join(MODELS_DIR, "tokenizer.pkl"))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    joblib.dump(max_len, os.path.join(MODELS_DIR, "max_len.pkl"))
    config = {
        "vocab_size": VOCAB_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": output_dim,
        "max_len": max_len,
        "num_layers": 1,
        "bidirectional": True,
        "dropout": 0.5,
    }
    with open(os.path.join(MODELS_DIR, "config.json"), "w") as f:
        json.dump(config, f)

    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
