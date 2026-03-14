# Sentiment Analysis Pipeline

This repository delivers a complete sentiment classification pipeline built with PyTorch. It contains data ingestion, preprocessing, model training (BiLSTM with attention), evaluation, and a prediction helper.

## 📁 Project layout
- `main.py`: orchestrates the workflow—loads data, preprocesses it, tokenizes text, trains the model, evaluates it, and saves the trained artifacts (model weights, tokenizer, label encoder, config).  
- `src/preprocessing.py`: text cleaning (punctuation removal, lowercasing, lemmatization, stop-word removal).  
- `src/feature_engineering.py`: tokenization, vocabulary creation, padding/token sequence helpers.  
- `src/train.py`: `SentimentModel` (BiLSTM + attention), data loader, training loop with gradient clipping, scheduler, early stopping, and artifact saving.  
- `src/evaluation.py`: evaluation helpers (classification report, confusion matrix).  
- `src/predict.py`: interactive inference entry point that reloads the artifacts saved by training.  
- `models/`: stores the outputs (`sentiment_model.pth`, `tokenizer.pkl`, `label_encoder.pkl`, `max_len.pkl`, `config.json`). This directory should be gitignored.

## ⚙️ Requirements
1. Set up a Python environment (conda or `venv`).
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   pip install torch numpy pandas scikit-learn nltk joblib matplotlib
   ```
3. Download the NLTK resources (once):
   ```python
   import nltk
   nltk.download("punkt")
   nltk.download("stopwords")
   nltk.download("wordnet")
   ```

> The repo already pins `keras-preprocessing >= 1.3.0`, which is the lightweight tokenizer backend. Add extra packages above to avoid missing modules.

## ▶️ Usage
### 1. Train the pipeline
```bash
python main.py
```
- Logs preprocessing steps, training progress, early stopping, and evaluation metrics per epoch.
- Saves the trained state, tokenizer, label encoder, padding length, and training config under `models/`.
- The training loop already uses a scheduler (`ReduceLROnPlateau`), gradient clipping, class-weighted loss, and early stopping.

### 2. Run inference
```bash
python src/predict.py
```
- Starts an interactive prompt; type any sentence, then you get the predicted sentiment label and confidence score (enter `q` to exit).
- Inference reuses the same preprocessing/tokenizer pipeline as training so predictions stay consistent.

## 📈 Last training results
- Early stopping triggered after epoch 5 when validation loss plateaued.
- **Train loss**: ~0.32 / **Train accuracy**: ~88.8%  
- **Validation loss**: ~0.49 / **Validation accuracy**: ~83.7%  
- **Classification report**: accuracy 84%, macro avg F1 0.83  
- **Confusion matrix**:
  ```
  [[ 8322  1463  1236]
   [ 1028 14230  1298]
   [ 1068  1764 17777]]
  ```

## 🧠 Technical notes
- `SentimentModel` is a BiLSTM with a simple attention mechanism that weights each time step before classification.  
- `train_model` applies gradient clipping, scheduler, class weights, and early stopping to improve generalization.  
- Tokenizer/encoder artifacts are saved with `joblib` so inference uses the exact same vocabulary and encodings.  
- The prediction helper pads inputs using the saved `max_len` to match training sequences.

## 🛠️ Suggestions before publishing
1. Add an initialization script that downloads NLTK assets automatically before training.  
2. Include scripts or notebooks that visualize performance (metrics, confusion matrix, training curves).  
3. Document the contents of `models/` and how to regenerate each artifact in a `docs/` folder or within the README.

## 🧾 Ready for GitHub
- Ensure `models/` is listed in `.gitignore` to avoid committing binaries.  
- Mention `python main.py` and `python src/predict.py` under the Usage section so contributors know how to run training and inference.  
- Keep the performance summary and artifact information in the README so reviewers can reproduce the results.
