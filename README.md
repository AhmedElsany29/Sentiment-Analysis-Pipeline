# Sentiment Analysis Pipeline
Below the tree is a complete explanation covering dependencies, usage, metrics, and architecture so anyone cloning this repo can reproduce training and serve the resulting model.
```
sentiment-analysis-project/
│
├── data/
│   ├── raw/
│   │   └── sentiment_analysis.csv
│   │
│   └── processed/
│
│
├── models/
│   ├── sentiment_model.pth
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│   └── max_len.pkl
│
│
├── notebooks/
│   ├── eda.ipynb
│   └── experiments.ipynb
│
│
├── src/
│   │
│   ├── data_ingestion.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluation.py
│   └── predict.py
│
│
├── main.py
├── requirements.txt
└── README.md
```

## ⚙️ Requirements
1. Create a Python environment (Conda or `venv`).
2. Install Python packages:
   ```bash
   pip install -r requirements.txt
   pip install torch numpy pandas scikit-learn nltk joblib matplotlib
   ```
3. Download NLTK resources once:
   ```python
   import nltk
   nltk.download("punkt")
   nltk.download("stopwords")
   nltk.download("wordnet")
   ```

## ▶️ Usage
### 1. Train the pipeline
```bash
python main.py
```
- Runs preprocessing, tokenizer building, model training with BiLSTM + attention, evaluation, and saves artifacts (`models/`).
- Training uses ReduceLROnPlateau, gradient clipping, class-weighted loss, and early stopping.

### 2. Run inference
```bash
python src/predict.py
```
- Starts an interactive prompt for sentence-level predictions using the saved tokenizer and model.

## 📈 Last experiment metrics
- Early stopping triggered after epoch 5 when validation loss plateaued.
- Train loss ~0.32, train accuracy ~88.8%.
- Validation loss ~0.49, validation accuracy ~83.7%.
- **Classification report**: accuracy 84%, macro avg F1 0.83
- **Confusion matrix**:
  ```
  [[ 8322  1463  1236]
   [ 1028 14230  1298]
   [ 1068  1764 17777]]
  ```

## 🧠 Technical notes
- `SentimentModel` implements a BiLSTM plus attention layer before classification.
- Training loop includes gradient clipping, scheduler, class weights, and early stopping (see `src/train.py`).
- Artifacts saved in `models/` are loaded by `src/predict.py`.

## 🛠️ Architecture reference
- Ingest → preprocess → feature engineering → train/evaluate → save artifacts (see `docs/README_ARCH.md` for the flow diagram).

## 🧾 Ready for GitHub
- Add `models/` to `.gitignore` to avoid checking in large binaries.
- Keep the `models/` contents documented in README so future contributors can reproduce artifacts.
