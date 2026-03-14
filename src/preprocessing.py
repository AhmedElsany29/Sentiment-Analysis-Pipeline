import pandas as pd  # For data manipulation
import re  # For regex-based text cleaning
from nltk.tokenize import word_tokenize  # For splitting text into word tokens
from nltk.corpus import stopwords as nltk_stopwords  # For loading common stop words
from nltk.stem import WordNetLemmatizer

# Download necessary resources
import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')


def preprocess_text(text: pd.Series):

    if not isinstance(text, pd.Series):
        raise TypeError("preprocess_text expects a pandas Series.")

    stopword_set = set(nltk_stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    if text.isna().any():
        print("Warning: Missing values detected in the dataset. Consider handling them before tokenization.")

    text_series = text.astype(str).str.lower()

    print("Tokenizing text...")

    def _clean_text(entry: str):
        normalized = re.sub(r'[^\w\s]', ' ', entry)
        normalized = re.sub(r"[^a-z\s]", " ", normalized)
        
        tokens = word_tokenize(normalized)
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        tokens = [token for token in tokens if token not in stopword_set]
        return " ".join(tokens)

    processed = text_series.apply(_clean_text)
    print("Tokenization completed.")

    return processed
