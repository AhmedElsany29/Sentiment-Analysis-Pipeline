
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def split_data(X, y, test_size=0.2):
    """
    Split dataset into train and test
    """
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    return X_train_text, X_test_text, y_train, y_test

def encode_labels(y_train, y_test):
    """
    Encode target labels into integers
    """
    encoder = LabelEncoder()

    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)

    return y_train_enc, y_test_enc, encoder

def one_hot_encode(y_train, y_test):
    """ One-hot encode the target variable"""
    encoder = OneHotEncoder(sparse_output=False)
    
    y_train_enc = encoder.fit_transform(y_train.values.reshape(-1, 1))
    y_test_enc = encoder.transform(y_test.values.reshape(-1, 1))
    
    return y_train_enc, y_test_enc ,encoder

def create_tokenizer(X_train_text , num_words = 5000):
    """ Create a tokenizer and fit it on the training text data"""
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(X_train_text)
    
    return tokenizer

def convert_to_sequences(tokenizer , X_train_text , X_test_text):
    
    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)
    
    max_len = max([len(seq)for seq in X_train_seq ])
    
    X_train_pad = pad_sequences(X_train_seq , maxlen = max_len , padding = 'pre')
    X_test_pad = pad_sequences(X_test_seq , maxlen = max_len , padding = 'pre')

    return X_train_pad , X_test_pad , max_len

