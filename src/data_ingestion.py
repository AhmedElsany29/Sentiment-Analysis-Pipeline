import pandas as pd
import os 


def load_data(file_path: str) -> pd.DataFrame:
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    try :
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise Exception(f"An error occurred while loading the data: {e}")

def remove_duplicates(data):
    
    return data.drop_duplicates()

def remove_missing_values(data: pd.DataFrame) -> pd.DataFrame:
  
    data = data.dropna(subset=["text", "sentiment"])
    
    return data

def ingest_data(file_path: str) -> pd.DataFrame:
    
    print("Loading dataset...")

    data = load_data(file_path)

    print("Removing missing values...")
    data = remove_missing_values(data)

    print("Removing duplicates...")
    data = remove_duplicates(data)

    print("Data ingestion completed")

    return data