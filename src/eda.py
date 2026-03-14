import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


def basic_info(data: pd.DataFrame) -> None:
    
    print("Performing Exploratory Data Analysis (EDA)...")
    
    print("\nSample Data:")
    print(data.head())
    
    print("\nDataset Overview:")
    print(data.info())
    
    print("\nDataset Shape:")
    print(data.shape)
    
    print("EDA completed.")
    
def missing_val(data : pd.DataFrame) -> None:
    
    print("\nMissing Values Analysis:")
    print(data.isnull().sum())
    
def sentimaint_distribution(data: pd.DataFrame) -> None:
    
    print("\nSentiment Distribution:") 
    print(data['sentiment'].value_counts())
    plt.figure(figsize=(8, 6))
    sns.countplot(x = 'sentiment', data=data)
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()


def run_eda (data : pd.DataFrame) -> None:
    
    basic_info(data)
    
    missing_val(data)
    
    # sentimaint_distribution(data)
    
    
    

    