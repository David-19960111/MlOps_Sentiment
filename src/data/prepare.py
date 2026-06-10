import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import os

def prepare_data():
    categories = ['rec.sport.hockey','sci.med']

    train = fetch_20newsgroups(subset='train', categories=categories,
                               remove=('headers','footers','quotes'))
    
    test = fetch_20newsgroups(subset='test', categories=categories,
                               remove=('headers','footers','quotes'))
    
    os.makedirs("data/processed", exist_ok=True)

    pd.DataFrame({"text": train.data, "label": train.target}).to_csv("data/processed/train.csv", index=False)
    pd.DataFrame({"text": test.data, "label": test.target}).to_csv("data/processed/test.csv", index=False)

    print(f"Train: {len(train.data)} muestras")
    print(f"Test: {len(test.data)} muestras")

if __name__ == "__main__":
    prepare_data()