import pandas as pd
import json

from src.config import config



def load_data() -> pd.DataFrame:
    with open(config.DATA_FILE_PATH, 'r') as file:
        data = json.load(file)
        data = pd.DataFrame(data)
    return data


if __name__ == "__main__":
    
    df = load_data()
    print(df.head())