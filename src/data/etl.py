import pandas as pd
import json

from src.config import config
from pathlib import Path


def load_data() -> pd.DataFrame:
    with Path(config.DATA_FILE_PATH).open('r') as file:
        data = json.load(file)
        data = pd.DataFrame(data)
    return data



if __name__ == "__main__":
    df = load_data()
    print(df.head())