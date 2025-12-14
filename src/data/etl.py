import pandas as pd
import json

from src.config import config
from pathlib import Path


def load_data() -> pd.DataFrame:
    """ Loading data from JSON file into a pandas DataFrame."""
    data = []
    with Path(config.DATA_FILE_PATH).open('r') as file:
        for line in file:
            data.append(json.loads(line))
    
    data = pd.DataFrame(data)
    return data

if __name__ == "__main__":
    df = load_data()