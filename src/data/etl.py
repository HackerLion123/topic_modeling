import pandas as pd
import json

from src.config import config
from pathlib import Path


def load_data() -> pd.DataFrame:
    """
    Loading data from various file types into a pandas DataFrame.
    
    Supported formats: JSON, JSONL, CSV, TSV, Excel (.xlsx, .xls), Parquet, Feather, Pickle
    
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        ValueError: If file type is not supported
        FileNotFoundError: If file doesn't exist
    """
    path = Path(config.DATA_FILE_PATH)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    file_extension = path.suffix.lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(path)
    elif file_extension == '.tsv':
        df = pd.read_csv(path, sep='\t')
    elif file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(path)
    elif file_extension == '.parquet':
        df = pd.read_parquet(path)
    elif file_extension == '.feather':
        df = pd.read_feather(path)
    elif file_extension in ['.pkl', '.pickle']:
        df = pd.read_pickle(path)
    elif file_extension == '.json':
        try:
            # First, try loading as regular JSON
            df = pd.read_json(path)
        except ValueError:
            # If that fails, try JSONL format (line-by-line)
            data = []
            with path.open('r') as file:
                for line in file:
                    data.append(json.loads(line))
            df = pd.DataFrame(data)
    else:
        raise ValueError(
            f"Unsupported file type: {file_extension}. "
            f"Supported types: .csv, .tsv, .json, .xlsx, .xls, .parquet, .feather, .pkl, .pickle"
        )
    
    return df

if __name__ == "__main__":
    df = load_data()