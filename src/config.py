from pathlib import Path
import os

class Settings:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    BASE_PATH = Path(__file__).parent.parent
    DATA_FILE_PATH = BASE_PATH / "data/raw/data.json"
    LLM_CONFIG = {}
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 2e-5


config = Settings()