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
    
    SEED = 23
    
    LLM_CONFIG = {
        "model_name": "Qwen/Qwen3-4B",  # Change as needed
    }
    LOG_CONFIG = {}
    
    embedding_model_config = {
        "model_name": "Qwen/Qwen3-Embedding-0.6B", # Change as needed.
        "batch_size": 32,
        "max_length": 4096
    }
    
    dr_config = {
        "model": 'umap',
        "n_neighbors": 15,
        "n_components": 5,
        "min_dist": 0.0,
        "metric": 'cosine',
        "random_state": SEED
    }
    
    clustering_config = {
        "model": 'hdbscan',
        "min_cluster_size": 10,
        "min_samples": 5,
        "metric": 'euclidean',
        "cluster_selection_method": 'eom',
        "random_state": SEED
    }
    
    c_tfidf_config = {
        "top_n_words": 10,
        "ngram_range": (1, 2)
    }
    
    random_state: int = 42
    verbose: bool = True


config = Settings()