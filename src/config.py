from pathlib import Path
import os

class Settings:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __str__(self):
        return (
            f"Settings Configuration:\n"
            f"  BASE_PATH: {self.BASE_PATH}\n"
            f"  DATA_FILE_PATH: {self.DATA_FILE_PATH}\n"
            f"  SEED: {self.SEED}\n"
            f"  LLM_CONFIG: {self.LLM_CONFIG}\n"
            f"  LOG_CONFIG: {self.LOG_CONFIG}\n"
            f"  embedding_model_config: {self.embedding_model_config}\n"
            f"  dr_config: {self.dr_config}\n"
            f"  clustering_config: {self.clustering_config}\n"
            f"  c_tfidf_config: {self.c_tfidf_config}\n"
            f"  random_state: {self.random_state}\n"
            f"  verbose: {self.verbose}"
        )

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