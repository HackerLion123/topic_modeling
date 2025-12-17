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
            f"  verbose: {self.verbose}"
        )

    BASE_PATH = Path(__file__).parent.parent
    DATA_FILE_PATH = BASE_PATH / "data/raw/data.json"
    
    SEED: int= 23
    
    LLM_CONFIG: dict = { # LLM to label topics
        "model_name": "Qwen/Qwen3-4B",  # Change as needed
    }
    LOG_CONFIG: dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "%(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {                              # Info and above logs will be displayed on console.
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },
            "file": {                     # Debug and above(that is all logs) logs will be saved to a file.
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": str(BASE_PATH / "output/topic_modeling.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf-8"
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "file"]
        }
    }
    
    embedding_model_config: dict = {
        "model_name": "BAAI/bge-large-en-v1.5", #"Qwen/Qwen3-Embedding-0.6B", # Change as needed.
        "batch_size": 8, # Adjust based on your hardware capabilities.
        # "max_length": 512
    }
    
    dr_config: dict = { # Dimensionality Reduction config
        "method": 'umap',
        "n_neighbors": 20,
        "n_components":5,
        "min_dist": 0.0,
        "metric": 'cosine',
        "random_state": SEED
    }
    
    clustering_config: dict = { 
        "method": 'hdbscan',
        "min_cluster_size": 3,
        "min_samples": 2,
        "metric": 'euclidean',
        "cluster_selection_method": 'eom',
        "random_state": SEED
    }
    
    c_tfidf_config: dict = { 
        "ngram_range": (1, 2),
        # "use_bm25": True,  
        # "bm25_k1": 1.5,     
        # "bm25_b": 0.75 
    }
    
    verbose: bool = True


config = Settings()