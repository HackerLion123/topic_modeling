"""
Custom BERT-based Topic Modeling Implementation

This module implements topic modeling from scratch using:
1. BERT embeddings (sentence-transformers)
2. Multiple dimensionality reduction options (UMAP, PCA, t-SNE)
3. Multiple clustering options (HDBSCAN, K-Means, DBSCAN)
4. c-TF-IDF for topic representation
"""

import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Literal
from dataclasses import dataclass
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN

from sentence_transformers import SentenceTransformer


import hdbscan
import umap
import json

from src.helper.utlis import get_device

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextEmbedder:
    """Text embedding for topic modeling using encoder models"""
    
    def __init__(self, **kwargs) -> None:
        self.model_name = kwargs.get('model_name')
        self.batch_size = kwargs.get('batch_size', 32) # Default batch size
        self.max_length = kwargs.get('max_length', 512) # Default max token length
        self.model = SentenceTransformer(self.model_name,local_files_only=True, backend='torch')
    
    def embed(
        self, documents: List[str]
    ) -> np.ndarray:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of documents to embed
        Returns:
            Numpy array of embeddings
        """
        pass
    

class DimensionalityReducer:
    """
    Dimensionality reduction with multiple algorithm options.
    """
    
    def __init__(
        self,
        method: Literal['umap', 'pca', 'tsne', 'svd'] = 'umap',
        n_components: int = 5,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize dimensionality reducer.
        
        Args:
            method: Reduction method ('umap', 'pca', 'tsne', 'svd')
            n_components: Number of dimensions to reduce to
            random_state: Random state for reproducibility
            **kwargs: Additional parameters for specific methods
        """
        self.method = method.lower()
        self.n_components = n_components
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the appropriate dimensionality reduction model."""
        if self.method == 'umap':
            # Default UMAP parameters optimized for topic modeling
            default_params = {
                'n_neighbors': 15,
                'min_dist': 0.0,
                'metric': 'cosine',
                'random_state': self.random_state
            }
            default_params.update(self.kwargs)
            self.model = umap.UMAP(
                n_components=self.n_components,
                **default_params
            )
            logger.info(f"Initialized UMAP with n_components={self.n_components}")
            
        elif self.method == 'pca':
            # PCA for linear dimensionality reduction
            default_params = {
                'random_state': self.random_state
            }
            default_params.update(self.kwargs)
            self.model = PCA(
                n_components=self.n_components,
                **default_params
            )
            logger.info(f"Initialized PCA with n_components={self.n_components}")
            
        elif self.method == 'tsne':
            # t-SNE for non-linear reduction (primarily for visualization)
            default_params = {
                'perplexity': 30,
                'learning_rate': 200,
                'n_iter': 1000,
                'random_state': self.random_state
            }
            default_params.update(self.kwargs)
            self.model = TSNE(
                n_components=self.n_components,
                **default_params
            )
            logger.info(f"Initialized t-SNE with n_components={self.n_components}")
            
        elif self.method == 'svd':
            # TruncatedSVD (LSA) for sparse data
            default_params = {
                'random_state': self.random_state
            }
            default_params.update(self.kwargs)
            self.model = TruncatedSVD(
                n_components=self.n_components,
                **default_params
            )
            logger.info(f"Initialized TruncatedSVD with n_components={self.n_components}")
            
        else:
            raise ValueError(
                f"Unknown method: {self.method}. "
                f"Choose from: 'umap', 'pca', 'tsne', 'svd'"
            )
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduce dimensionality of embeddings.
        
        Args:
            embeddings: Numpy array of embeddings (n_samples, n_features)
            
        Returns:
            Numpy array of reduced embeddings (n_samples, n_components)
        """
        logger.info(f"Reducing dimensionality from {embeddings.shape[1]} to {self.n_components} using {self.method.upper()}")
        
        reduced = self.model.fit_transform(embeddings)
        
        logger.info(f"Dimensionality reduction complete: {reduced.shape}")
        
        # Log explained variance for PCA/SVD
        if hasattr(self.model, 'explained_variance_ratio_'):
            total_variance = self.model.explained_variance_ratio_.sum()
            logger.info(f"Explained variance: {total_variance:.2%}")
        
        return reduced


class ClusteringModel:
    """
    Clustering with multiple algorithm options.
    """
    
    def __init__(
        self,
        method: Literal['hdbscan', 'kmeans', 'dbscan', 'agglomerative'] = 'hdbscan',
        n_clusters: Optional[int] = None,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize clustering model.
        
        Args:
            method: Clustering method ('hdbscan', 'kmeans', 'dbscan', 'agglomerative')
            n_clusters: Number of clusters (required for kmeans and agglomerative)
            random_state: Random state for reproducibility
            **kwargs: Additional parameters for specific methods
        """
        self.method = method.lower()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the appropriate clustering model."""
        if self.method == 'hdbscan':
            # HDBSCAN for automatic topic discovery
            default_params = {
                'min_cluster_size': 10,
                'min_samples': 5,
                'metric': 'euclidean',
                'cluster_selection_method': 'eom',
                'prediction_data': True
            }
            default_params.update(self.kwargs)
            self.model = hdbscan.HDBSCAN(**default_params)
            logger.info(f"Initialized HDBSCAN (auto-discovers topics)")
            
        elif self.method == 'kmeans':
            # K-Means requires number of clusters
            if self.n_clusters is None:
                raise ValueError("n_clusters must be specified for K-Means")
            
            default_params = {
                'n_init': 10,
                'max_iter': 300,
                'random_state': self.random_state
            }
            default_params.update(self.kwargs)
            self.model = KMeans(
                n_clusters=self.n_clusters,
                **default_params
            )
            logger.info(f"Initialized K-Means with {self.n_clusters} clusters")
            
        elif self.method == 'dbscan':
            # DBSCAN for density-based clustering
            default_params = {
                'eps': 0.5,
                'min_samples': 5,
                'metric': 'euclidean'
            }
            default_params.update(self.kwargs)
            self.model = DBSCAN(**default_params)
            logger.info(f"Initialized DBSCAN")
            
        elif self.method == 'agglomerative':
            # Agglomerative clustering
            if self.n_clusters is None:
                raise ValueError("n_clusters must be specified for Agglomerative Clustering")
            
            from sklearn.cluster import AgglomerativeClustering
            default_params = {
                'linkage': 'ward'
            }
            default_params.update(self.kwargs)
            self.model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                **default_params
            )
            logger.info(f"Initialized Agglomerative Clustering with {self.n_clusters} clusters")
            
        else:
            raise ValueError(
                f"Unknown method: {self.method}. "
                f"Choose from: 'hdbscan', 'kmeans', 'dbscan', 'agglomerative'"
            )
    
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster embeddings.
        
        Args:
            embeddings: Numpy array of embeddings (n_samples, n_features)
            
        Returns:
            Numpy array of cluster labels (n_samples,)
            Note: HDBSCAN and DBSCAN may return -1 for noise/outliers
        """
        logger.info(f"Clustering {embeddings.shape[0]} documents using {self.method.upper()}")
        
        labels = self.model.fit_predict(embeddings)
        
        # Count unique topics
        unique_labels = set(labels)
        n_topics = len(unique_labels - {-1}) if -1 in unique_labels else len(unique_labels)
        n_outliers = sum(labels == -1)
        
        logger.info(f"Clustering complete: Found {n_topics} topics")
        if n_outliers > 0:
            logger.info(f"Outliers/Noise: {n_outliers} documents ({n_outliers/len(labels)*100:.1f}%)")
        
        # Log quality metrics if available
        if hasattr(self.model, 'inertia_'):
            logger.info(f"K-Means Inertia: {self.model.inertia_:.2f}")
        
        return labels


class CTFIDFVectorizer:
    """Class-based TF-IDF for topic representation"""
    
    def __init__(self):
        """
        Initialize c-TF-IDF vectorizer.
        
        Args:
            config: Topic modeling configuration
        """
        pass
        
class BERTTopicModel:
    """
    Complete BERT-based topic modeling pipeline.
    
    Recommended Combinations:
    -------------------------
    - **Default (Best Quality)**: UMAP + HDBSCAN
      - Auto-discovers topics, handles noise, best cluster separation
      
    - **Fast (Large Datasets)**: PCA + K-Means
      - Specify number of topics, very fast, scalable
      
    - **Balanced**: UMAP + K-Means
      - Good cluster separation with known number of topics
      
    - **Exploratory**: UMAP + HDBSCAN
      - Discover topics without prior knowledge
      
    - **Small Dataset**: t-SNE + HDBSCAN
      - Good for visualization and exploration
    """
    
    def __init__(
        self,
        embedding_model_config: Optional[Dict] = None,
        dr_config: Optional[Dict] = None,
        clustering_config: Optional[Dict] = None,
        c_tfidf_config: Optional[Dict] = None,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize the topic model.
        
        Args:
            embedding_model_config: Configuration for embedding model
            dr_config: Configuration for dimensionality reduction
            clustering_config: Configuration for clustering
            c_tfidf_config: Configuration for c-TF-IDF
            random_state: Random state for reproducibility
            verbose: Whether to print verbose output
        """
        # Use provided configs or defaults from config
        from src.config import config as default_config
        
        self.embedding_config = embedding_model_config or default_config.embedding_model_config
        self.dr_config = dr_config or default_config.dr_config
        self.clustering_config = clustering_config or default_config.clustering_config
        self.c_tfidf_config = c_tfidf_config or default_config.c_tfidf_config
        self.random_state = random_state
        self.verbose = verbose
        
        # Extract configurations
        dr_method = self.dr_config.get('model', 'umap')
        n_components = self.dr_config.get('n_components', 5)
        
        clustering_method = self.clustering_config.get('model', 'hdbscan')
        n_clusters = self.clustering_config.get('n_clusters', None)
        
        # Initialize components
        self.embedder = TextEmbedder()
        
        # Prepare DR params (exclude 'model' key)
        dr_params = {k: v for k, v in self.dr_config.items() if k not in ['model', 'n_components']}
        self.dim_reducer = DimensionalityReducer(
            method=dr_method,
            n_components=n_components,
            random_state=random_state,
            **dr_params
        )
        
        # Prepare clustering params (exclude 'model' and 'n_clusters' keys)
        clustering_params = {k: v for k, v in self.clustering_config.items() 
                           if k not in ['model', 'n_clusters']}
        self.clusterer = ClusteringModel(
            method=clustering_method,
            n_clusters=n_clusters,
            random_state=random_state,
            **clustering_params
        )
        
        self.ctfidf = CTFIDFVectorizer()
        
        # Fitted data
        self.embeddings_ = None
        self.reduced_embeddings_ = None
        self.labels_ = None
        self.topics_ = None
        self.topic_words_ = None
        self.documents_ = None
        
        if self.verbose:
            logger.info(f"Initialized BERTTopicModel:")
            logger.info(f"  - Embedding Model: {self.embedding_config.get('model_name', 'N/A')}")
            logger.info(f"  - Dimensionality Reduction: {dr_method.upper()}")
            logger.info(f"  - Clustering: {clustering_method.upper()}")
            if n_clusters:
                logger.info(f"  - Number of clusters: {n_clusters}")
        
    def fit(
        self, 
        documents: List[str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> 'BERTTopicModel':
        """
        Fit the topic model on documents.
        
        Args:
            documents: List of documents to model
            batch_size: Batch size for embedding (defaults to config)
            max_length: Max token length (defaults to config)
            
        Returns:
            Self for chaining
        """
        logger.info(f"Starting topic modeling on {len(documents)} documents")
        logger.info("=" * 80)
        
        self.documents_ = documents
        
        # Get batch_size and max_length from config if not provided
        batch_size = batch_size or self.embedding_config.get('batch_size', 32)
        max_length = max_length or self.embedding_config.get('max_length', 512)
        
        # Step 1: Embed documents
        logger.info("Step 1/4: Generating embeddings")
        self.embeddings_ = self.embedder.embed(
            documents, 
            batch_size=batch_size,
            max_length=max_length
        )
        
        # Step 2: Reduce dimensionality
        logger.info("Step 2/4: Reducing dimensionality")
        self.reduced_embeddings_ = self.dim_reducer.fit_transform(self.embeddings_)
        
        # Step 3: Cluster documents
        logger.info("Step 3/4: Clustering documents")
        self.labels_ = self.clusterer.fit_predict(self.reduced_embeddings_)
        
        # Step 4: Extract topics
        logger.info("Step 4/4: Extracting topic representations")
        ctfidf_matrix, feature_names = self.ctfidf.fit_transform(documents, self.labels_)
        self.topic_words_ = self.ctfidf.extract_topic_words(ctfidf_matrix, feature_names)
        
        logger.info("=" * 80)
        logger.info("✓ Topic modeling complete!")
        
        return self
    
    def get_topics(self) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get the extracted topics.
        
        Returns:
            Dictionary mapping topic ID to list of (word, score) tuples
        """
        if self.topic_words_ is None:
            raise ValueError("Model has not been fitted yet!")
        
        return self.topic_words_
    
    def get_topic_info(self) -> pd.DataFrame:
        """
        Get topic information as a DataFrame.
        
        Returns:
            DataFrame with topic information
        """
        if self.topic_words_ is None or self.labels_ is None:
            raise ValueError("Model has not been fitted yet!")
        
        topic_info = []
        
        # Count documents per topic
        label_counts = Counter(self.labels_)
        
        for topic_id, words in self.topic_words_.items():
            # Get top words
            top_words = [word for word, _ in words[:5]]
            
            # Create topic name
            topic_name = "_".join(top_words[:3])
            
            topic_info.append({
                'Topic': topic_id,
                'Count': label_counts.get(topic_id, 0),
                'Name': topic_name,
                'Representation': ", ".join(top_words),
                'Top_words': words
            })
        
        # Add outlier info if exists
        if -1 in label_counts:
            topic_info.append({
                'Topic': -1,
                'Count': label_counts[-1],
                'Name': 'Outliers',
                'Representation': 'Documents not assigned to any topic',
                'Top_words': []
            })
        
        df = pd.DataFrame(topic_info)
        df = df.sort_values('Count', ascending=False).reset_index(drop=True)
        
        return df
    
    def get_document_info(self) -> pd.DataFrame:
        """
        Get document-level information.
        
        Returns:
            DataFrame with document information
        """
        if self.labels_ is None or self.documents_ is None:
            raise ValueError("Model has not been fitted yet!")
        
        doc_info = pd.DataFrame({
            'Document': self.documents_,
            'Topic': self.labels_,
            'Document_Length': [len(doc.split()) for doc in self.documents_]
        })
        
        return doc_info
    
    def print_topics(self, n_words: int = 10) -> None:
        """
        Print topics in a readable format.
        
        Args:
            n_words: Number of words to display per topic
        """
        if self.topic_words_ is None:
            raise ValueError("Model has not been fitted yet!")
        
        dr_method = self.dr_config.get('model', 'unknown')
        clustering_method = self.clustering_config.get('model', 'unknown')
        
        print("\n" + "="*80)
        print(f"DISCOVERED {len(self.topic_words_)} TOPICS")
        print(f"Method: {dr_method.upper()} + {clustering_method.upper()}")
        print("="*80 + "\n")
        
        for topic_id, words in self.topic_words_.items():
            print(f"Topic {topic_id}:")
            word_str = ", ".join([f"{word} ({score:.3f})" for word, score in words[:n_words]])
            print(f"  {word_str}\n")
    
    def save(self, output_path: Union[str, Path]) -> None:
        """
        Save the topic model results.
        
        Args:
            output_path: Directory to save results
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to {output_path}")
        
        # Save topic info
        topic_info = self.get_topic_info()
        topic_info.to_csv(output_path / "topic_info.csv", index=False)
        
        # Save document info
        doc_info = self.get_document_info()
        doc_info.to_csv(output_path / "document_info.csv", index=False)
        
        # Save embeddings
        np.save(output_path / "embeddings.npy", self.embeddings_)
        np.save(output_path / "reduced_embeddings.npy", self.reduced_embeddings_)
        
        # Save configuration
        config_info = {
            'embedding_model_config': self.embedding_config,
            'dr_config': self.dr_config,
            'clustering_config': self.clustering_config,
            'c_tfidf_config': self.c_tfidf_config,
            'random_state': self.random_state
        }
        
        with open(output_path / "config.json", 'w') as f:
            json.dump(config_info, f, indent=2)
        
        logger.info("Results saved successfully!")

if __name__ == "__main__":
    # Example usage - Different configurations
    sample_docs = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand human language",
        "Computer vision enables machines to interpret visual information",
        "The stock market showed significant gains today",
        "Investors are optimistic about the economic recovery",
        "The company's quarterly earnings exceeded expectations",
        "Football is a popular sport worldwide",
        "Basketball requires teamwork and strategy",
        "Tennis players need excellent coordination"
    ]
    
    print("\n" + "="*80)
    print("="*80)
    model1 = BERTTopicModel(
        dim_reduction_method='umap',
        clustering_method='hdbscan',
        n_components=5
    )
    model1.fit(sample_docs)
    model1.print_topics()