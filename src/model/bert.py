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
from scipy import sparse
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

from src.config import config
from src.helper.utlis import get_device

logging.config.dictConfig(config.LOG_CONFIG)
logger = logging.getLogger(__name__)


class TextEmbedder:
    """Text embedding for topic modeling using encoder models"""
    
    def __init__(self, model_name: str, **kwargs) -> None:
        """
        Initialize text embedder.
        
        Args:
            model_name: Name of the sentence transformer model
            **kwargs: Optional parameters (batch_size, max_length, verbose)
        """
        if not model_name:
            raise ValueError("model_name is required")
        
        self.model_name = model_name
        self.batch_size = kwargs.get('batch_size')
        self.max_length = kwargs.get('max_length')
        self.verbose = kwargs.get('verbose', True)
        
        self.model = SentenceTransformer(
            self.model_name,
            #local_files_only=True,
            backend='torch'
        )
    
    def embed(
        self, docs: List[str]
    ) -> np.ndarray:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of documents to embed
        Returns:
            Numpy array of embeddings
        """
        encode_kwargs = {
            'device': get_device(),
            'show_progress_bar': self.verbose,
            'normalize_embeddings': True
        }
        
        # Only add if explicitly provided
        if self.batch_size is not None:
            encode_kwargs['batch_size'] = self.batch_size
        if self.max_length is not None:
            encode_kwargs['chunk_size'] = self.max_length
            
        return self.model.encode(docs, **encode_kwargs)
    
    def save_embed(self, embeddings: np.ndarray, output_path: Union[str, Path]) -> None:
        """
        Save embeddings to disk.
        
        Args:
            embeddings: Numpy array of embeddings to save
            output_path: Path to save the embeddings file (.npy)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure .npy extension
        if output_path.suffix != '.npy':
            output_path = output_path.with_suffix('.npy')
        
        np.save(output_path, embeddings)
        logger.info(f"Saved embeddings to {output_path} (shape: {embeddings.shape})")

    def load_embed(self, embeddings_path: Union[str, Path]) -> np.ndarray:
        """
        Load saved embeddings from disk.
        
        Args:
            embeddings_path: Path to the saved embeddings file
            
        Returns:
            Numpy array of embeddings
        """
        embeddings_path = Path(embeddings_path)
        if not embeddings_path.exists():
            raise ValueError(f"Embeddings path {embeddings_path} does not exist")
        
        embeddings = np.load(embeddings_path)
        logger.info(f"Loaded embeddings from {embeddings_path} (shape: {embeddings.shape})")
        
        return embeddings
    

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
            model: Clustering method ('hdbscan', 'kmeans', 'dbscan', 'agglomerative')
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
            
            default_params = {
                'min_cluster_size': 10,
                'min_samples':  5,
                'metric':'euclidean',
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
                f"Unknown method: {self.model}. "
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
    """
    Class-based TF-IDF for topic representation.
    
    Unlike traditional TF-IDF (document-level), c-TF-IDF works at cluster/topic level:
    - Treats all documents in a cluster as a single document
    - Computes importance of words for each topic relative to all topics
    - Formula: c-TF-IDF = tf * log(1 + A/df) where:
        - tf: frequency of word in cluster
        - A: average number of words per cluster  
        - df: frequency of word across all clusters
    """

    def __init__(self, **kwargs):
        """
        Initialize c-TF-IDF vectorizer.
        Args:
        **kwargs: Configuration parameters including:
            - ngram_range: tuple, default (1, 1)
            - use_bm25: bool, default False
            - bm25_k1: float, default 1.5
            - bm25_b: float, default 0.75
        """
        ngram_range = kwargs.get('ngram_range', (1, 1))
        
        # BM25 parameters
        self.use_bm25 = kwargs.get('use_bm25', False)
        self.bm25_k1 = kwargs.get('bm25_k1', 1.5)
        self.bm25_b = kwargs.get('bm25_b', 0.75)
        # Basic count vectorizer; tweak params as needed
        self.vectorizer = CountVectorizer(stop_words='english', ngram_range=ngram_range)
        self.vocabulary_: List[str] | None = None
        self.c_tf_idf_: np.ndarray | None = None

    def fit_transform(
        self,
        documents: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Fit the c-TF-IDF representation on the given cluster documents and return
        the c-TF-IDF matrix and feature names.

        Args:
            documents: List of strings where each string is the concatenation
                       of all documents in a cluster (one per topic/cluster).

        Returns:
            c_tf_idf: 2D numpy array, shape (n_clusters, n_features)
            feature_names: list of feature/word strings
        """
        # Apply Count Vectorizer
        X = self.vectorizer.fit_transform(documents)  
        print(X.shape)
        feature_names = self.vectorizer.get_feature_names_out().tolist()

        word_counts_per_cluster = np.asarray(X.sum(axis=1)).ravel()
        
        # Avoid division by zero
        word_counts_per_cluster[word_counts_per_cluster == 0] = 1.0

        # Average number of words per cluster (A)
        A = float(word_counts_per_cluster.mean())

        # Term frequency per cluster: tf = count / total_words_in_cluster
        #    Use sparse operations to keep it efficient, then densify at the end.
        tf = X.multiply(1.0 / word_counts_per_cluster[:, None])

        # Document frequency across clusters: df = in how many clusters term appears
        #    (i.e., number of clusters where count > 0)
        df = np.asarray((X > 0).sum(axis=0)).ravel().astype(float)
        df[df == 0] = 1.0  # avoid division by zero

        # c-TF-IDF weighting: tf * log(1 + A/df)
        idf = np.log(1.0 + (A / df)) 

        # Multiply each column by its idf
        c_tf_idf_sparse = tf.multiply(idf)

        # Store dense matrix
        c_tf_idf = c_tf_idf_sparse.toarray()

        self.vocabulary_ = feature_names
        self.c_tf_idf_ = c_tf_idf

        return c_tf_idf, feature_names

    def _bm25_weighting(
        self,
        X: np.ndarray,
        k1: float = 1.5,
        b: float = 0.75
    ) -> np.ndarray:
        """
        Apply BM25 weighting to the c-TF-IDF matrix.

        This treats each row as a "document" (cluster/topic) and each column as a term.

        Args:
            X: c-TF-IDF matrix (or any doc-term weight matrix), shape (n_docs, n_terms)
            
            k1: BM25 k1 parameter. Term Frequency saturation parameter. 
            Lower values lead to quick staturation. 
            High values leads to linear growth. (More important to frequent words)
            
            b: BM25 b parameter. Document length normalization parameter. Helps to normalize for document length.
            
        Returns:
            BM25 weighted matrix (same shape as X)
        """
        if sparse.issparse(X):
            X = X.toarray()

        X = X.astype(float, copy=False)
        n_docs, n_terms = X.shape

        if n_docs == 0 or n_terms == 0:
            return np.zeros_like(X)

        # Document lengths (here: sum of weights per row)
        dl = X.sum(axis=1)
        avgdl = dl.mean()
        if avgdl == 0:
            return np.zeros_like(X)

        # Document frequency for each term: number of docs where term is non-zero
        df = np.count_nonzero(X, axis=0).astype(float)
        df[df == 0] = 1.0

        # BM25 IDF
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

        # BM25 term weighting
        numerator = X * (k1 + 1.0)
        denominator = X + k1 * (1.0 - b + b * (dl[:, None] / avgdl))
        # Avoid division by zero
        denominator[denominator == 0] = 1e-12

        bm25 = idf * (numerator / denominator)
        return bm25



        
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
        
        self.embedding_config = embedding_model_config or {}
        self.dr_config = dr_config or {}
        self.clustering_config = clustering_config or {}
        self.c_tfidf_config = c_tfidf_config or {}
        self.random_state = random_state
        self.verbose = verbose
        
        # Extract configurations
        dr_method = self.dr_config.get('method', 'umap')
        n_components = self.dr_config.get('n_components', 5)
        
        clustering_method = self.clustering_config.get('method', 'hdbscan')
        n_clusters = self.clustering_config.get('n_clusters', None)
        
        # Initialize components
        self.embedder = TextEmbedder(**self.embedding_config)
        
        # Prepare DR params (exclude 'method' and 'n_components' keys)
        dr_params = {k: v for k, v in self.dr_config.items() 
                    if k not in ['method', 'n_components']}
        self.dim_reducer = DimensionalityReducer(
            method=dr_method,
            n_components=n_components,
            random_state=random_state,
            **dr_params
        )
        
        # Prepare clustering params (exclude 'method' and 'n_clusters' keys)
        clustering_params = {k: v for k, v in self.clustering_config.items() 
                           if k not in ['method', 'n_clusters']}
        self.clusterer = ClusteringModel(
            method=clustering_method,
            n_clusters=n_clusters,
            random_state=random_state,
            **clustering_params
        )
        
        self.ctfidf = CTFIDFVectorizer()
        
        # Fitted data
        self.embeddings_: Optional[np.ndarray] = None
        self.reduced_embeddings_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.topics_: Optional[Dict[int, List[str]]] = None
        self.topic_words_: Optional[Dict[int, List[Tuple[str, float]]]] = None
        self.documents_: Optional[List[str]] = None
        
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
        embeddings: Optional[np.ndarray] = None
    ) -> 'BERTTopicModel':
        """
        Fit the topic model on documents.
        
        Args:
            documents: List of documents to model
            embeddings: Pre-computed embeddings (optional, if None will compute)
            
        Returns:
            Self for chaining
        """
        logger.info(f"Starting topic modeling on {len(documents)} documents")
        logger.info("=" * 80)
        
        self.documents_ = documents

        if embeddings is None:
            logger.info("\nStep 1: Generating embeddings")
            self.embeddings_ = self.embedder.embed(documents)
        else:
            logger.info("\nStep 1: Using pre-computed embeddings")
            self.embeddings_ = embeddings
        
        logger.info("\nStep 2: Reducing dimensionality")
        self.reduced_embeddings_ = self.dim_reducer.fit_transform(self.embeddings_)
        
        logger.info("\nStep 3: Clustering documents")
        self.labels_ = self.clusterer.fit_predict(self.reduced_embeddings_)
        
        
        logger.info("\nStep 4: Generating topic representations with c-TF-IDF")
        self._extract_topics()
        
        logger.info("\n" + "="*80)
        logger.info("✓ Topic modeling pipeline complete!")
        logger.info(f"✓ Found {len(self.topic_words_)} topics")
        
        return self

    def _extract_topics(self):
        """Extract topic representations using c-TF-IDF."""
        if self.labels_ is None or self.documents_ is None:
            return
        
        # Group documents by topic (excluding outliers)
        docs_per_topic = {}
        for doc, label in zip(self.documents_, self.labels_):
            if label == -1:  # Skip outliers
                continue
            if label not in docs_per_topic:
                docs_per_topic[label] = []
            docs_per_topic[label].append(doc)
        
        # Concatenate documents per topic
        topic_docs = []
        topic_ids = []
        for topic_id in sorted(docs_per_topic.keys()):
            topic_docs.append(' '.join(docs_per_topic[topic_id]))
            topic_ids.append(topic_id)
        
        if len(topic_docs) == 0:
            logger.warning("No topics found (all documents are outliers)")
            self.topic_words_ = {}
            self.topics_ = {}
            return
        
        # Fit c-TF-IDF
        c_tfidf_matrix, vocab = self.ctfidf.fit_transform(topic_docs)
        
        # Extract top words for each topic
        top_n_words = self.c_tfidf_config.get("top_n_words", 10)
        self.topic_words_ = {}
        
        for i, topic_id in enumerate(topic_ids):
            # Get scores for this topic
            topic_scores = c_tfidf_matrix[i]
            
            # Get indices of top N words (sorted descending)
            top_indices = topic_scores.argsort()[-top_n_words:][::-1]
            
            # Get words and their scores
            words = [vocab[j] for j in top_indices]
            scores = [topic_scores[j] for j in top_indices]
            
            self.topic_words_[topic_id] = list(zip(words, scores))

        # Create a simple topics_ dictionary (just words, no scores)
        self.topics_ = {
            topic: [word for word, score in words]
            for topic, words in self.topic_words_.items()
        }
        
        logger.info(f"Extracted keywords for {len(self.topic_words_)} topics")

        
    
    def topic_over_time(
        self,
        documents: List[str],
        timestamps: List,
        nr_bins: int = 10
    ) -> pd.DataFrame:
        """
        Analyze topic prevalence over time.
        
        Args:
            documents: List of documents
            timestamps: Corresponding timestamps for documents
            nr_bins: Number of time bins
            
        Returns:
            DataFrame with topic prevalence over time
        """
        pass
    
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