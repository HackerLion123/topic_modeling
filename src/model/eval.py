"""
Simplified evaluation metrics for topic modeling.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict
from collections import Counter

from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

from src.config import config
from src.helper.utlis import get_device

logging.config.dictConfig(config.LOG_CONFIG)
logger = logging.getLogger(__name__)


class TopicModelEvaluator:
    """
    Simple evaluation of topic modeling results.
    
    Evaluates:
    - Clustering quality (Silhouette, Davies-Bouldin)
    - Topic coherence and diversity
    - Coverage statistics
    """
    
    def __init__(self, model):
        """
        Initialize evaluator with a fitted topic model.
        
        Args:
            model: Fitted BERTTopicModel instance
        """
        self.model = model
        
        if model.labels_ is None:
            raise ValueError("Model must be fitted before evaluation!")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run all evaluation metrics.
        
        Returns:
            Dictionary of metric names to scores
        """
        logger.info("Running evaluation")
        
        results = {}
        
        # Get basic stats
        results.update(self._get_coverage_stats())
        
        # Get clustering quality
        results.update(self._evaluate_clustering())
        
        # Get topic quality
        results.update(self._evaluate_topics())
        
        # Print report
        self._print_report(results)
        
        return results
    
    def _get_coverage_stats(self) -> Dict[str, float]:
        """Get basic coverage statistics."""
        labels = self.model.labels_
        
        n_total = len(labels)
        n_outliers = np.sum(labels == -1)
        n_assigned = n_total - n_outliers
        n_topics = len(set(labels)) - (1 if -1 in labels else 0)
        
        return {
            'n_topics': n_topics,
            'n_documents': n_total,
            'n_assigned': n_assigned,
            'n_outliers': n_outliers,
            'coverage_pct': (n_assigned / n_total) * 100
        }
    
    def _evaluate_clustering(self) -> Dict[str, float]:
        """Evaluate clustering quality."""
        embeddings = self.model.reduced_embeddings_
        labels = self.model.labels_
        
        # Filter out outliers
        mask = labels >= 0
        embeddings_filtered = embeddings[mask]
        labels_filtered = labels[mask]
        
        metrics = {}
        
        if len(set(labels_filtered)) > 1:
            # Silhouette: -1 to 1, higher is better
            metrics['silhouette'] = silhouette_score(
                embeddings_filtered,
                labels_filtered,
                metric='euclidean'
            )
            
            # Davies-Bouldin: lower is better
            metrics['davies_bouldin'] = davies_bouldin_score(
                embeddings_filtered,
                labels_filtered
            )
        
        return metrics
    
    def _evaluate_topics(self) -> Dict[str, float]:
        """Evaluate topic quality."""
        metrics = {}
        
        # Topic coherence (within-topic similarity)
        metrics['coherence'] = self._compute_coherence()
        
        # Topic diversity (between-topic distinctiveness)
        metrics['diversity'] = self._compute_diversity()
        
        # Topic balance
        metrics['balance'] = self._compute_balance()
        
        return metrics
    
    def _compute_coherence(self) -> float:
        """Compute average within-topic similarity."""
        embeddings = self.model.embeddings_
        labels = self.model.labels_
        
        coherences = []
        
        for topic_id in set(labels):
            if topic_id == -1:
                continue
            
            topic_embeddings = embeddings[labels == topic_id]
            
            if len(topic_embeddings) < 2:
                continue
            
            # Average pairwise similarity within topic
            similarities = cosine_similarity(topic_embeddings)
            triu_indices = np.triu_indices_from(similarities, k=1)
            coherences.append(similarities[triu_indices].mean())
        
        return np.mean(coherences) if coherences else 0.0
    
    def _compute_diversity(self) -> float:
        """Compute how different topics are from each other."""
        embeddings = self.model.embeddings_
        labels = self.model.labels_
        
        # Get topic centroids
        centroids = []
        for topic_id in sorted(set(labels)):
            if topic_id == -1:
                continue
            
            topic_embeddings = embeddings[labels == topic_id]
            if len(topic_embeddings) > 0:
                centroids.append(topic_embeddings.mean(axis=0))
        
        if len(centroids) < 2:
            return 0.0
        
        centroids = np.array(centroids)
        
        # Compute pairwise similarities between centroids
        similarities = cosine_similarity(centroids)
        triu_indices = np.triu_indices_from(similarities, k=1)
        
        # Diversity = 1 - average similarity
        return 1 - similarities[triu_indices].mean()
    
    def _compute_balance(self) -> float:
        """Compute how balanced topic sizes are."""
        labels = self.model.labels_
        
        # Get topic sizes (excluding outliers)
        topic_sizes = [np.sum(labels == tid) for tid in set(labels) if tid != -1]
        
        if len(topic_sizes) < 2:
            return 1.0
        
        topic_sizes = np.array(topic_sizes)
        
        # Coefficient of variation
        cv = topic_sizes.std() / topic_sizes.mean() if topic_sizes.mean() > 0 else 0
        
        # Convert to 0-1 scale (1 is perfectly balanced)
        return np.exp(-cv)
    
    def _print_report(self, metrics: Dict[str, float]) -> None:
        """Print evaluation report."""
        print("\n" + "="*60)
        print("TOPIC MODEL EVALUATION")
        print("="*60)
        
        # Coverage
        print("\n📊 COVERAGE")
        print(f"  Topics:      {metrics['n_topics']}")
        print(f"  Documents:   {metrics['n_documents']}")
        print(f"  Assigned:    {metrics['n_assigned']} ({metrics['coverage_pct']:.1f}%)")
        print(f"  Outliers:    {metrics['n_outliers']}")
        
        # Clustering
        if 'silhouette' in metrics:
            print("\n🎯 CLUSTERING QUALITY")
            print(f"  Silhouette:     {metrics['silhouette']:.3f} (higher is better, -1 to 1)")
            print(f"  Davies-Bouldin: {metrics['davies_bouldin']:.3f} (lower is better)")
        
        # Topics
        if 'coherence' in metrics:
            print("\n📝 TOPIC QUALITY")
            print(f"  Coherence:   {metrics['coherence']:.3f} (higher is better)")
            print(f"  Diversity:   {metrics['diversity']:.3f} (higher is better)")
            print(f"  Balance:     {metrics['balance']:.3f} (1.0 is perfect)")
        
        print("\n" + "="*60 + "\n")
    
    def save(self) -> None:
        """
        Save evaluation metrics to JSON file.
        """
        from pathlib import Path
        import json
        
        metrics = self.evaluate()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Evaluation saved to {output_path}")
        


if __name__ == "__main__":
    pass