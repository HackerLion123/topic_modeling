"""
Minimal Topic Modeling Pipeline Runner
"""

import logging
from typing import List, Optional
from pathlib import Path

from src.config import config
from src.data.etl import load_data
from src.model.bert import BERTTopicModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_topic_modeling(
    text_column: str = 'text',
    max_documents: Optional[int] = None,
    output_dir: Optional[Path] = None
) -> BERTTopicModel:
    """
    Run complete topic modeling pipeline.
    
    Args:
        text_column: Column name containing text data
        max_documents: Optional limit on number of documents
        output_dir: Directory to save results
        
    Returns:
        Fitted BERTTopicModel
    """
    logger.info("Starting topic modeling pipeline")
    logger.info("="*80)
    
    # Step 1: Load data
    logger.info("Step 1: Loading data")
    df = load_data()
    logger.info(f"Loaded {len(df)} documents")
    
    # Extract documents
    documents = df[text_column].astype(str).tolist()
    
    # Limit if specified
    if max_documents:
        documents = documents[:max_documents]
        logger.info(f"Limited to {max_documents} documents")
    
    # Step 2: Initialize model
    logger.info("\nStep 2: Initializing model")
    model = BERTTopicModel(
        dim_reduction_method='umap',
        clustering_method='hdbscan',
        n_components=config.umap_config['n_components'],
        random_state=config.random_state,
        dim_reduction_params={
            'n_neighbors': config.umap_config['n_neighbors'],
            'min_dist': config.umap_config['min_dist'],
            'metric': config.umap_config['metric']
        },
        clustering_params={
            'min_cluster_size': config.hdbscan_config['min_cluster_size'],
            'min_samples': config.hdbscan_config['min_samples'],
            'metric': config.hdbscan_config['metric']
        }
    )
    
    # Step 3: Fit model
    logger.info("\nStep 3: Fitting model")
    model.fit(documents)
    
    # Step 4: Save results
    if output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = config.BASE_PATH / f"output/run_{timestamp}"
    
    logger.info(f"\nStep 4: Saving results to {output_dir}")
    model.save(output_dir)
    
    # Print topics
    logger.info("\n" + "="*80)
    model.print_topics(n_words=10)
    
    logger.info("="*80)
    logger.info(f"✓ Pipeline complete! Results saved to: {output_dir}")
    logger.info("="*80)
    
    return model


if __name__ == "__main__":
    # Run pipeline with default settings
    model = run_topic_modeling(
        text_column='text',
        max_documents=1000  # Limit for testing
    )
    
    # Get topic info
    topic_info = model.get_topic_info()
    print("\nTopic Summary:")
    print(topic_info[['Topic', 'Count', 'Name']].head(10))