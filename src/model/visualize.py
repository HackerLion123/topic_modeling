import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from src.config import config

logging.config.dictConfig(config.LOG_CONFIG)
logger = logging.getLogger(__name__)


class TopicVisualizer:
    """Clean visualizations for topic modeling results."""
    
    def __init__(self, model=None):
        """
        Initialize visualizer.
        
        Args:
            model: Fitted BERTTopicModel instance (optional)
        """
        self.model = model
        
    def plot_embeddings_2d(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        hover_text: Optional[List[str]] = None,
        title: str = "Document Embeddings (2D)",
        method: str = "auto"
    ) -> go.Figure:
        """
        Plot 2D embeddings with cluster colors.
        
        Args:
            embeddings: 2D embeddings (n_docs, 2)
            labels: Cluster labels (n_docs,)
            hover_text: Hover text for points (document snippets)
            title: Plot title
            method: Reduction method for title
            
        Returns:
            Plotly figure
        """
        if embeddings.shape[1] > 2:
            logger.warning(f"Embeddings have {embeddings.shape[1]} dimensions, using first 2")
            embeddings = embeddings[:, :2]
        
        df = pd.DataFrame({
            'x': embeddings[:, 0],
            'y': embeddings[:, 1]
        })
        
        if labels is not None:
            df['Topic'] = [f"Topic {l}" if l >= 0 else "Outlier" for l in labels]
        else:
            df['Topic'] = 'Document'
        
        if hover_text is not None:
            df['Text'] = [t[:100] + "..." if len(t) > 100 else t for t in hover_text]
        
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='Topic',
            hover_data=['Text'] if hover_text else None,
            title=f"{title} ({method.upper()})",
            template='plotly_white',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(marker=dict(size=6, opacity=0.7, line=dict(width=0.5, color='white')))
        fig.update_layout(
            width=900,
            height=600,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        
        return fig
    
    def plot_embeddings_3d(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        hover_text: Optional[List[str]] = None,
        title: str = "Document Embeddings (3D)"
    ) -> go.Figure:
        """
        Plot 3D embeddings with cluster colors.
        
        Args:
            embeddings: 3D embeddings (n_docs, 3)
            labels: Cluster labels
            hover_text: Hover text for points
            title: Plot title
            
        Returns:
            Plotly figure
        """
        if embeddings.shape[1] < 3:
            raise ValueError("Need at least 3 dimensions for 3D plot")
        
        if embeddings.shape[1] > 3:
            logger.warning(f"Embeddings have {embeddings.shape[1]} dimensions, using first 3")
            embeddings = embeddings[:, :3]
        
        df = pd.DataFrame({
            'x': embeddings[:, 0],
            'y': embeddings[:, 1],
            'z': embeddings[:, 2]
        })
        
        if labels is not None:
            df['Topic'] = [f"Topic {l}" if l >= 0 else "Outlier" for l in labels]
        else:
            df['Topic'] = 'Document'
        
        if hover_text is not None:
            df['Text'] = [t[:100] + "..." if len(t) > 100 else t for t in hover_text]
        
        fig = px.scatter_3d(
            df,
            x='x',
            y='y',
            z='z',
            color='Topic',
            hover_data=['Text'] if hover_text else None,
            title=title,
            template='plotly_white',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(marker=dict(size=4, opacity=0.7))
        fig.update_layout(width=900, height=700)
        
        return fig
    
    def plot_topic_distribution(
        self,
        labels: np.ndarray,
        top_n: int = 20,
        title: str = "Topic Distribution"
    ) -> go.Figure:
        """
        Plot topic size distribution as bar chart.
        
        Args:
            labels: Cluster labels
            top_n: Show top N topics
            title: Plot title
            
        Returns:
            Plotly figure
        """
        topic_counts = pd.Series(labels).value_counts()
        
        # Separate outliers
        if -1 in topic_counts.index:
            outlier_count = topic_counts[-1]
            topic_counts = topic_counts.drop(-1)
        else:
            outlier_count = 0
        
        # Get top N
        topic_counts = topic_counts.head(top_n)
        
        df = pd.DataFrame({
            'Topic': [f"Topic {i}" for i in topic_counts.index],
            'Count': topic_counts.values
        })
        
        fig = px.bar(
            df,
            x='Topic',
            y='Count',
            title=f"{title} (Top {top_n})",
            template='plotly_white',
            color='Count',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            width=900,
            height=500,
            showlegend=False,
            xaxis_tickangle=-45
        )
        
        # Add outlier annotation if present
        if outlier_count > 0:
            fig.add_annotation(
                text=f"Outliers: {outlier_count:,} documents",
                xref="paper", yref="paper",
                x=0.98, y=0.98,
                showarrow=False,
                bgcolor="rgba(255,255,0,0.2)",
                bordercolor="orange",
                borderwidth=1
            )
        
        return fig
    
    def plot_topic_words(
        self,
        topic_words: Dict[int, List[Tuple[str, float]]],
        topic_id: int,
        n_words: int = 10,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Plot top words for a specific topic.
        
        Args:
            topic_words: Dict mapping topic_id to list of (word, score)
            topic_id: Topic to visualize
            n_words: Number of words to show
            title: Plot title
            
        Returns:
            Plotly figure
        """
        if topic_id not in topic_words:
            raise ValueError(f"Topic {topic_id} not found")
        
        words_scores = topic_words[topic_id][:n_words]
        words, scores = zip(*words_scores)
        
        df = pd.DataFrame({
            'Word': words,
            'Score': scores
        })
        
        if title is None:
            title = f"Topic {topic_id}: Top {n_words} Words"
        
        fig = px.bar(
            df,
            x='Score',
            y='Word',
            orientation='h',
            title=title,
            template='plotly_white',
            color='Score',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            width=700,
            height=400,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def plot_all_topics_words(
        self,
        topic_words: Dict[int, List[Tuple[str, float]]],
        n_topics: int = 9,
        n_words: int = 5
    ) -> go.Figure:
        """
        Plot word clouds for multiple topics in a grid.
        
        Args:
            topic_words: Dict mapping topic_id to list of (word, score)
            n_topics: Number of topics to show
            n_words: Number of words per topic
            
        Returns:
            Plotly figure
        """
        # Get top topics by ID (excluding -1)
        topic_ids = [tid for tid in sorted(topic_words.keys()) if tid >= 0][:n_topics]
        
        # Calculate grid dimensions
        n_cols = 3
        n_rows = (len(topic_ids) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f"Topic {tid}" for tid in topic_ids],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        for idx, topic_id in enumerate(topic_ids):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            
            words_scores = topic_words[topic_id][:n_words]
            words, scores = zip(*words_scores)
            
            fig.add_trace(
                go.Bar(
                    x=list(scores),
                    y=list(words),
                    orientation='h',
                    marker=dict(
                        color=scores,
                        colorscale='Viridis',
                        showscale=False
                    ),
                    showlegend=False
                ),
                row=row,
                col=col
            )
            
            fig.update_xaxes(title_text="Score", row=row, col=col)
        
        fig.update_layout(
            title_text="Topic Words Overview",
            height=300 * n_rows,
            width=1200,
            template='plotly_white'
        )
        
        return fig
    
    def plot_cluster_sizes(
        self,
        labels: np.ndarray,
        title: str = "Cluster Size Distribution"
    ) -> go.Figure:
        """
        Plot cluster size distribution as pie chart.
        
        Args:
            labels: Cluster labels
            title: Plot title
            
        Returns:
            Plotly figure
        """
        topic_counts = pd.Series(labels).value_counts()
        
        # Separate outliers
        has_outliers = -1 in topic_counts.index
        if has_outliers:
            outlier_count = topic_counts[-1]
            topic_counts = topic_counts.drop(-1)
        
        topic_labels = [f"Topic {i}" for i in topic_counts.index]
        
        if has_outliers:
            topic_labels.append("Outliers")
            counts = list(topic_counts.values) + [outlier_count]
        else:
            counts = list(topic_counts.values)
        
        fig = go.Figure(data=[go.Pie(
            labels=topic_labels,
            values=counts,
            hole=0.3,
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        
        fig.update_layout(
            title=title,
            width=700,
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def plot_document_length_by_topic(
        self,
        documents: List[str],
        labels: np.ndarray,
        top_n_topics: int = 10
    ) -> go.Figure:
        """
        Plot document length distribution by topic.
        
        Args:
            documents: List of documents
            labels: Cluster labels
            top_n_topics: Number of top topics to show
            
        Returns:
            Plotly figure
        """
        doc_lengths = [len(doc.split()) for doc in documents]
        
        df = pd.DataFrame({
            'Topic': [f"Topic {l}" if l >= 0 else "Outlier" for l in labels],
            'Length': doc_lengths
        })
        
        # Get top N topics by count
        topic_counts = df['Topic'].value_counts().head(top_n_topics)
        df = df[df['Topic'].isin(topic_counts.index)]
        
        fig = px.box(
            df,
            x='Topic',
            y='Length',
            title=f"Document Length by Topic (Top {top_n_topics})",
            template='plotly_white',
            color='Topic',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            width=1000,
            height=500,
            showlegend=False,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def save_figure(self, fig: go.Figure, filepath: Path, format: str = 'html'):
        """
        Save figure to file.
        
        Args:
            fig: Plotly figure
            filepath: Output file path
            format: Format ('html', 'png', 'jpg', 'svg', 'pdf')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'html':
            fig.write_html(str(filepath))
        else:
            fig.write_image(str(filepath))
        
        logger.info(f"Saved visualization to {filepath}")


def visualize_pipeline_results(
    model,
    output_dir: Optional[Path] = None,
    save_format: str = 'html'
) -> Dict[str, go.Figure]:
    """
    Generate all visualizations for a fitted model.
    
    Args:
        model: Fitted BERTTopicModel
        output_dir: Directory to save visualizations
        save_format: Format to save ('html', 'png')
        
    Returns:
        Dictionary of figure names to figures
    """
    if model.labels_ is None:
        raise ValueError("Model must be fitted before visualization")
    
    logger.info("Generating visualizations...")
    
    visualizer = TopicVisualizer(model)
    figures = {}
    
    # 1. 2D Embeddings
    if model.reduced_embeddings_.shape[1] >= 2:
        dr_method = model.dr_config.get('method', 'unknown')
        figures['embeddings_2d'] = visualizer.plot_embeddings_2d(
            model.reduced_embeddings_,
            model.labels_,
            model.documents_,
            method=dr_method
        )
    
    # 2. 3D Embeddings (if available)
    if model.reduced_embeddings_.shape[1] >= 3:
        figures['embeddings_3d'] = visualizer.plot_embeddings_3d(
            model.reduced_embeddings_,
            model.labels_,
            model.documents_
        )
    
    # 3. Topic Distribution
    figures['topic_distribution'] = visualizer.plot_topic_distribution(model.labels_)
    
    # 4. Cluster Sizes
    figures['cluster_sizes'] = visualizer.plot_cluster_sizes(model.labels_)
    
    # 5. All Topics Words
    if model.topic_words_:
        figures['topic_words_grid'] = visualizer.plot_all_topics_words(
            model.topic_words_,
            n_topics=9,
            n_words=5
        )
    
    # 6. Document Length by Topic
    figures['doc_length_by_topic'] = visualizer.plot_document_length_by_topic(
        model.documents_,
        model.labels_
    )
    
    # Save if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        for name, fig in figures.items():
            filepath = vis_dir / f"{name}.{save_format}"
            visualizer.save_figure(fig, filepath, format=save_format)
        
        logger.info(f"Saved {len(figures)} visualizations to {vis_dir}")
    
    return figures


if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)
    
    # Simulate reduced embeddings (2D)
    n_docs = 500
    embeddings_2d = np.random.randn(n_docs, 2)
    
    # Simulate cluster labels (3 topics + outliers)
    labels = np.random.choice([0, 1, 2, -1], size=n_docs, p=[0.3, 0.3, 0.3, 0.1])
    
    # Simulate documents
    documents = [f"Sample document {i}" for i in range(n_docs)]
    
    # Simulate topic words
    topic_words = {
        0: [("machine", 0.85), ("learning", 0.78), ("data", 0.65), ("model", 0.60), ("algorithm", 0.55)],
        1: [("market", 0.80), ("stock", 0.75), ("invest", 0.70), ("price", 0.65), ("finance", 0.60)],
        2: [("football", 0.82), ("game", 0.75), ("team", 0.68), ("player", 0.63), ("sport", 0.58)]
    }
    
    visualizer = TopicVisualizer()
    
    # Create visualizations
    fig1 = visualizer.plot_embeddings_2d(embeddings_2d, labels, documents)
    fig2 = visualizer.plot_topic_distribution(labels)
    fig3 = visualizer.plot_topic_words(topic_words, topic_id=0)
    fig4 = visualizer.plot_all_topics_words(topic_words, n_topics=3, n_words=5)
    
    fig1.show()
    
    logger.info("Visualization examples created successfully")