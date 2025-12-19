import logging
import logging.config
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import config

logging.config.dictConfig(config.LOG_CONFIG)
logger = logging.getLogger(__name__)


class TopicVisualizer:
    """
    Simple, input-driven visualizations for topic modeling (Plotly).

    This class does NOT store a model. Pass inputs directly to methods.
    """

    def __init__(self, template: str = "plotly_white") -> None:
        self.template = template

    @staticmethod
    def _snippet(text: str, max_chars: int = 120) -> str:
        s = "" if text is None else str(text)
        s = " ".join(s.split())
        return s if len(s) <= max_chars else s[:max_chars] + "..."

    def plot_embeddings_2d(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        hover_text: Optional[List[str]] = None,
        title: str = "Embeddings (2D)",
        max_points: int = 12000,
        random_state: int = 23,
    ) -> go.Figure:
        """
        2D scatter of embeddings with optional topic coloring.
        - If embeddings has >2 dims, uses first 2.
        - If many topics, uses continuous color to avoid huge legends.
        """
        emb = np.asarray(embeddings)
        if emb.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape={emb.shape}")
        if emb.shape[1] < 2:
            raise ValueError(f"Need at least 2 dims, got {emb.shape[1]}")

        n = emb.shape[0]
        if labels is not None:
            lab = np.asarray(labels)
            if lab.shape != (n,):
                raise ValueError(f"labels must be shape ({n},), got {lab.shape}")
        else:
            lab = None

        if hover_text is not None and len(hover_text) != n:
            raise ValueError(f"hover_text must have length {n}, got {len(hover_text)}")

        # Downsample for speed if needed
        if max_points is not None and n > max_points:
            rng = np.random.default_rng(random_state)
            idx = np.sort(rng.choice(n, size=max_points, replace=False))
            emb = emb[idx]
            if lab is not None:
                lab = lab[idx]
            if hover_text is not None:
                hover_text = [hover_text[i] for i in idx.tolist()]

        emb2 = emb[:, :2]
        df = pd.DataFrame({"x": emb2[:, 0], "y": emb2[:, 1]})

        if hover_text is not None:
            df["text"] = [self._snippet(t) for t in hover_text]

        # No labels: plain scatter
        if lab is None:
            fig = px.scatter(
                df,
                x="x",
                y="y",
                title=title,
                template=self.template,
                hover_data=["text"] if "text" in df.columns else None,
            )
            fig.update_traces(marker=dict(size=5, opacity=0.75))
            fig.update_layout(width=950, height=620, showlegend=False)
            return fig

        unique_topics = sorted(set(lab.tolist()))
        n_topics = len([t for t in unique_topics if t != -1])

        # Many topics: continuous color for inliers + separate outlier trace
        if n_topics > 25:
            inlier_mask = lab >= 0
            df_in = df[inlier_mask].copy()
            df_out = df[~inlier_mask].copy()
            df_in["topic_id"] = lab[inlier_mask].astype(int)

            fig = px.scatter(
                df_in,
                x="x",
                y="y",
                color="topic_id",
                color_continuous_scale="Turbo",
                title=title,
                template=self.template,
                hover_data=["text"] if "text" in df_in.columns else None,
            )
            fig.update_traces(marker=dict(size=5, opacity=0.7))

            if len(df_out) > 0:
                fig.add_trace(
                    go.Scattergl(
                        x=df_out["x"],
                        y=df_out["y"],
                        mode="markers",
                        name="Outliers",
                        marker=dict(size=5, color="rgba(120,120,120,0.55)"),
                        text=df_out["text"] if "text" in df_out.columns else None,
                        hovertemplate="%{text}<extra>Outlier</extra>"
                        if "text" in df_out.columns
                        else "<extra>Outlier</extra>",
                    )
                )

            fig.update_layout(width=950, height=620, showlegend=True)
            return fig

        # Few topics: categorical legend
        df["topic"] = ["Outlier" if t == -1 else f"Topic {int(t)}" for t in lab.tolist()]
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="topic",
            title=title,
            template=self.template,
            color_discrete_sequence=px.colors.qualitative.Set3,
            hover_data=["text"] if "text" in df.columns else None,
        )
        fig.update_traces(marker=dict(size=5, opacity=0.75))
        fig.update_layout(width=950, height=620)
        return fig

    def plot_topic_distribution(
        self,
        labels: np.ndarray,
        top_n: int = 25,
        title: str = "Topic Distribution",
    ) -> go.Figure:
        """Bar chart of largest topics (excludes outliers from ranking, but annotates them)."""
        lab = np.asarray(labels)
        counts = pd.Series(lab).value_counts()

        outliers = int(counts.get(-1, 0)) if -1 in counts.index else 0
        if -1 in counts.index:
            counts = counts.drop(-1)

        counts = counts.head(top_n)
        df = pd.DataFrame({"topic_id": counts.index.astype(int), "count": counts.values.astype(int)})
        df["topic"] = df["topic_id"].map(lambda t: f"Topic {t}")

        fig = px.bar(
            df,
            x="topic",
            y="count",
            title=f"{title} (Top {len(df)})",
            template=self.template,
            color="count",
            color_continuous_scale="Blues",
        )
        fig.update_layout(width=950, height=520, showlegend=False, xaxis_tickangle=-45)

        if outliers > 0:
            fig.add_annotation(
                text=f"Outliers: {outliers:,}",
                xref="paper",
                yref="paper",
                x=0.99,
                y=0.99,
                showarrow=False,
                align="right",
                bgcolor="rgba(230,230,230,0.7)",
                bordercolor="rgba(120,120,120,0.6)",
                borderwidth=1,
            )

        return fig

    def plot_topic_words(
        self,
        topic_words: Dict[int, List[Tuple[str, float]]],
        topic_id: int,
        n_words: int = 12,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Horizontal bar chart for top words in one topic."""
        if topic_id not in topic_words:
            raise ValueError(f"topic_id={topic_id} not found")

        pairs = topic_words[topic_id][:n_words]
        if not pairs:
            raise ValueError(f"topic_id={topic_id} has no words")

        words, scores = zip(*pairs)
        df = pd.DataFrame({"word": list(words), "score": list(scores)}).sort_values("score", ascending=True)

        fig = px.bar(
            df,
            x="score",
            y="word",
            orientation="h",
            title=title or f"Topic {topic_id}: Top {len(df)} Words",
            template=self.template,
            color="score",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(width=780, height=420, showlegend=False)
        return fig

    def plot_topics_words_grid(
        self,
        topic_words: Dict[int, List[Tuple[str, float]]],
        n_topics: int = 9,
        n_words: int = 7,
        n_cols: int = 3,
        title: str = "Topic Words Overview",
    ) -> go.Figure:
        """Grid of small horizontal bar charts for multiple topics."""
        if not topic_words:
            raise ValueError("topic_words is empty")

        topic_ids = [tid for tid in sorted(topic_words.keys()) if tid != -1][:n_topics]
        if not topic_ids:
            raise ValueError("No valid topic IDs to plot")

        n_rows = (len(topic_ids) + n_cols - 1) // n_cols
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f"Topic {tid}" for tid in topic_ids],
            vertical_spacing=0.12,
            horizontal_spacing=0.09,
        )

        for i, tid in enumerate(topic_ids):
            r = i // n_cols + 1
            c = i % n_cols + 1

            pairs = topic_words[tid][:n_words]
            if not pairs:
                continue

            words, scores = zip(*pairs)
            words = list(words)[::-1]   # highest at top
            scores = list(scores)[::-1]

            fig.add_trace(
                go.Bar(
                    x=list(scores),
                    y=list(words),
                    orientation="h",
                    marker=dict(color=list(scores), colorscale="Viridis", showscale=False),
                    showlegend=False,
                ),
                row=r,
                col=c,
            )
            fig.update_xaxes(title_text="Score", row=r, col=c)

        fig.update_layout(
            title_text=title,
            height=260 * n_rows + 120,
            width=1100,
            template=self.template,
        )
        return fig

    def save_figure(self, fig: go.Figure, filepath: Union[str, Path], fmt: str = "html") -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        fmt = fmt.lower().lstrip(".")

        if fmt == "html":
            fig.write_html(str(path))
        else:
            # For png/pdf/svg export you need kaleido installed.
            fig.write_image(str(path))

        logger.info(f"Saved visualization to {path}")


def visualize_pipeline_results(embeddings, labels, topics: Optional = None, output_dir: Optional[Path] = None, save_format: str = "html") -> Dict[str, go.Figure]:
    """
    Convenience helper for a fitted model (still input-driven; no model stored in TopicVisualizer).
    Args:
        embeddings: np.ndarray of shape (n_samples, n_dims)
        labels: np.ndarray of shape (n_samples,)
        topics: Optional dict of topic_id to list of (word, score) tuples
        output_dir: Optional Path to save visualizations
        save_format: "html", "png", "pdf", or "svg"
    Returns:
        Dict of plotly Figures keyed by name.
    """

    vis = TopicVisualizer()
    figs: Dict[str, go.Figure] = {}


    figs["embeddings_2d"] = vis.plot_embeddings_2d(
        embeddings=embeddings,
        labels=labels,
        hover_text=topics,
        title="Embeddings (2D)",
    )

    figs["topic_distribution"] = vis.plot_topic_distribution(labels=labels, title="Topic Distribution")
    figs["topic_words_grid"] = vis.plot_topics_words_grid(topics)

    if output_dir:
        out = Path(output_dir) / "visualizations"
        out.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            vis.save_figure(fig, out / f"{name}.{save_format}", fmt=save_format)

    return figs