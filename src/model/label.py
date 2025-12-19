"""
Topic Representation Refinement and Naming

Simplified implementations for:
1. KeyBERT-Inspired: Embedding-based keyword refinement
2. LLM Topic Namer: Generate descriptive topic names
"""

import pandas as pd
import numpy as np
import logging
import json
import re

from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import pipeline

from src.config import config
from src.model.bert import TextEmbedder
from src.helper.utlis import get_device

logging.config.dictConfig(config.LOG_CONFIG)
logger = logging.getLogger(__name__)


class KeyBERTInspired:
    """
    Simplified KeyBERT-inspired refinement.
    
    Re-ranks c-TF-IDF keywords using embedding similarity between
    words and topic centroids.
    """
    
    def __init__(
        self,
        embedder: TextEmbedder,
        top_n_words: int = 10,
        nr_candidate_words: int = 50,
    ):
        """
        Initialize KeyBERT refiner.
        
        Args:
            embedder: TextEmbedder instance
            top_n_words: Final number of words per topic
            nr_candidate_words: Candidates from c-TF-IDF to re-rank
        """
        self.embedder = embedder
        self.top_n_words = top_n_words
        self.nr_candidate_words = nr_candidate_words
    
    def refine_topics(
        self,
        documents: List[str],
        labels: np.ndarray,
        embeddings: np.ndarray,
        c_tf_idf: np.ndarray,
        vocab: List[str],
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Refine topics using embedding similarity.
        
        Args:
            documents: All documents
            labels: Cluster labels
            embeddings: Document embeddings
            c_tf_idf: c-TF-IDF matrix (n_topics x n_vocab)
            vocab: Vocabulary list
            
        Returns:
            {topic_id: [(word, score), ...]}
        """
        logger.info("Refining topics with KeyBERT-inspired approach")
        
        unique_topics = sorted([t for t in set(labels) if t != -1])
        refined_topics = {}
        
        # Get candidate words from c-TF-IDF
        all_candidates = set()
        for i in range(len(unique_topics)):
            top_indices = c_tf_idf[i].argsort()[-self.nr_candidate_words:]
            all_candidates.update([vocab[idx] for idx in top_indices])
        
        # Embed all candidate words once
        candidate_list = list(all_candidates)
        logger.info(f"Embedding {len(candidate_list)} candidate words")
        word_embeddings = self.embedder.embed(candidate_list)
        word_embed_dict = dict(zip(candidate_list, word_embeddings))
        
        # Process each topic
        for i, topic_id in enumerate(unique_topics):
            # Get topic centroid from document embeddings
            topic_mask = labels == topic_id
            topic_centroid = embeddings[topic_mask].mean(axis=0).reshape(1, -1)
            
            # Get candidate words for this topic
            top_indices = c_tf_idf[i].argsort()[-self.nr_candidate_words:][::-1]
            candidates = [vocab[idx] for idx in top_indices]
            
            # Get embeddings for candidates
            candidate_embeddings = np.array([word_embed_dict[word] for word in candidates])
            
            # Calculate similarity to topic centroid
            similarities = cosine_similarity(topic_centroid, candidate_embeddings)[0]
            
            # Sort by similarity and take top N
            top_word_indices = similarities.argsort()[-self.top_n_words:][::-1]
            
            refined_topics[topic_id] = [
                (candidates[idx], float(similarities[idx]))
                for idx in top_word_indices
            ]
        
        logger.info(f"Refined {len(refined_topics)} topics")
        return refined_topics


class MaximalMarginalRelevance:
    """
    Simplified MMR for diverse keyword selection.
    
    Selects keywords that are relevant to topic but diverse from each other.
    """
    
    def __init__(
        self,
        embedder: TextEmbedder,
        top_n_words: int = 10,
        nr_candidate_words: int = 50,
        diversity: float = 0.5,
    ):
        """
        Initialize MMR refiner.
        
        Args:
            embedder: TextEmbedder instance
            top_n_words: Final number of words
            nr_candidate_words: Candidates to consider
            diversity: 0=relevance only, 1=diversity only
        """
        self.embedder = embedder
        self.top_n_words = top_n_words
        self.nr_candidate_words = nr_candidate_words
        self.diversity = diversity
    
    def refine_topics(
        self,
        documents: List[str],
        labels: np.ndarray,
        embeddings: np.ndarray,
        c_tf_idf: np.ndarray,
        vocab: List[str],
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Refine topics using MMR for diversity.
        
        Args:
            documents: All documents
            labels: Cluster labels
            embeddings: Document embeddings
            c_tf_idf: c-TF-IDF matrix
            vocab: Vocabulary list
            
        Returns:
            {topic_id: [(word, score), ...]}
        """
        logger.info("Refining topics with MMR")
        
        unique_topics = sorted([t for t in set(labels) if t != -1])
        refined_topics = {}
        
        for i, topic_id in enumerate(unique_topics):
            # Get topic centroid
            topic_mask = labels == topic_id
            topic_centroid = embeddings[topic_mask].mean(axis=0)
            
            # Get candidate words
            top_indices = c_tf_idf[i].argsort()[-self.nr_candidate_words:][::-1]
            candidates = [vocab[idx] for idx in top_indices]
            
            # Embed candidates
            word_embeddings = self.embedder.embed(candidates)
            
            # MMR selection
            selected = self._mmr_select(topic_centroid, candidates, word_embeddings)
            refined_topics[topic_id] = selected
        
        logger.info(f"Refined {len(refined_topics)} topics")
        return refined_topics
    
    def _mmr_select(
        self,
        topic_centroid: np.ndarray,
        candidates: List[str],
        word_embeddings: np.ndarray,
    ) -> List[Tuple[str, float]]:
        """MMR selection algorithm."""
        # Similarities to topic
        topic_sim = cosine_similarity(
            word_embeddings,
            topic_centroid.reshape(1, -1)
        ).flatten()
        
        # Word-to-word similarities
        word_sim = cosine_similarity(word_embeddings)
        
        selected_indices = []
        remaining = list(range(len(candidates)))
        
        # Select first (most relevant)
        best = topic_sim.argmax()
        selected_indices.append(best)
        remaining.remove(best)
        
        # Select rest with MMR
        while len(selected_indices) < self.top_n_words and remaining:
            mmr_scores = []
            for idx in remaining:
                relevance = topic_sim[idx]
                redundancy = max(word_sim[idx, s] for s in selected_indices)
                mmr = (1 - self.diversity) * relevance - self.diversity * redundancy
                mmr_scores.append((idx, mmr))
            
            best = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best)
            remaining.remove(best)
        
        return [(candidates[i], float(topic_sim[i])) for i in selected_indices]


class LLMTopicNamer:
    """
    Simplified LLM-based topic naming.
    
    Generates short, descriptive names for topics using an LLM.
    """

    def __init__(
        self,
        model_name: str = None,
        max_new_tokens: int = 20,
        temperature: float = 0.3,
        use_case: str = "customer_reviews",
        **kwargs,
    ):
        """
        Initialize LLM namer.

        Args:
            model_name: HuggingFace model name
            max_new_tokens: Max tokens to generate
            temperature: Lower = more focused
        """
        self.model_name = model_name or config.LLM_CONFIG.get("model_name")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_case = use_case
        
        self.prompt = self._system_prompt()
        
        logger.info(f"Initializing LLM: {self.model_name}")
        
        try:
            device = get_device(verbose=False)
            device_id = 0 if device.type in ["cuda", "mps"] else -1
            
            self.pipe = pipeline(
                "text-generation",
                model=self.model_name,
                device=device_id,
                dtype=torch.float16 if device.type == "cuda" else torch.float32,
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}")
            self.pipe = None
            
    
    def _system_prompt(self) -> str:
        base = (
            "Task: Create a short, accurate topic label from ranked keywords.\n"
            "Write a natural-sounding noun phrase (not a keyword list).\n"
            "Be specific when the keywords support it; do not invent details.\n"
            "Avoid hype/marketing language. Avoid emojis.\n"
            "If ambiguous, choose the broadest correct umbrella label.\n"
        )

        use_case = (self.use_case or "").strip().lower()

        if use_case == "customer_reviews":
            return (
                base
                + "Domain: Customer reviews.\n"
                + "Common themes include: quality, usability, fit/size, value/price, shipping/delivery, packaging, "
                "customer support, returns/refunds, reliability.\n"
                + "Keep wording neutral unless sentiment is clearly indicated.\n"
            )

        if use_case in {"customer_complaints", "costomer_complaints"}:
            return (
                base
                + "Domain: Customer complaints.\n"
                + "Emphasize the issue/failure mode (e.g., billing problems, delivery delays, defects, account access, poor support).\n"
                + "Use professional, objective phrasing.\n"
            )

        if use_case == "social_media":
            return (
                base
                + "Domain: Social media.\n"
                + "Focus on the discussion topic/event/campaign; avoid generic platform terms.\n"
            )

        return base + "Domain: General.\n"
    

    def name_topics(
        self,
        topic_words: Dict[int, List[Tuple[str, float]]],
        top_n_words: int = 15,
    ) -> Dict[int, str]:
        """
        Generate topic names from keywords.

        Args:
            topic_words: {topic_id: [(word, score), ...]}
            top_n_words: Number of keywords to use

        Returns:
            {topic_id: "Topic Name"}
        """
        if self.pipe is None:
            return self._fallback_names(topic_words, top_n_words)
        
        topic_names = {}
        
        for topic_id, words in topic_words.items():
            
            keywords = ", ".join([w if isinstance(w, str) else w[0] for w in words[:top_n_words]])
            
            prompt = (
                f"{self.prompt}\n\n"
                "You will be given ranked keywords for ONE topic.\n"
                "Return ONLY a valid JSON object (no markdown, no code fences, no extra text).\n\n"
                "JSON schema (exact keys):\n"
                '{"topic_name": "2-6 words", "reasoning": "one short sentence"}\n\n'
                "Constraints:\n"
                "- topic_name: 2 to 6 words, descriptive, no trailing period\n"
                "- reasoning: ONE short sentence (<= 20 words) explaining why the name fits the keywords\n\n"
                f"Keywords (ranked): {keywords}\n"
                "JSON:"
            )
            
            try:
                result = self.pipe(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature
                )[0]['generated_text']
                
                print(prompt)
                # Extract response (remove prompt)
                response = result[len(prompt):].strip()
                print("#"*50)
                print(response)
                print("#"*50)
                try:
                    parsed = json.loads(response)
                    name = parsed.get("topic_name", "").strip()
                except json.JSONDecodeError:
                    match = re.search(r'"topic_name"\s*:\s*"([^"]+)"', response)
                    name = match.group(1).strip() if match else ""
                
                topic_names[topic_id] = name if name else ", ".join([w if isinstance(w, str) else w[0] for w in words[:3]])
                logger.info(f"Topic {topic_id}: {topic_names[topic_id]}")
                
            except Exception as e:
                logger.warning(f"Failed to name topic {topic_id}: {e}")
                topic_names[topic_id] = "_".join([w for w, _ in words[:3]])
        
        return topic_names
    
    def _fallback_names(
        self,
        topic_words: Dict[int, List[Tuple[str, float]]],
        n: int = 3,
    ) -> Dict[int, str]:
        """Simple keyword-based names when LLM unavailable."""
        return {
            tid: " & ".join([w for w, _ in words[:n]])
            for tid, words in topic_words.items()
        }