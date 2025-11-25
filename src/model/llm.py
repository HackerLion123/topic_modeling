import pandas as pd
from transformers import pipeline
from src.config import LLM_MODEL_NAME, MAX_NEW_TOKENS_LLM
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_topic_representative_docs(df: pd.DataFrame, topic_id: int, n_docs: int = 5) -> list[str]:
    """
    Gets representative documents for a specific topic.

    Args:
        df (pd.DataFrame): The DataFrame with topic assignments.
        topic_id (int): The ID of the topic.
        n_docs (int, optional): The number of representative documents to return. Defaults to 5.

    Returns:
        list[str]: A list of representative documents.
    """
    # For simplicity, we'll just sample some documents from the topic
    # A more advanced approach would use document-topic probabilities or centrality
    topic_docs = df[df['topic'] == topic_id]['cleaned_text'].tolist()
    return topic_docs[:n_docs]

def generate_topic_labels(df: pd.DataFrame) -> dict[int, str]:
    """
    Generates descriptive labels for each topic using a language model.

    Args:
        df (pd.DataFrame): The DataFrame with topic assignments and text data.

    Returns:
        dict[int, str]: A dictionary mapping topic IDs to their generated labels.
    """
    logging.info(f"Loading LLM for topic labeling: {LLM_MODEL_NAME}...")
    summarizer = pipeline("summarization", model=LLM_MODEL_NAME)
    
    topic_labels = {}
    unique_topics = sorted(df['topic'].unique())
    
    for topic in unique_topics:
        if topic == -1:  # Skip the outlier topic
            continue
            
        logging.info(f"Generating label for topic {topic}...")
        
        # Get representative documents for the topic
        representative_docs = get_topic_representative_docs(df, topic)
        
        # Create a prompt for the LLM
        prompt = f"Summarize the following documents into a short, descriptive topic label:\n\n"
        prompt += "\n".join([f"- {doc}" for doc in representative_docs])
        
        try:
            # Generate the summary (topic label)
            summary = summarizer(prompt, max_length=MAX_NEW_TOKENS_LLM, min_length=5, do_sample=False)
            label = summary[0]['summary_text']
            topic_labels[topic] = label
            logging.info(f"Generated label for topic {topic}: {label}")
        except Exception as e:
            logging.error(f"Failed to generate label for topic {topic}: {e}")
            topic_labels[topic] = f"Topic {topic}" # Fallback label

    return topic_labels
