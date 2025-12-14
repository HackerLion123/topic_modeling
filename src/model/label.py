import pandas as pd
from transformers import pipeline
from src.config import config
import logging

from src.config import config
from src.helper.utlis import get_device

logging.config.dictConfig(config.LOG_CONFIG)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LLMTopicNamer:
    """Class to name topics using a Large Language Model (LLM)"""

    def __init__(self, model_name: str = ""):
        """
        Initialize the LLM topic namer.

        Args:
            model_name: Name of the LLM model to use
        """
        self.model_name = model_name
        self.nlp = pipeline("text-generation", model=model_name)

    def name_topics(self, topics: pd.DataFrame) -> pd.DataFrame:
        """
        Name topics using the LLM.

        Args:
            topics: DataFrame with topic information

        Returns:
            DataFrame with named topics
        """
        named_topics = []
        for _, row in topics.iterrows():
            topic_keywords = row['keywords']
            prompt = f"Provide a concise and descriptive name for a topic with the following keywords: {topic_keywords}"
            response = self.nlp(prompt, max_length=20, num_return_sequences=1)
            topic_name = response[0]['generated_text'].strip()
            named_topics.append({
                'topic_id': row['topic_id'],
                'topic_name': topic_name,
                'keywords': topic_keywords
            })
            logging.info(f"Named topic {row['topic_id']}: {topic_name}")

        return pd.DataFrame(named_topics)