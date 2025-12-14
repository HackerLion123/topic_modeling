import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pickle
import re
from datetime import datetime

from src.config import config
from src.data.etl import load_data
from src.model.bert import BERTTopicModel
from src.model.eval import TopicModelEvaluator

st.set_page_config(
    page_title="BERTopic Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    pass