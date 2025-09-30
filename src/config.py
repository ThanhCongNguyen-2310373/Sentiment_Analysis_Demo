"""
Configuration settings for the sentiment analysis project
"""
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', '')
TWITTER_API_BASE_URL = os.getenv('TWITTER_API_BASE_URL', 'https://api.twitterapi.io/v1')

# Keywords for data collection (similar to original project + new AI terms)
KEYWORDS = [
    "GPT",
    "Copilot", 
    "Gemini",
    "GPT-4o",
    "Sora",
    "Llama 3",
    "Claude",
    "ChatGPT"
]

# Data collection settings
MAX_TWEETS_PER_KEYWORD = int(os.getenv('MAX_TWEETS_PER_KEYWORD', 1000))
RATE_LIMIT_DELAY = int(os.getenv('RATE_LIMIT_DELAY', 2))

# Model configuration
SENTIMENT_MODEL = os.getenv('SENTIMENT_MODEL', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))

# File paths
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './results')
DATA_DIR = os.getenv('DATA_DIR', './data')

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Text preprocessing settings
TEXT_CLEANING_CONFIG = {
    'remove_urls': True,
    'remove_mentions': True,
    'remove_hashtags': True,
    'remove_special_chars': True,
    'convert_lowercase': True,
    'remove_extra_whitespace': True,
    'min_text_length': 10  # Minimum text length after cleaning
}

# Visualization settings
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
}

# Sentiment label mapping (RoBERTa output to readable labels)
SENTIMENT_LABEL_MAPPING = {
    'LABEL_0': 'Negative',
    'LABEL_1': 'Neutral', 
    'LABEL_2': 'Positive'
}

def validate_config() -> bool:
    """Validate configuration settings"""
    if not TWITTER_API_KEY:
        print("Warning: TWITTER_API_KEY not set. Please update .env file.")
        return False
    
    if not os.path.exists(DATA_DIR):
        print(f"Creating data directory: {DATA_DIR}")
        os.makedirs(DATA_DIR, exist_ok=True)
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    return True

if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Keywords: {KEYWORDS}")
    print(f"Model: {SENTIMENT_MODEL}")
    print(f"Max tweets per keyword: {MAX_TWEETS_PER_KEYWORD}")
    validate_config()