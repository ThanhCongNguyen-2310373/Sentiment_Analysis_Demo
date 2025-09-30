"""
Sentiment Analysis Project Package
Author: Your Name
Description: Twitter sentiment analysis using RoBERTa model, recreating and extending the original project
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes for easy access
from .config import *
from .crawler import TwitterCrawler
from .data_preprocessing import TextPreprocessor
from .sentiment_analysis import SentimentAnalyzer
from .visualization import SentimentVisualizer
from .main import SentimentAnalysisPipeline

__all__ = [
    'TwitterCrawler',
    'TextPreprocessor', 
    'SentimentAnalyzer',
    'SentimentVisualizer',
    'SentimentAnalysisPipeline'
]