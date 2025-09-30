"""
Sentiment Analysis Module using RoBERTa-Twitter
Implementation following the original project's approach with Hugging Face Transformers
"""
import pandas as pd
import numpy as np
import torch
import logging
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. Please install it using: pip install transformers")

from config import (
    SENTIMENT_MODEL, 
    BATCH_SIZE, 
    SENTIMENT_LABEL_MAPPING,
    DATA_DIR,
    OUTPUT_DIR
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Sentiment Analysis using RoBERTa-Twitter model
    Similar to the original project's Pandas UDF approach but simplified for single machine
    """
    
    def __init__(self, model_name: str = SENTIMENT_MODEL, batch_size: int = BATCH_SIZE):
        self.model_name = model_name
        self.batch_size = batch_size
        self.pipeline = None
        self.device = None
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required. Install with: pip install transformers")
        
        self._setup_model()
    
    def _setup_model(self):
        """Initialize the sentiment analysis pipeline"""
        # Check for GPU availability
        self.device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if self.device == 0 else "CPU"
        
        logger.info(f"Setting up sentiment analysis model: {self.model_name}")
        logger.info(f"Using device: {device_name}")
        
        try:
            # Load the pipeline similar to original project's approach
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device,
                return_all_scores=True  # Get scores for all labels
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Falling back to CPU...")
            
            # Fallback to CPU
            self.device = -1
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device,
                return_all_scores=True
            )
    
    def _process_predictions(self, predictions: List[List[Dict]]) -> List[Dict]:
        """
        Process raw model predictions into readable format
        Following the original project's label mapping approach
        """
        processed_results = []
        
        for pred_list in predictions:
            # Find the prediction with highest score
            best_pred = max(pred_list, key=lambda x: x['score'])
            
            # Map label to readable format
            raw_label = best_pred['label']
            readable_label = SENTIMENT_LABEL_MAPPING.get(raw_label, raw_label)
            
            # Create result dictionary
            result = {
                'sentiment_label': readable_label,
                'sentiment_score': best_pred['score'],
                'raw_label': raw_label,
                'all_scores': {
                    SENTIMENT_LABEL_MAPPING.get(item['label'], item['label']): item['score'] 
                    for item in pred_list
                }
            }
            
            processed_results.append(result)
        
        return processed_results
    
    def predict_sentiment(self, texts: List[str]) -> List[Dict]:
        """
        Predict sentiment for a list of texts
        Using batch processing for efficiency (similar to original project's batch processing)
        """
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        
        if not valid_texts:
            logger.warning("No valid texts provided for sentiment analysis")
            return []
        
        logger.info(f"Analyzing sentiment for {len(valid_texts)} texts")
        
        results = []
        
        # Process in batches for better performance
        for i in tqdm(range(0, len(valid_texts), self.batch_size), desc="Processing batches"):
            batch_texts = valid_texts[i:i + self.batch_size]
            
            try:
                # Get predictions from model
                batch_predictions = self.pipeline(batch_texts)
                
                # Process predictions
                batch_results = self._process_predictions(batch_predictions)
                results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"Error processing batch {i//self.batch_size + 1}: {e}")
                # Add empty results for failed batch
                empty_results = [{
                    'sentiment_label': 'Unknown',
                    'sentiment_score': 0.0,
                    'raw_label': 'ERROR',
                    'all_scores': {}
                }] * len(batch_texts)
                results.extend(empty_results)
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'cleaned_text') -> pd.DataFrame:
        """
        Analyze sentiment for a DataFrame of tweets
        Similar to the original project's Spark DataFrame processing
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for sentiment analysis")
            return df
        
        if text_column not in df.columns:
            logger.error(f"Column '{text_column}' not found in DataFrame")
            return df
        
        logger.info(f"Starting sentiment analysis for {len(df)} tweets")
        
        # Create a copy to avoid modifying original data
        result_df = df.copy()
        
        # Get texts for analysis
        texts = df[text_column].fillna('').astype(str).tolist()
        
        # Perform sentiment analysis
        predictions = self.predict_sentiment(texts)
        
        # Add predictions to DataFrame
        if predictions:
            # Extract main results
            result_df['sentiment_label'] = [pred['sentiment_label'] for pred in predictions]
            result_df['sentiment_score'] = [pred['sentiment_score'] for pred in predictions]
            result_df['raw_sentiment_label'] = [pred['raw_label'] for pred in predictions]
            
            # Add individual sentiment scores
            result_df['positive_score'] = [pred['all_scores'].get('Positive', 0.0) for pred in predictions]
            result_df['negative_score'] = [pred['all_scores'].get('Negative', 0.0) for pred in predictions]
            result_df['neutral_score'] = [pred['all_scores'].get('Neutral', 0.0) for pred in predictions]
            
            # Add analysis metadata
            result_df['analysis_timestamp'] = pd.Timestamp.now()
            result_df['model_used'] = self.model_name
            
            logger.info("Sentiment analysis completed successfully")
            
            # Log summary statistics
            sentiment_counts = result_df['sentiment_label'].value_counts()
            logger.info("Sentiment distribution:")
            for sentiment, count in sentiment_counts.items():
                percentage = (count / len(result_df)) * 100
                logger.info(f"  {sentiment}: {count} ({percentage:.1f}%)")
        
        else:
            logger.error("No predictions generated")
        
        return result_df
    
    def save_results(self, df: pd.DataFrame, filename: str = "tweets_with_sentiment.csv") -> str:
        """Save sentiment analysis results to CSV file"""
        filepath = f"{OUTPUT_DIR}/{filename}"
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Sentiment analysis results saved to: {filepath}")
        return filepath

def load_processed_data(filename: str = "tweets_processed.csv") -> pd.DataFrame:
    """Load processed data from CSV file"""
    filepath = f"{DATA_DIR}/{filename}"
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
        logger.info(f"Loaded {len(df)} processed tweets from {filepath}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        return pd.DataFrame()

def main():
    """
    Main function to test the sentiment analyzer
    """
    # Test with sample texts
    sample_texts = [
        "I love using GPT-4! It's amazing and so helpful.",
        "This AI tool is terrible and doesn't work at all.",
        "The new update is okay, nothing special but not bad either.",
        "ChatGPT helped me solve a complex problem today.",
        "I'm not sure about this new AI technology."
    ]
    
    print("Testing sentiment analysis...")
    
    try:
        analyzer = SentimentAnalyzer()
        
        # Test individual predictions
        results = analyzer.predict_sentiment(sample_texts)
        
        print("\nSample predictions:")
        for text, result in zip(sample_texts, results):
            print(f"Text: {text}")
            print(f"Sentiment: {result['sentiment_label']} (Score: {result['sentiment_score']:.3f})")
            print(f"All scores: {result['all_scores']}")
            print("-" * 70)
        
        # Test with DataFrame
        sample_df = pd.DataFrame({
            'cleaned_text': sample_texts,
            'id': range(len(sample_texts)),
            'keyword': ['GPT'] * len(sample_texts)
        })
        
        result_df = analyzer.analyze_dataframe(sample_df)
        print("\nDataFrame results:")
        print(result_df[['cleaned_text', 'sentiment_label', 'sentiment_score']])
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Make sure you have installed the required libraries:")
        print("pip install transformers torch")

if __name__ == "__main__":
    main()