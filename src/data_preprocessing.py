"""
Data Preprocessing Module
Text cleaning and preprocessing functions similar to the original project
"""
import pandas as pd
import re
import string
import logging
from typing import List, Dict, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from config import TEXT_CLEANING_CONFIG, DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    logger.warning("Could not download NLTK data. Some features may not work.")

class TextPreprocessor:
    """
    Text preprocessing class following the original project's approach
    """
    
    def __init__(self, config: Dict = TEXT_CLEANING_CONFIG):
        self.config = config
        self.stop_words = set(stopwords.words('english')) if nltk else set()
        
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        return url_pattern.sub('', text)
    
    def remove_mentions(self, text: str) -> str:
        """Remove @mentions from text"""
        mention_pattern = re.compile(r'@[\w_]+')
        return mention_pattern.sub('', text)
    
    def remove_hashtags(self, text: str) -> str:
        """Remove hashtags from text (keeping the text part)"""
        # Option 1: Remove hashtag symbol but keep text
        # hashtag_pattern = re.compile(r'#')
        # return hashtag_pattern.sub('', text)
        
        # Option 2: Remove entire hashtag (following original project)
        hashtag_pattern = re.compile(r'#[\w_]+')
        return hashtag_pattern.sub('', text)
    
    def remove_special_chars(self, text: str) -> str:
        """Remove special characters and punctuation"""
        # Keep only letters, numbers, and spaces
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return cleaned
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace and normalize spaces"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def remove_numbers(self, text: str) -> str:
        """Remove standalone numbers"""
        return re.sub(r'\b\d+\b', '', text)
    
    def remove_short_words(self, text: str, min_length: int = 2) -> str:
        """Remove words shorter than min_length"""
        words = text.split()
        return ' '.join(word for word in words if len(word) >= min_length)
    
    def clean_text(self, text: str) -> str:
        """
        Apply all cleaning steps to text
        Following the original project's SQLTransformer approach
        """
        if not isinstance(text, str):
            return ""
        
        original_text = text
        
        # Convert to lowercase first (like the original project)
        if self.config.get('convert_lowercase', True):
            text = text.lower()
        
        # Remove URLs (important for Twitter data)
        if self.config.get('remove_urls', True):
            text = self.remove_urls(text)
        
        # Remove mentions and hashtags
        if self.config.get('remove_mentions', True):
            text = self.remove_mentions(text)
            
        if self.config.get('remove_hashtags', True):
            text = self.remove_hashtags(text)
        
        # Remove special characters (keeping only letters and spaces)
        if self.config.get('remove_special_chars', True):
            text = self.remove_special_chars(text)
        
        # Remove numbers
        text = self.remove_numbers(text)
        
        # Remove extra whitespace
        if self.config.get('remove_extra_whitespace', True):
            text = self.remove_extra_whitespace(text)
        
        # Remove short words
        text = self.remove_short_words(text)
        
        # Check minimum length
        min_length = self.config.get('min_text_length', 10)
        if len(text) < min_length:
            logger.debug(f"Text too short after cleaning: '{original_text}' -> '{text}'")
            return ""
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """Remove common English stopwords"""
        if not self.stop_words:
            return text
        
        words = text.split()
        return ' '.join(word for word in words if word.lower() not in self.stop_words)
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Preprocess a DataFrame with tweet data
        Similar to the original project's Spark DataFrame processing
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for preprocessing")
            return df
        
        logger.info(f"Starting preprocessing of {len(df)} tweets")
        
        # Create a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Apply text cleaning
        logger.info("Cleaning text...")
        processed_df['cleaned_text'] = processed_df[text_column].apply(self.clean_text)
        
        # Remove empty texts after cleaning
        original_count = len(processed_df)
        processed_df = processed_df[processed_df['cleaned_text'].str.len() > 0]
        removed_count = original_count - len(processed_df)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} tweets with empty text after cleaning")
        
        # Optional: Remove stopwords (commented out by default as it might remove important sentiment words)
        # processed_df['cleaned_text'] = processed_df['cleaned_text'].apply(self.remove_stopwords)
        
        # Add preprocessing metadata
        processed_df['preprocessing_timestamp'] = pd.Timestamp.now()
        processed_df['original_length'] = df[text_column].str.len()
        processed_df['cleaned_length'] = processed_df['cleaned_text'].str.len()
        
        logger.info(f"Preprocessing completed. {len(processed_df)} tweets remaining")
        
        return processed_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "tweets_processed.csv") -> str:
        """Save processed data to CSV file"""
        filepath = f"{DATA_DIR}/{filename}"
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Processed data saved to: {filepath}")
        return filepath

def load_raw_data(filename: str = "tweets_raw.csv") -> pd.DataFrame:
    """Load raw data from CSV file"""
    filepath = f"{DATA_DIR}/{filename}"
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
        logger.info(f"Loaded {len(df)} tweets from {filepath}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        return pd.DataFrame()

def main():
    """
    Main function to test the preprocessor
    """
    # Test text cleaning
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "Check out this amazing #AI tool at https://example.com! @username",
        "I love GPT-4!!! It's so good 😍 #OpenAI #NLP",
        "RT @someone: This is a retweet with lots of !@#$%^&*() special chars",
        "Short",  # Should be removed due to length
        ""  # Empty text
    ]
    
    print("Testing text cleaning:")
    for text in test_texts:
        cleaned = preprocessor.clean_text(text)
        print(f"Original: {text}")
        print(f"Cleaned:  {cleaned}")
        print("-" * 50)
    
    # Test with sample DataFrame
    sample_data = pd.DataFrame({
        'text': test_texts,
        'id': range(len(test_texts)),
        'keyword': ['GPT'] * len(test_texts)
    })
    
    processed_df = preprocessor.preprocess_dataframe(sample_data)
    print("\nProcessed DataFrame:")
    print(processed_df[['text', 'cleaned_text']])

if __name__ == "__main__":
    main()