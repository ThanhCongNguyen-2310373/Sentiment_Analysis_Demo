"""
Twitter Data Crawler Module
Collects tweets using twitterapi.io service similar to the original project
"""
import requests
import pandas as pd
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from tqdm import tqdm

from config import (
    TWITTER_API_KEY, 
    TWITTER_API_BASE_URL, 
    KEYWORDS, 
    MAX_TWEETS_PER_KEYWORD,
    RATE_LIMIT_DELAY,
    DATA_DIR
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwitterCrawler:
    """
    Twitter data crawler using twitterapi.io API
    Similar to the original project's approach
    """
    
    def __init__(self, api_key: str = TWITTER_API_KEY):
        self.api_key = api_key
        self.base_url = TWITTER_API_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'X-API-Key': api_key
        })
        
    def search_tweets(self, query: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Search for tweets using the Advanced Search API endpoint
        Similar to the original project's approach
        """
        endpoint = f"{self.base_url}/search"
        
        params = {
            'query': query,
            'max_results': min(max_results, 100),  # API limit per request
            'tweet.fields': 'created_at,author_id,public_metrics,context_annotations,lang',
            'user.fields': 'name,username,verified,public_metrics',
            'expansions': 'author_id'
        }
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            tweets = []
            
            if 'data' in data:
                # Extract user information for mapping
                users = {}
                if 'includes' in data and 'users' in data['includes']:
                    users = {user['id']: user for user in data['includes']['users']}
                
                for tweet in data['data']:
                    # Get user info
                    user_info = users.get(tweet.get('author_id', ''), {})
                    
                    tweet_data = {
                        'id': tweet.get('id', ''),
                        'text': tweet.get('text', ''),
                        'created_at': tweet.get('created_at', ''),
                        'author_id': tweet.get('author_id', ''),
                        'username': user_info.get('username', ''),
                        'user_name': user_info.get('name', ''),
                        'verified': user_info.get('verified', False),
                        'retweet_count': tweet.get('public_metrics', {}).get('retweet_count', 0),
                        'like_count': tweet.get('public_metrics', {}).get('like_count', 0),
                        'reply_count': tweet.get('public_metrics', {}).get('reply_count', 0),
                        'quote_count': tweet.get('public_metrics', {}).get('quote_count', 0),
                        'lang': tweet.get('lang', ''),
                        'keyword': query
                    }
                    tweets.append(tweet_data)
                    
            return tweets
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching tweets for query '{query}': {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response for query '{query}': {e}")
            return []
    
    def crawl_multiple_keywords(self, 
                              keywords: List[str] = KEYWORDS,
                              max_tweets_per_keyword: int = MAX_TWEETS_PER_KEYWORD) -> pd.DataFrame:
        """
        Crawl tweets for multiple keywords
        """
        all_tweets = []
        
        logger.info(f"Starting to crawl tweets for {len(keywords)} keywords")
        
        for keyword in tqdm(keywords, desc="Crawling keywords"):
            logger.info(f"Crawling tweets for keyword: {keyword}")
            
            tweets_collected = 0
            max_id = None
            
            while tweets_collected < max_tweets_per_keyword:
                remaining = max_tweets_per_keyword - tweets_collected
                batch_size = min(100, remaining)  # API limit per request
                
                # Modify query for pagination if needed
                query = keyword
                if max_id:
                    query += f" max_id:{max_id}"
                
                tweets = self.search_tweets(query, batch_size)
                
                if not tweets:
                    logger.info(f"No more tweets found for keyword: {keyword}")
                    break
                
                all_tweets.extend(tweets)
                tweets_collected += len(tweets)
                
                # Get the oldest tweet ID for pagination
                if tweets:
                    max_id = min(tweet['id'] for tweet in tweets)
                
                logger.info(f"Collected {tweets_collected}/{max_tweets_per_keyword} tweets for '{keyword}'")
                
                # Rate limiting delay
                time.sleep(RATE_LIMIT_DELAY)
            
            logger.info(f"Finished crawling for keyword '{keyword}': {tweets_collected} tweets")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_tweets)
        
        if not df.empty:
            # Convert created_at to datetime
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Remove duplicates based on tweet ID
            df = df.drop_duplicates(subset=['id'])
            
            # Filter English tweets only (similar to original project)
            df = df[df['lang'] == 'en']
            
            logger.info(f"Total unique English tweets collected: {len(df)}")
        
        return df
    
    def save_raw_data(self, df: pd.DataFrame, filename: str = "tweets_raw.csv") -> str:
        """
        Save raw crawled data to CSV file
        """
        filepath = f"{DATA_DIR}/{filename}"
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Raw data saved to: {filepath}")
        return filepath

def main():
    """
    Main function to test the crawler
    """
    if not TWITTER_API_KEY:
        logger.error("Twitter API key not found. Please set TWITTER_API_KEY in .env file")
        return
    
    crawler = TwitterCrawler()
    
    # Test with a smaller subset first
    test_keywords = ["GPT", "ChatGPT"]
    logger.info("Testing crawler with keywords: " + ", ".join(test_keywords))
    
    # Crawl data
    df = crawler.crawl_multiple_keywords(test_keywords, max_tweets_per_keyword=50)
    
    if not df.empty:
        print(f"Successfully crawled {len(df)} tweets")
        print("\nSample data:")
        print(df[['text', 'username', 'created_at', 'keyword']].head())
        
        # Save data
        filepath = crawler.save_raw_data(df, "tweets_test.csv")
        print(f"\nData saved to: {filepath}")
    else:
        print("No tweets were collected. Please check your API key and connection.")

if __name__ == "__main__":
    main()