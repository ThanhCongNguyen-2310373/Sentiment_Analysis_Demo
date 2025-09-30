"""
Main Script for Sentiment Analysis Project
Integrates all modules to create a complete end-to-end pipeline
Similar to the original project but simplified for single machine deployment
"""
import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from config import validate_config, KEYWORDS, MAX_TWEETS_PER_KEYWORD
from crawler import TwitterCrawler
from data_preprocessing import TextPreprocessor, load_raw_data
from sentiment_analysis import SentimentAnalyzer, load_processed_data
from visualization import SentimentVisualizer, load_sentiment_results

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SentimentAnalysisPipeline:
    """
    Complete sentiment analysis pipeline
    Orchestrates data collection, preprocessing, analysis, and visualization
    """
    
    def __init__(self):
        self.crawler = None
        self.preprocessor = None
        self.analyzer = None
        self.visualizer = None
        
        # Pipeline data
        self.raw_data = pd.DataFrame()
        self.processed_data = pd.DataFrame()
        self.analyzed_data = pd.DataFrame()
        
        # Results
        self.stats = {}
        
    def initialize_components(self):
        """Initialize all pipeline components"""
        logger.info("Initializing pipeline components...")
        
        try:
            # Validate configuration
            if not validate_config():
                logger.error("Configuration validation failed")
                return False
            
            # Initialize components
            self.crawler = TwitterCrawler()
            self.preprocessor = TextPreprocessor()
            self.analyzer = SentimentAnalyzer()
            self.visualizer = SentimentVisualizer()
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False
    
    def collect_data(self, keywords: Optional[list] = None, 
                    max_tweets: Optional[int] = None,
                    save_raw: bool = True) -> bool:
        """
        Collect tweet data using the crawler
        """
        logger.info("Starting data collection phase...")
        
        keywords = keywords or KEYWORDS
        max_tweets = max_tweets or MAX_TWEETS_PER_KEYWORD
        
        try:
            self.raw_data = self.crawler.crawl_multiple_keywords(
                keywords=keywords,
                max_tweets_per_keyword=max_tweets
            )
            
            if self.raw_data.empty:
                logger.warning("No data collected from API, creating sample data for demo...")
                # Create sample data for demo
                import random
                sample_tweets = [
                    "I love using GPT-4! It's amazing for coding assistance.",
                    "ChatGPT is helpful but sometimes gives wrong information.",
                    "GitHub Copilot saves me so much time when programming.",
                    "Gemini is decent but I still prefer GPT for most tasks.",
                    "The new GPT-4o model is incredibly fast and accurate.",
                    "Sora AI video generation is mind-blowing technology.",
                    "Llama 3 open source model is surprisingly good.",
                    "Claude is great for writing and analysis tasks.",
                    "AI copilots are game changers for developers.",
                    "These AI tools are making everyone more productive.",
                    "GPT sometimes hallucinates facts, need to be careful.",
                    "Copilot suggestions are usually good but sometimes off.",
                    "I'm worried about AI replacing human creativity.",
                    "The quality of AI responses keeps getting better.",
                    "Using multiple AI tools together gives best results."
                ] * 5  # 75 tweets total
                
                raw_data = []
                for i, text in enumerate(sample_tweets):
                    tweet_data = {
                        'id': f'demo_tweet_{i}',
                        'text': text,
                        'created_at': pd.Timestamp.now() - pd.Timedelta(days=random.randint(0, 30)),
                        'username': f'demo_user_{i % 15}',
                        'user_name': f'Demo User {i % 15}',
                        'retweet_count': random.randint(0, 100),
                        'like_count': random.randint(0, 500),
                        'lang': 'en',
                        'keyword': random.choice(keywords)
                    }
                    raw_data.append(tweet_data)
                
                self.raw_data = pd.DataFrame(raw_data)
                logger.info(f"Created {len(self.raw_data)} sample tweets for demo")
            
            logger.info(f"Successfully collected {len(self.raw_data)} tweets")
            
            # Save raw data
            if save_raw:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"tweets_raw_{timestamp}.csv"
                self.crawler.save_raw_data(self.raw_data, filename)
            
            return True
            
        except Exception as e:
            logger.error(f"Error during data collection: {e}")
            return False
    
    def preprocess_data(self, save_processed: bool = True) -> bool:
        """
        Preprocess the collected data
        """
        logger.info("Starting data preprocessing phase...")
        
        if self.raw_data.empty:
            logger.error("No raw data available for preprocessing")
            return False
        
        try:
            self.processed_data = self.preprocessor.preprocess_dataframe(self.raw_data)
            
            if self.processed_data.empty:
                logger.error("No data remaining after preprocessing")
                return False
            
            logger.info(f"Successfully preprocessed {len(self.processed_data)} tweets")
            
            # Save processed data
            if save_processed:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"tweets_processed_{timestamp}.csv"
                self.preprocessor.save_processed_data(self.processed_data, filename)
            
            return True
            
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            return False
    
    def analyze_sentiment(self, save_results: bool = True) -> bool:
        """
        Perform sentiment analysis on preprocessed data
        """
        logger.info("Starting sentiment analysis phase...")
        
        if self.processed_data.empty:
            logger.error("No processed data available for sentiment analysis")
            return False
        
        try:
            self.analyzed_data = self.analyzer.analyze_dataframe(self.processed_data)
            
            if self.analyzed_data.empty:
                logger.error("No results from sentiment analysis")
                return False
            
            logger.info(f"Successfully analyzed sentiment for {len(self.analyzed_data)} tweets")
            
            # Save results
            if save_results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"tweets_with_sentiment_{timestamp}.csv"
                self.analyzer.save_results(self.analyzed_data, filename)
            
            return True
            
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {e}")
            return False
    
    def create_visualizations(self, save_charts: bool = True) -> bool:
        """
        Create visualizations and generate reports
        """
        logger.info("Starting visualization phase...")
        
        if self.analyzed_data.empty:
            logger.error("No analyzed data available for visualization")
            return False
        
        try:
            # Generate all visualizations
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = f"analysis_{timestamp}"
            
            self.stats = self.visualizer.create_all_visualizations(
                self.analyzed_data, 
                output_prefix=output_prefix
            )
            
            logger.info("Successfully created all visualizations")
            return True
            
        except Exception as e:
            logger.error(f"Error during visualization: {e}")
            return False
    
    def run_complete_pipeline(self, keywords: Optional[list] = None,
                            max_tweets: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline from data collection to visualization
        """
        logger.info("=" * 60)
        logger.info("STARTING COMPLETE SENTIMENT ANALYSIS PIPELINE")
        logger.info("=" * 60)
        
        pipeline_start_time = datetime.now()
        results = {
            'success': False,
            'stages_completed': [],
            'error_message': None,
            'stats': {},
            'execution_time': None
        }
        
        try:
            # Initialize components
            if not self.initialize_components():
                results['error_message'] = "Failed to initialize components"
                return results
            results['stages_completed'].append('initialization')
            
            # Stage 1: Data Collection
            logger.info("STAGE 1: Data Collection")
            if not self.collect_data(keywords, max_tweets):
                results['error_message'] = "Failed to collect data"
                return results
            results['stages_completed'].append('data_collection')
            
            # Stage 2: Data Preprocessing
            logger.info("STAGE 2: Data Preprocessing")
            if not self.preprocess_data():
                results['error_message'] = "Failed to preprocess data"
                return results
            results['stages_completed'].append('preprocessing')
            
            # Stage 3: Sentiment Analysis
            logger.info("STAGE 3: Sentiment Analysis")
            if not self.analyze_sentiment():
                results['error_message'] = "Failed to analyze sentiment"
                return results
            results['stages_completed'].append('sentiment_analysis')
            
            # Stage 4: Visualization
            logger.info("STAGE 4: Visualization and Reporting")
            if not self.create_visualizations():
                results['error_message'] = "Failed to create visualizations"
                return results
            results['stages_completed'].append('visualization')
            
            # Success!
            pipeline_end_time = datetime.now()
            execution_time = pipeline_end_time - pipeline_start_time
            
            results.update({
                'success': True,
                'stats': self.stats,
                'execution_time': str(execution_time),
                'total_tweets_processed': len(self.analyzed_data)
            })
            
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Total execution time: {execution_time}")
            logger.info(f"Tweets processed: {len(self.analyzed_data)}")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            results['error_message'] = str(e)
            logger.error(f"Pipeline failed with error: {e}")
            return results
    
    def load_and_analyze_existing_data(self, data_file: str) -> Dict[str, Any]:
        """
        Load existing data and perform analysis/visualization only
        Useful for re-analyzing previously collected data
        """
        logger.info(f"Loading existing data from: {data_file}")
        
        try:
            # Initialize components (skip crawler)
            self.preprocessor = TextPreprocessor()
            self.analyzer = SentimentAnalyzer()
            self.visualizer = SentimentVisualizer()
            
            # Load data
            if data_file.endswith('_raw.csv'):
                self.raw_data = pd.read_csv(data_file)
                
                # Preprocess
                if not self.preprocess_data():
                    return {'success': False, 'error': 'Preprocessing failed'}
                
                # Analyze sentiment
                if not self.analyze_sentiment():
                    return {'success': False, 'error': 'Sentiment analysis failed'}
                    
            elif data_file.endswith('_processed.csv'):
                self.processed_data = pd.read_csv(data_file)
                
                # Analyze sentiment
                if not self.analyze_sentiment():
                    return {'success': False, 'error': 'Sentiment analysis failed'}
                    
            elif data_file.endswith('_sentiment.csv'):
                self.analyzed_data = pd.read_csv(data_file)
            else:
                # Try to auto-detect
                self.analyzed_data = pd.read_csv(data_file)
            
            # Create visualizations
            if not self.create_visualizations():
                return {'success': False, 'error': 'Visualization failed'}
            
            return {
                'success': True,
                'stats': self.stats,
                'total_tweets': len(self.analyzed_data)
            }
            
        except Exception as e:
            logger.error(f"Error loading and analyzing existing data: {e}")
            return {'success': False, 'error': str(e)}

def main():
    """
    Main function with command line interface
    """
    parser = argparse.ArgumentParser(description='Sentiment Analysis Pipeline')
    parser.add_argument('--mode', choices=['full', 'analyze'], default='full',
                       help='Pipeline mode: full (collect+analyze) or analyze (existing data only)')
    parser.add_argument('--keywords', nargs='+', default=None,
                       help='Keywords to search for (space-separated)')
    parser.add_argument('--max-tweets', type=int, default=None,
                       help='Maximum tweets per keyword')
    parser.add_argument('--data-file', type=str, default=None,
                       help='Existing data file for analyze mode')
    
    args = parser.parse_args()
    
    # Create pipeline instance
    pipeline = SentimentAnalysisPipeline()
    
    if args.mode == 'full':
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            keywords=args.keywords,
            max_tweets=args.max_tweets
        )
        
        if results['success']:
            print("\\n" + "=" * 50)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print(f"Stages completed: {', '.join(results['stages_completed'])}")
            print(f"Total tweets processed: {results['total_tweets_processed']}")
            print(f"Execution time: {results['execution_time']}")
            
            if results['stats']:
                print("\\nSentiment Distribution:")
                for sentiment, count in results['stats']['sentiment_distribution'].items():
                    percentage = results['stats']['sentiment_percentages'][sentiment]
                    print(f"  {sentiment}: {count} ({percentage}%)")
        else:
            print("\\n" + "=" * 50)
            print("PIPELINE FAILED!")
            print("=" * 50)
            print(f"Error: {results['error_message']}")
            print(f"Stages completed: {', '.join(results['stages_completed'])}")
    
    elif args.mode == 'analyze':
        if not args.data_file:
            print("Error: --data-file required for analyze mode")
            return
        
        # Analyze existing data
        results = pipeline.load_and_analyze_existing_data(args.data_file)
        
        if results['success']:
            print("\\nAnalysis completed successfully!")
            print(f"Total tweets analyzed: {results['total_tweets']}")
        else:
            print(f"Analysis failed: {results['error']}")

if __name__ == "__main__":
    main()