"""
Data Visualization Module
Creating charts and graphs similar to the original project (Figures 6.4, 6.5)
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from config import VISUALIZATION_CONFIG, OUTPUT_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('default')  # Changed from seaborn-v0_8 to default
sns.set_palette("husl")

class SentimentVisualizer:
    """
    Visualization class for sentiment analysis results
    Recreating charts similar to the original project
    """
    
    def __init__(self, config: Dict = VISUALIZATION_CONFIG):
        self.config = config
        self.figure_size = config.get('figure_size', (12, 8))
        self.dpi = config.get('dpi', 300)
        self.color_palette = config.get('color_palette', ['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        # Color mapping for sentiments
        self.sentiment_colors = {
            'Positive': '#2ca02c',    # Green
            'Negative': '#d62728',    # Red  
            'Neutral': '#ff7f0e'      # Orange
        }
    
    def create_sentiment_distribution_by_keyword(self, df: pd.DataFrame, 
                                               save_path: Optional[str] = None,
                                               show_plot: bool = True) -> plt.Figure:
        """
        Create stacked bar chart showing sentiment distribution by keyword
        Similar to original project's Figure 6.4
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for visualization")
            return None
        
        # Calculate sentiment distribution by keyword
        sentiment_by_keyword = df.groupby(['keyword', 'sentiment_label']).size().unstack(fill_value=0)
        
        # Calculate percentages
        sentiment_percentages = sentiment_by_keyword.div(sentiment_by_keyword.sum(axis=1), axis=0) * 100
        
        # Create the plot
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Create stacked bar chart
        colors = [self.sentiment_colors.get(col, '#808080') for col in sentiment_percentages.columns]
        sentiment_percentages.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.7)
        
        # Customize the plot
        ax.set_title('Sentiment Distribution by Keyword', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Keywords', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Rotate x-axis labels
        ax.set_xticklabels(sentiment_percentages.index, rotation=45, ha='right')
        
        # Add percentage labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', label_type='center', fontsize=9)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Sentiment distribution chart saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def create_sentiment_count_chart(self, df: pd.DataFrame,
                                   save_path: Optional[str] = None,
                                   show_plot: bool = True) -> plt.Figure:
        """
        Create bar chart showing total sentiment counts
        """
        if df.empty:
            return None
        
        # Count sentiments
        sentiment_counts = df['sentiment_label'].value_counts()
        
        # Create the plot
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        colors = [self.sentiment_colors.get(sentiment, '#808080') for sentiment in sentiment_counts.index]
        bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors, alpha=0.8)
        
        # Customize the plot
        ax.set_title('Overall Sentiment Distribution', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Sentiment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Tweets', fontsize=12, fontweight='bold')
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(sentiment_counts),
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Sentiment count chart saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def create_timeline_analysis(self, df: pd.DataFrame,
                               save_path: Optional[str] = None,
                               show_plot: bool = True) -> plt.Figure:
        """
        Create timeline chart showing sentiment over time
        """
        if df.empty or 'created_at' not in df.columns:
            logger.warning("DataFrame missing 'created_at' column for timeline analysis")
            return None
        
        # Convert created_at to datetime if not already
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Group by date and sentiment
        df['date'] = df['created_at'].dt.date
        timeline_data = df.groupby(['date', 'sentiment_label']).size().unstack(fill_value=0)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Plot lines for each sentiment
        for sentiment in timeline_data.columns:
            color = self.sentiment_colors.get(sentiment, '#808080')
            ax.plot(timeline_data.index, timeline_data[sentiment], 
                   marker='o', label=sentiment, color=color, linewidth=2, markersize=6)
        
        # Customize the plot
        ax.set_title('Sentiment Trends Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Tweets', fontsize=12, fontweight='bold')
        ax.legend(title='Sentiment')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Timeline analysis chart saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def create_interactive_dashboard(self, df: pd.DataFrame,
                                   save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive dashboard using Plotly
        """
        if df.empty:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment by Keyword', 'Overall Distribution', 
                          'Top Keywords', 'Sentiment Scores'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # 1. Sentiment by keyword (stacked bar)
        sentiment_by_keyword = df.groupby(['keyword', 'sentiment_label']).size().unstack(fill_value=0)
        
        for sentiment in sentiment_by_keyword.columns:
            fig.add_trace(
                go.Bar(name=sentiment, 
                      x=sentiment_by_keyword.index, 
                      y=sentiment_by_keyword[sentiment],
                      marker_color=self.sentiment_colors.get(sentiment, '#808080')),
                row=1, col=1
            )
        
        # 2. Overall sentiment pie chart
        sentiment_counts = df['sentiment_label'].value_counts()
        fig.add_trace(
            go.Pie(labels=sentiment_counts.index, 
                  values=sentiment_counts.values,
                  marker_colors=[self.sentiment_colors.get(s, '#808080') for s in sentiment_counts.index]),
            row=1, col=2
        )
        
        # 3. Top keywords
        keyword_counts = df['keyword'].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=keyword_counts.values, 
                  y=keyword_counts.index,
                  orientation='h',
                  marker_color='lightblue'),
            row=2, col=1
        )
        
        # 4. Sentiment score distribution
        if 'sentiment_score' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['sentiment_score'], 
                           nbinsx=30,
                           marker_color='lightgreen'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Sentiment Analysis Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Keywords", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Keywords", row=2, col=1)
        fig.update_xaxes(title_text="Sentiment Score", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to: {save_path}")
        
        return fig
    
    def create_comparison_with_original(self, df: pd.DataFrame,
                                     original_results: Dict,
                                     save_path: Optional[str] = None,
                                     show_plot: bool = True) -> plt.Figure:
        """
        Create comparison chart with original project results
        """
        # This would compare with the original project's results
        # For now, we'll create a placeholder implementation
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
        
        # Current results
        current_sentiment = df['sentiment_label'].value_counts(normalize=True) * 100
        ax1.pie(current_sentiment.values, labels=current_sentiment.index, autopct='%1.1f%%',
               colors=[self.sentiment_colors.get(s, '#808080') for s in current_sentiment.index])
        ax1.set_title('Current Results', fontsize=14, fontweight='bold')
        
        # Original results (placeholder - you would replace with actual data)
        if original_results:
            original_sentiment = pd.Series(original_results)
            ax2.pie(original_sentiment.values, labels=original_sentiment.index, autopct='%1.1f%%',
                   colors=[self.sentiment_colors.get(s, '#808080') for s in original_sentiment.index])
        ax2.set_title('Original Project Results', fontsize=14, fontweight='bold')
        
        plt.suptitle('Comparison with Original Project', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Comparison chart saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def generate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the report
        """
        if df.empty:
            return {}
        
        stats = {
            'total_tweets': len(df),
            'unique_keywords': df['keyword'].nunique() if 'keyword' in df.columns else 0,
            'date_range': {
                'start': df['created_at'].min() if 'created_at' in df.columns else None,
                'end': df['created_at'].max() if 'created_at' in df.columns else None
            },
            'sentiment_distribution': df['sentiment_label'].value_counts().to_dict(),
            'sentiment_percentages': (df['sentiment_label'].value_counts(normalize=True) * 100).round(2).to_dict(),
            'avg_sentiment_score': df['sentiment_score'].mean() if 'sentiment_score' in df.columns else None,
            'keywords_analysis': {}
        }
        
        # Per-keyword analysis
        if 'keyword' in df.columns:
            for keyword in df['keyword'].unique():
                keyword_data = df[df['keyword'] == keyword]
                stats['keywords_analysis'][keyword] = {
                    'count': len(keyword_data),
                    'sentiment_distribution': keyword_data['sentiment_label'].value_counts().to_dict(),
                    'avg_score': keyword_data['sentiment_score'].mean() if 'sentiment_score' in keyword_data.columns else None
                }
        
        return stats
    
    def create_all_visualizations(self, df: pd.DataFrame, output_prefix: str = "sentiment_analysis"):
        """
        Create all visualizations and save them
        """
        logger.info("Creating all visualizations...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Sentiment distribution by keyword
        self.create_sentiment_distribution_by_keyword(
            df, 
            save_path=f"{OUTPUT_DIR}/{output_prefix}_by_keyword_{timestamp}.png",
            show_plot=False
        )
        
        # 2. Overall sentiment distribution
        self.create_sentiment_count_chart(
            df,
            save_path=f"{OUTPUT_DIR}/{output_prefix}_overall_{timestamp}.png", 
            show_plot=False
        )
        
        # 3. Timeline analysis (if date column exists)
        if 'created_at' in df.columns:
            self.create_timeline_analysis(
                df,
                save_path=f"{OUTPUT_DIR}/{output_prefix}_timeline_{timestamp}.png",
                show_plot=False
            )
        
        # 4. Interactive dashboard
        self.create_interactive_dashboard(
            df,
            save_path=f"{OUTPUT_DIR}/{output_prefix}_dashboard_{timestamp}.html"
        )
        
        # 5. Generate summary statistics
        stats = self.generate_summary_stats(df)
        
        # Save stats to JSON
        import json
        with open(f"{OUTPUT_DIR}/{output_prefix}_stats_{timestamp}.json", 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"All visualizations saved with timestamp: {timestamp}")
        
        return stats

def load_sentiment_results(filename: str = "tweets_with_sentiment.csv") -> pd.DataFrame:
    """Load sentiment analysis results from CSV file"""
    filepath = f"{OUTPUT_DIR}/{filename}"
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
        logger.info(f"Loaded {len(df)} tweets with sentiment data from {filepath}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        return pd.DataFrame()

def main():
    """
    Main function to test the visualizer
    """
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'keyword': ['GPT', 'GPT', 'ChatGPT', 'ChatGPT', 'Copilot', 'Copilot'] * 10,
        'sentiment_label': ['Positive', 'Negative', 'Positive', 'Neutral', 'Positive', 'Negative'] * 10,
        'sentiment_score': np.random.uniform(0.5, 1.0, 60),
        'created_at': pd.date_range('2024-01-01', periods=60, freq='D')
    })
    
    print("Testing visualization with sample data...")
    
    visualizer = SentimentVisualizer()
    
    # Test individual visualizations
    print("Creating sentiment distribution by keyword...")
    visualizer.create_sentiment_distribution_by_keyword(sample_data, show_plot=False)
    
    print("Creating overall sentiment count chart...")
    visualizer.create_sentiment_count_chart(sample_data, show_plot=False)
    
    print("Creating timeline analysis...")
    visualizer.create_timeline_analysis(sample_data, show_plot=False)
    
    print("Generating summary statistics...")
    stats = visualizer.generate_summary_stats(sample_data)
    print("Summary Stats:", stats)
    
    print("All visualizations completed successfully!")

if __name__ == "__main__":
    main()