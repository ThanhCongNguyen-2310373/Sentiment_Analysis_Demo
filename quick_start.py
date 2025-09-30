#!/usr/bin/env python3
"""
Quick Start Script for Sentiment Analysis Project
Chạy toàn bộ pipeline một cách nhanh chóng để test
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("🚀 TWITTER SENTIMENT ANALYSIS - QUICK START")
    print("=" * 60)
    print("Dự án tái hiện đồ án NLP với RoBERTa model")
    print("Kiến trúc đơn giản hóa: Python + Pandas + Transformers")
    print("=" * 60)
    
    try:
        # Import các modules chính
        from src.main import SentimentAnalysisPipeline
        
        # Tạo pipeline
        pipeline = SentimentAnalysisPipeline()
        
        # Chạy demo với dữ liệu mẫu
        print("\n🔄 Chạy demo với dữ liệu mẫu...")
        
        # Tạo keywords mẫu (giảm số lượng để demo nhanh)
        demo_keywords = ["GPT", "ChatGPT", "Copilot"]
        
        results = pipeline.run_complete_pipeline(
            keywords=demo_keywords,
            max_tweets=50  # Giảm để demo nhanh
        )
        
        if results['success']:
            print("\n" + "=" * 60)
            print("✅ DEMO THÀNH CÔNG!")
            print("=" * 60)
            print(f"Keywords analyzed: {', '.join(demo_keywords)}")
            print(f"Total tweets: {results['total_tweets_processed']}")
            print(f"Execution time: {results['execution_time']}")
            
            if results['stats']:
                print("\nSentiment Distribution:")
                for sentiment, count in results['stats']['sentiment_distribution'].items():
                    percentage = results['stats']['sentiment_percentages'][sentiment]
                    print(f"  {sentiment}: {count} ({percentage}%)")
            
            print(f"\nFiles created:")
            print(f"  - Check /data folder for raw and processed data")
            print(f"  - Check /results folder for charts and analysis")
            
        else:
            print("\n❌ Demo failed:", results['error_message'])
            print("Stages completed:", ', '.join(results['stages_completed']))
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\nCách khắc phục:")
        print("1. Đảm bảo bạn đang ở thư mục gốc của project")
        print("2. Cài đặt dependencies: pip install -r requirements.txt")
        print("3. Chạy lại script này")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\nVui lòng kiểm tra:")
        print("1. Tất cả files trong /src đã được tạo chưa")
        print("2. Dependencies đã được cài đặt chưa")
        print("3. API key (nếu sử dụng)")

if __name__ == "__main__":
    main()