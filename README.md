# Twitter Sentiment Analysis Project

## Tổng quan

Dự án này tái hiện và mở rộng nghiên cứu về phân tích cảm xúc Twitter sử dụng mô hình RoBERTa-Twitter, dựa trên đồ án gốc của anh Thịnh Lâm Tấn. Thay vì xây dựng lại toàn bộ hệ thống Big Data phức tạp (Apache Kafka, Spark, MongoDB), chúng tôi tập trung vào việc tái hiện mô hình AI và kết quả phân tích với dữ liệu mới.

## Kiến trúc Đơn giản hóa

```
Raw Data Collection → Text Preprocessing → Sentiment Analysis → Visualization
     (Twitter API)      (Text Cleaning)      (RoBERTa Model)     (Charts & Reports)
```

**So sánh với đồ án gốc:**
- **Gốc**: Producer → Kafka → Spark Streaming → MongoDB → Visualization
- **Hiện tại**: Python Script → CSV Files → RoBERTa → Charts & Reports

## Tính năng chính

- 🔍 **Thu thập dữ liệu**: Tự động crawl tweets với từ khóa AI (GPT, ChatGPT, Copilot, etc.)
- 🧹 **Tiền xử lý**: Làm sạch văn bản (loại bỏ URL, mentions, hashtags, ký tự đặc biệt)
- 🤖 **Phân tích cảm xúc**: Sử dụng mô hình RoBERTa-Twitter từ Hugging Face
- 📊 **Trực quan hóa**: Tạo biểu đồ tương tự đồ án gốc (Figures 6.4, 6.5)
- 📈 **Báo cáo**: Thống kê chi tiết và so sánh kết quả

## Cấu trúc thư mục

```
sentiment-analysis-project/
├── src/                      # Source code
│   ├── config.py            # Cấu hình chung
│   ├── crawler.py           # Module thu thập dữ liệu
│   ├── data_preprocessing.py # Module tiền xử lý
│   ├── sentiment_analysis.py # Module phân tích cảm xúc
│   ├── visualization.py     # Module trực quan hóa
│   ├── main.py             # Script chính
│   └── __init__.py         # Package initialization
├── data/                    # Dữ liệu thô và đã xử lý
├── results/                 # Kết quả phân tích và biểu đồ
├── notebooks/              # Jupyter notebooks demo
├── requirements.txt        # Danh sách thư viện
├── .env.example           # Template cho biến môi trường
└── README.md              # Tài liệu này
```

## Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd sentiment-analysis-project
```

### 2. Tạo môi trường ảo (khuyến nghị)
```bash
python -m venv venv
# Windows
venv\\Scripts\\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 4. Cấu hình API Key
```bash
# Sao chép template
copy .env.example .env

# Chỉnh sửa file .env và thêm Twitter API key
TWITTER_API_KEY=your_api_key_here
```

## Sử dụng

### Chạy pipeline hoàn chỉnh
```bash
cd src
python main.py --mode full --keywords GPT ChatGPT Copilot --max-tweets 500
```

### Phân tích dữ liệu có sẵn
```bash
python main.py --mode analyze --data-file ../data/tweets_processed_20240101_120000.csv
```

### Sử dụng từng module riêng lẻ

#### 1. Thu thập dữ liệu
```python
from src.crawler import TwitterCrawler

crawler = TwitterCrawler()
df = crawler.crawl_multiple_keywords(['GPT', 'ChatGPT'], max_tweets_per_keyword=100)
crawler.save_raw_data(df, 'my_tweets.csv')
```

#### 2. Tiền xử lý
```python
from src.data_preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()
processed_df = preprocessor.preprocess_dataframe(df)
```

#### 3. Phân tích cảm xúc
```python
from src.sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result_df = analyzer.analyze_dataframe(processed_df)
```

#### 4. Trực quan hóa
```python
from src.visualization import SentimentVisualizer

visualizer = SentimentVisualizer()
visualizer.create_all_visualizations(result_df)
```

## Mô hình và Phương pháp

### Mô hình RoBERTa-Twitter
- **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Nguồn**: Hugging Face Transformers
- **Đặc điểm**: Được tinh chỉnh chuyên biệt cho dữ liệu Twitter
- **Output**: 3 lớp (Positive, Negative, Neutral) với confidence scores

### Pipeline Tiền xử lý (theo đồ án gốc)
1. **Chuyển chữ thường**: Chuẩn hóa text
2. **Loại bỏ URL**: Xóa các liên kết web
3. **Loại bỏ mentions**: Xóa @username
4. **Loại bỏ hashtags**: Xóa #hashtag
5. **Loại bỏ ký tự đặc biệt**: Chỉ giữ chữ cái và số
6. **Loại bỏ khoảng trắng thừa**: Chuẩn hóa spaces

### Từ khóa thu thập
```python
KEYWORDS = [
    "GPT", "Copilot", "Gemini",     # Từ đồ án gốc
    "GPT-4o", "Sora", "Llama 3",    # Từ khóa mới
    "Claude", "ChatGPT"             # Bổ sung thêm
]
```

## Kết quả và Biểu đồ

Hệ thống tự động tạo các biểu đồ sau (tương tự đồ án gốc):

1. **Sentiment Distribution by Keyword**: Biểu đồ cột chồng thể hiện tỷ lệ cảm xúc theo từ khóa
2. **Overall Sentiment Distribution**: Biểu đồ tổng quan phân bố cảm xúc
3. **Timeline Analysis**: Xu hướng cảm xúc theo thời gian
4. **Interactive Dashboard**: Dashboard tương tác bằng Plotly

## So sánh với Đồ án Gốc

| Aspect | Đồ án Gốc | Dự án Hiện tại |
|--------|-----------|----------------|
| **Kiến trúc** | Big Data (Kafka + Spark + MongoDB) | Simple Python Pipeline |
| **Mô hình** | RoBERTa-Twitter + Logistic Regression | RoBERTa-Twitter |
| **Thu thập dữ liệu** | Real-time streaming | Batch collection |
| **Xử lý** | Spark Streaming + MLlib | Pandas + Transformers |
| **Lưu trữ** | MongoDB | CSV Files |
| **Triển khai** | Distributed cluster | Single machine |
| **Phức tạp** | High | Low |
| **Khả năng mở rộng** | High | Medium |

## Kết quả Dự kiến

Dựa trên đồ án gốc, chúng ta dự kiến:
- **Accuracy**: ~82% (tương đương kết quả gốc)
- **Processing time**: Nhanh hơn do không có overhead của distributed system
- **Resource usage**: Thấp hơn đáng kể

## Troubleshooting

### Lỗi thường gặp

1. **ImportError: transformers**
   ```bash
   pip install transformers torch
   ```

2. **API Key không hợp lệ**
   - Kiểm tra file `.env`
   - Đảm bảo API key từ twitterapi.io đúng format

3. **CUDA out of memory**
   ```python
   # Trong config.py, giảm BATCH_SIZE
   BATCH_SIZE = 16  # từ 32 xuống 16
   ```

4. **Rate limit exceeded**
   ```python
   # Tăng RATE_LIMIT_DELAY trong config.py
   RATE_LIMIT_DELAY = 5  # từ 2s lên 5s
   ```

## Development

### Chạy tests
```bash
cd src
python -m pytest tests/  # (nếu có)
```

### Test từng module
```bash
python config.py      # Test configuration
python crawler.py     # Test data collection
python data_preprocessing.py  # Test preprocessing
python sentiment_analysis.py  # Test sentiment model
python visualization.py      # Test charts
```

## Kế hoạch tương lai

- [ ] Thêm mô hình Logistic Regression để so sánh
- [ ] Hỗ trợ nhiều ngôn ngữ
- [ ] Real-time dashboard với Streamlit
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure)
- [ ] A/B testing framework

## Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Tạo Pull Request

## License

MIT License - xem file LICENSE để biết chi tiết

## Tác giả

- **Tên của bạn** - *Initial work* - [GitHub](https://github.com/yourusername)
- **Tham khảo**: Đồ án của Thịnh Lâm Tấn - Twitter Sentiment Analysis using Big Data

## Acknowledgments

- Cảm ơn anh Thịnh Lâm Tấn vì đồ án tham khảo xuất sắc
- Hugging Face team vì mô hình RoBERTa-Twitter
- Cardiff NLP team vì pre-trained model
- twitterapi.io vì API service

---

*Dự án này được thực hiện như một phần của khóa học NLP, tái hiện và mở rộng nghiên cứu về sentiment analysis với approach đơn giản hóa nhưng vẫn đảm bảo chất lượng kết quả.*