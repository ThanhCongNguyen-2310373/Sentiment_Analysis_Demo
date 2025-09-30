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

## Yêu cầu hệ thống

### System Requirements
- **Python**: 3.8+ (Đã test trên Python 3.13.7)
- **RAM**: Tối thiểu 4GB, khuyến nghị 8GB+
- **Disk**: 2GB free space (cho models và data)
- **OS**: Windows 10/11, macOS, Linux
- **Internet**: Cần để download models và API calls

### Package Dependencies
```
pandas>=1.5.0          # Data manipulation
numpy>=1.24.0           # Numerical computing  
torch>=2.0.0            # Deep learning framework
transformers>=4.30.0    # Hugging Face models
matplotlib>=3.7.0       # Basic plotting
seaborn>=0.12.0         # Statistical visualization
plotly>=5.15.0          # Interactive charts
requests>=2.31.0        # HTTP requests
python-dotenv>=1.0.0    # Environment variables
nltk>=3.8.0             # Natural language toolkit
jupyter>=1.0.0          # Notebook interface (optional)
```

## Cài đặt

### 1. Clone repository
```bash
git clone https://github.com/ThanhCongNguyen-2310373/Sentiment_Analysis_Demo.git
cd Sentiment_Analysis_Demo
```

### 2. Tạo môi trường ảo (khuyến nghị)
```bash
# Tạo virtual environment
python -m venv .venv

# Kích hoạt environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate
```

### 3. Cài đặt các package cần thiết

#### Phương pháp 1: Cài đặt từ requirements.txt (Khuyến nghị)
```bash
pip install -r requirements.txt
```

#### Phương pháp 2: Cài đặt từng package (Chi tiết)
```bash
# Core packages cho data processing
pip install pandas>=1.5.0
pip install numpy>=1.24.0

# Machine Learning và NLP
pip install torch>=2.0.0
pip install transformers>=4.30.0

# Visualization
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install plotly>=5.15.0

# API và networking
pip install requests>=2.31.0

# Environment management
pip install python-dotenv>=1.0.0

# Natural Language Processing utilities
pip install nltk>=3.8.0

# Optional: Jupyter for notebooks
pip install jupyter>=1.0.0
pip install ipykernel>=6.25.0
```

#### Phương pháp 3: Cài đặt với upgrade (Nếu gặp conflict)
```bash
pip install --upgrade pip
pip install --upgrade pandas numpy torch transformers matplotlib seaborn plotly requests python-dotenv nltk
```

### 4. Cấu hình API Key
```bash
# Sao chép template
copy .env.template .env

# Chỉnh sửa file .env và thêm Twitter API key từ twitterapi.io
TWITTER_API_KEY=your_api_key_here
TWITTER_BEARER_TOKEN=your_bearer_token_here
```

### 5. Kiểm tra cài đặt
```bash
# Kiểm tra Python environment
python --version  # Đảm bảo >= 3.8

# Test import các package chính
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"

# Chạy quick test
python quick_start.py
```

### 6. Download NLTK data (Tự động khi chạy lần đầu)
```python
# Sẽ tự động download khi chạy, hoặc chạy manual:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Sử dụng

### Cách nhanh nhất: Chạy demo
```bash
# Chạy demo với sample data
python quick_start.py
```

### Chạy pipeline hoàn chỉnh
```bash
# Chạy phân tích hoàn chỉnh
python src/main.py

# Hoặc chạy với custom settings
cd src
python main.py --keywords "GPT,ChatGPT,Copilot" --max-tweets 100
```

### Sử dụng Jupyter Notebook (Tương tác)
```bash
# Khởi động Jupyter
jupyter notebook

# Mở file: notebooks/twitter_sentiment_analysis_demo.ipynb
# Chạy từng cell để xem demo step-by-step
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

1. **ImportError: No module named 'transformers'**
   ```bash
   # Đảm bảo đang trong virtual environment
   .venv\Scripts\Activate.ps1
   pip install transformers torch
   ```

2. **ModuleNotFoundError: No module named 'torch'**
   ```bash
   # Cài đặt PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   # Hoặc với GPU support:
   # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **API Key không hợp lệ**
   - Kiểm tra file `.env` có tồn tại
   - Đảm bảo API key từ twitterapi.io đúng format
   - Kiểm tra quyền của API key

4. **CUDA out of memory (nếu dùng GPU)**
   ```python
   # Trong config.py, giảm BATCH_SIZE
   BATCH_SIZE = 8  # từ 32 xuống 8
   ```

5. **Rate limit exceeded**
   ```python
   # Tăng RATE_LIMIT_DELAY trong config.py
   RATE_LIMIT_DELAY = 5  # từ 2s lên 5s
   ```

6. **SSL Certificate errors**
   ```bash
   # Nếu gặp lỗi SSL khi download model
   pip install --upgrade certifi
   # Hoặc set environment variable
   set CURL_CA_BUNDLE=""
   ```

7. **Permission denied khi tạo virtual environment**
   ```bash
   # Chạy PowerShell as Administrator
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
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

- **ThanhCongNguyen-2310373** - *Main Developer* - [GitHub](https://github.com/ThanhCongNguyen-2310373)
- **Tham khảo**: Đồ án của Thịnh Lâm Tấn - Twitter Sentiment Analysis using Big Data

## Demo và Kết quả

### 📊 Sample Results (Demo Data)
- **Tổng tweets phân tích**: 75
- **Sentiment Distribution**:
  - 🟢 Positive: 80% (60 tweets)
  - 🔴 Negative: 13.33% (10 tweets)
  - ⚪ Neutral: 6.67% (5 tweets)
- **Average Sentiment Score**: 0.818/1.0
- **Processing Time**: ~22 seconds
- **Model Used**: cardiffnlp/twitter-roberta-base-sentiment-latest

### 📈 Generated Outputs
- **Visualization Charts**: PNG format (overall, by-keyword, timeline)
- **Interactive Dashboard**: HTML với Plotly
- **Processed Data**: CSV với sentiment scores
- **Statistics**: JSON summary report

## Acknowledgments

- Cảm ơn anh Thịnh Lâm Tấn vì đồ án tham khảo xuất sắc
- Hugging Face team vì mô hình RoBERTa-Twitter
- Cardiff NLP team vì pre-trained model
- twitterapi.io vì API service

---

*Dự án này được thực hiện như một phần của khóa học NLP, tái hiện và mở rộng nghiên cứu về sentiment analysis với approach đơn giản hóa nhưng vẫn đảm bảo chất lượng kết quả.*