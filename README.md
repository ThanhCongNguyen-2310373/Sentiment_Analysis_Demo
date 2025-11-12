# 🎮 Gaming Sentiment Analysis Project

## 📋 Tổng quan

Dự án nghiên cứu và so sánh các phương pháp **Sentiment Analysis** cho Gaming Domain, bao gồm **Weak Supervision** và **Supervised Learning** với các kỹ thuật xử lý class imbalance. Dự án được thực hiện qua **4 stages** để so sánh hiệu quả của các phương pháp khác nhau.

### 🎯 Mục tiêu
- So sánh **Weak Supervision** (Reddit Gaming) vs **Supervised Learning** (Kaggle Dataset)
- Nghiên cứu hiệu quả của **Focal Loss** trong xử lý class imbalance
- Áp dụng **Gaming-specific features** và **Community signals** vào sentiment analysis
- Đánh giá trade-off giữa **manual labeling cost** vs **model accuracy**

## 🏗️ Kiến trúc 4-Stage Pipeline

```
Stage 1: Weak Supervision          Stage 2: Supervised (Balanced)
  (Reddit Gaming)         →          (Kaggle Dataset + DistilBERT)
        ↓                                      ↓
Stage 3: Focal Loss                Stage 4: Final Comparison
  (Imbalanced Kaggle)      →         (All Methods Analysis)
```

### **4 Giai Đoạn Thực Hiện:**

1. **Stage 1**: Weak Supervision với Reddit Gaming signals (6-8 signals)
2. **Stage 2**: Supervised Learning với balanced Kaggle dataset
3. **Stage 3**: Supervised Learning + Focal Loss (class imbalance)
4. **Stage 4**: So sánh và đánh giá tất cả các phương pháp

## ✨ Tính năng chính

### 🎮 Stage 1: Weak Supervision (Reddit Gaming)
- **6-8 Signals Approach**: Awards, Comments, Upvote Ratio, Score, Text Features, Sarcasm, Flair, Top Comments
- **No Manual Labeling**: Tự động tạo labels từ Reddit community signals
- **Gaming-Specific**: Vocabulary và patterns cho gaming domain
- **Fast Training**: 8 phút training time
- **Results**: 68.88% accuracy với 3,847 gaming posts

### 📚 Stage 3: Supervised Learning (Focal Loss)
- **Focal Loss**: Xử lý class imbalance (alpha=0.25, gamma=2.0)
- **Large Dataset**: 23,189 pre-labeled Kaggle game reviews
- **High Accuracy**: 86.75% accuracy, 86.84% F1-score
- **Class Balance**: Handle 2.46:1 imbalance ratio
- **Longer Training**: 109 phút với larger dataset

### 📊 Phân tích & So sánh
- **Comprehensive Metrics**: Accuracy, F1-Score, Training Time, Dataset Size
- **Trade-off Analysis**: Manual labeling cost vs model performance
- **Hybrid Approach**: Kết hợp weak supervision + supervised learning
- **Detailed Reports**: JSON results + markdown analysis

## 📁 Cấu trúc thư mục

```
sentiment-analysis-project/
├── Notebook/                       # 🎯 Jupyter notebooks cho 4 stages
│   ├── stage1_weak_supervision.ipynb       # Stage 1: Reddit Gaming Weak Supervision
│   ├── stage2.ipynb                        # Stage 2: Balanced Supervised Learning
│   ├── stage3_supervised_focal_loss.ipynb  # Stage 3a: Focal Loss (Imbalanced)
│   ├── stage3_supervised_class_weighting.ipynb  # Stage 3b: Class Weighting (Imbalanced)
│   └── stage4_final_comparison.ipynb       # Stage 4: Compare all methods
│
├── data/                           # 📊 Datasets
│   └── 23k_r_gaming_comments_sentiments.csv   # Kaggle Game Reviews (23K)
│
├── results/                        # 📈 Training results
│   ├── stage1_results.json        # Stage 1 metrics (Weak Supervision)
│   ├── stage2_results.json        # Stage 2 metrics (Balanced Supervised) - TODO
│   ├── stage3_focal_results.json  # Stage 3a metrics (Focal Loss)
│   ├── stage3_weighted_results.json # Stage 3b metrics (Class Weighting) - TODO
│   └── [model_checkpoints]/       # Saved models
│
├── RESULTS_ANALYSIS.md            # 📊 Comprehensive results analysis
├── README.md                      # 📖 This file
├── requirements.txt               # 📦 Python dependencies
└── .gitignore                     # Git ignore patterns
```

### 📦 Core Dependencies
```
# Deep Learning & NLP
torch>=2.0.0              # PyTorch framework
transformers>=4.30.0      # Hugging Face models (RoBERTa, DistilBERT)

# Data Processing
pandas>=1.5.0             # DataFrame manipulation
numpy>=1.24.0             # Numerical operations
scikit-learn>=1.3.0       # ML utilities, metrics

# Reddit API
praw>=7.7.0               # Reddit API wrapper
python-dotenv>=1.0.0      # Environment variables

# Visualization
matplotlib>=3.7.0         # Basic plotting
seaborn>=0.12.0          # Statistical visualization
plotly>=5.15.0           # Interactive dashboards

# Jupyter Notebooks
jupyter>=1.0.0           # Notebook interface
ipykernel>=6.25.0        # Jupyter kernel
tqdm>=4.65.0             # Progress bars
```

## 🚀 Quick Start

### Option 1: Google Colab (Khuyến nghị - Có GPU miễn phí)

```python
# Upload notebooks lên Google Drive
# Mở notebook trong Colab và chạy:

!pip install transformers torch praw scikit-learn

# Chạy từng stage theo thứ tự:
# Stage 1 → Stage 2 → Stage 3 → Stage 4
```

### Option 2: Local Setup

#### 1. Clone repository
```bash
git clone https://github.com/ThanhCongNguyen-2310373/Sentiment_Analysis_Demo.git
cd Sentiment_Analysis_Demo
```

#### 2. Tạo môi trường ảo
```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Cài đặt dependencies
```bash
# One-command install
pip install -r requirements.txt

# Hoặc cài đặt từng group
pip install torch transformers scikit-learn
pip install pandas numpy matplotlib seaborn plotly
pip install praw python-dotenv jupyter tqdm
```

#### 4. Cấu hình Reddit API (Chỉ cho Stage 1)
```bash
# Copy template
cp .env.template .env

# Edit .env và thêm Reddit credentials
# Get from: https://www.reddit.com/prefs/apps
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent
```

#### 5. Khởi động Jupyter
```bash
jupyter notebook

# Mở notebooks/ và chạy theo thứ tự:
# stage1_weak_supervision.ipynb → stage2 → stage3 → stage4
```

## 🔄 Future Improvements

### 🎯 For Stage 1 (Weak Supervision)
- [ ] Implement 8-signal approach (currently 6 signals)
- [ ] Add flair analysis và top comments signals
- [ ] Increase dataset to 500 posts/subreddit
- [ ] Enhanced gaming vocabulary với N-grams
- [ ] Dynamic weighting system
- [ ] Better class balancing

### 🎯 For Stage 2 (Balanced Supervised)
- [ ] Implement và train DistilBERT model
- [ ] Compare với RoBERTa performance
- [ ] Test different balancing techniques
- [ ] Hyperparameter tuning

### 🎯 For Stage 3 (Focal Loss)
- [ ] Tune Focal Loss parameters (alpha, gamma)
- [ ] Try different loss functions (Label Smoothing, etc.)
- [ ] Address class imbalance further
- [ ] Combine với gaming-specific features from Stage 1

### 🚀 Advanced Features
- [ ] Hybrid approach: Stage 1 → Human verify → Stage 3
- [ ] Active learning for sample selection
- [ ] Ensemble methods (combine all stages)
- [ ] Real-time sentiment dashboard
- [ ] API deployment (FastAPI)
- [ ] Docker containerization
- [ ] Multi-language support
- [ ] A/B testing framework


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## � Contact

- GitHub: [@ThanhCongNguyen-2310373](https://github.com/ThanhCongNguyen-2310373)
- Project Link: [https://github.com/ThanhCongNguyen-2310373/Sentiment_Analysis_Demo](https://github.com/ThanhCongNguyen-2310373/Sentiment_Analysis_Demo)

## ⭐ Star History

If you find this project useful, please consider giving it a star ⭐

---

**📅 Last Updated**: November 12, 2025  
**🔬 Research Focus**: Weak Supervision vs Supervised Learning for Gaming Sentiment Analysis  

---

*Built with ❤️ for the gaming community and NLP research*