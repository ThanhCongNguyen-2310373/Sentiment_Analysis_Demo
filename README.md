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
Stage 1: Weak Supervision (Reddit Gaming, no labels needed)
        ↓
Stage 2: Supervised Balanced (Kaggle Dataset + Undersampling)
        ↓
Stage 3a: Focal Loss (Imbalanced)  +  Stage 3b: Class Weighting (Imbalanced)
        ↓
Stage 4: Final Comparison (All Methods)
```

### **5 Giai Đoạn Thực Hiện:**

1. **Stage 1**: Weak Supervision - Reddit Gaming signals (8 signals), no manual labels
2. **Stage 2**: Supervised Balanced - Kaggle dataset + undersampling
3. **Stage 3a**: Focal Loss - Handle imbalance with α=0.25, γ=2.0
4. **Stage 3b**: Class Weighting - Alternative imbalance handling
5. **Stage 4**: Final Comparison - Comprehensive analysis

## ✨ Tính năng chính

### 🎮 Stage 1: Weak Supervision
- **8 Signals**: Awards, Comments, Upvote Ratio, Score, Text, Sarcasm, flair, top comments
- **No Labels Needed**: Auto-generate from Reddit signals
- **Fast**: 8 min training, 68.88% accuracy, 3,847 samples

### 📚 Stage 2: Supervised Balanced
- **Undersampling**: Balance classes (23K → 12K samples)
- **Mid-Range**: 10 min training, 84.67% accuracy
- **Trade-off**: Data loss for balanced performance

### 🔥 Stage 3a: Focal Loss
- **Full Dataset**: 23,189 samples (no undersampling)
- **Best Accuracy**: 86.75% accuracy, 86.84% F1-score
- **Slow**: 109 min training, α=0.25, γ=2.0

### ⚖️ Stage 3b: Class Weighting
- **Alternative**: Weighted CrossEntropy (no Focal Loss)
- **Full Dataset**: 23,189 samples
- **Compare**: vs Focal Loss performance

### 📊 Stage 4: Comparison
- **Visualizations**: Bar charts, radar plots, scatter plots
- **Trade-offs**: Speed vs Accuracy vs Data Efficiency
- **Recommendations**: Best approach for each scenario

## 📁 Cấu trúc thư mục

```
sentiment-analysis-project/
├── Notebook/                       # 🎯 5 Jupyter notebooks
│   ├── stage1_weak_supervision.ipynb              # Weak Supervision
│   ├── stage2.ipynb                               # Balanced Supervised
│   ├── stage3_supervised_focal_loss.ipynb         # Focal Loss
│   ├── stage3_supervised_class_weighting.ipynb    # Class Weighting
│   └── stage4_final_comparison.ipynb              # Final Comparison
│
├── data/                           # 📊 Dataset
│   └── 23k_r_gaming_comments_sentiments.csv
│
├── results/                        # 📈 JSON results
│   ├── stage1_results.json
│   ├── stage2_results.json        # TODO: Upload from Colab
│   ├── stage3_results.json        # Stage 3a: Focal Loss
│   └── stage3_weighted_results.json # TODO: Stage 3b training
│
├── documentation/                  # 📄 Analysis documents
│   ├── RESULTS_ANALYSIS.md       # Stage 1 vs 3a comparison
│   └── PROJECT_STATUS.md         # Project status tracker
│
├── README.md                      # 📖 This file
├── requirements.txt               # 📦 Dependencies
└── .gitignore                     # Git ignore
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

---
## � Contact

- GitHub: [@ThanhCongNguyen-2310373](https://github.com/ThanhCongNguyen-2310373)
- Project Link: [https://github.com/ThanhCongNguyen-2310373/Sentiment_Analysis_Demo](https://github.com/ThanhCongNguyen-2310373/Sentiment_Analysis_Demo)


**📅 Last Updated**: November 12, 2025  
**🔬 Research Focus**: Weak Supervision vs Supervised Learning for Gaming Sentiment Analysis  

---

*Built with ❤️ for the gaming community and NLP research*