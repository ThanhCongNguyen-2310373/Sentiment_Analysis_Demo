# 🎮 Gaming Sentiment Analysis Project

## 🎥 Video Thuyết Trình

📺 **[Xem video thuyết trình đồ án tại đây](https://drive.google.com/file/d/10ohP2BdB1QMj40pevKfli-U8YlF-Ya2V/view?usp=sharing)**

---

## 📋 Tổng quan

Dự án nghiên cứu và so sánh các phương pháp **Sentiment Analysis** cho Gaming Domain, bao gồm **Weak Supervision** và **Supervised Learning** với các kỹ thuật xử lý class imbalance. Dự án được thực hiện qua **4 stages** để so sánh hiệu quả của các phương pháp khác nhau.

### 🎯 Mục tiêu
- So sánh **Weak Supervision** (Reddit Gaming) vs **Supervised Learning** (Kaggle Dataset)
- Nghiên cứu hiệu quả của **Focal Loss** và **Class Weighting** trong xử lý class imbalance
- Áp dụng **Gaming-specific features** và **Community signals** vào sentiment analysis
- Đánh giá trade-off giữa **manual labeling cost** vs **model accuracy**

## 📊 Kết quả Chính (All Stages Complete ✅)

| Stage | Method | Dataset | Samples | Accuracy | F1-Score | Training Time |
|-------|--------|---------|---------|----------|----------|---------------|
| **1** | Weak Supervision (8-Signal) | Reddit | 2,204 | **86.70%** | 0.866 | 19.16 min |
| **1-X** | Stage 1 Cross-Eval | Kaggle | 23,189 | **47.88%** | 0.440 | - |
| **2** | Supervised (Balanced) | Kaggle | 12,216 | **79.44%** | 0.794 | 51.45 min |
| **3a** | Supervised (Focal Loss) ⭐ | Kaggle | 23,189 | **82.35%** | 0.823 | 35.86 min |
| **3b** | Supervised (Class Weight) | Kaggle | 23,189 | **82.22%** | 0.822 | 86.17 min |

### 🔑 Key Findings

✅ **Stage 3a (Focal Loss) is BEST for Kaggle Dataset**:
- Highest accuracy: **82.35%** (+2.91% vs Stage 2 Balanced)
- Best F1-Score: **0.823**
- Uses **full dataset** (no data loss from undersampling)
- **Fastest training** among supervised methods (35.86 min)
- Optimal for production deployment

⚠️ **Weak Supervision (Stage 1) Shows Severe Generalization Gap**:
- Excellent on **Reddit domain**: 86.70% accuracy
- **Fails on Kaggle domain**: Only 47.88% accuracy (38.82% drop!)
- Conclusion: Domain-specific signals don't transfer well
- Useful for **rapid prototyping** but needs domain-matched data

📊 **Stage 3b (Class Weighting) vs Stage 3a (Focal Loss)**:
- Class Weighting: 82.22% accuracy, **86.17 min** training
- Focal Loss: 82.35% accuracy, **35.86 min** training (2.4× faster)
- Focal Loss achieves **0.13% higher accuracy** while being significantly faster
- Focal Loss outperforms by +0.46% with significantly faster training
- Recommendation: **Use Focal Loss** unless hardware constraints exist

🎯 **Stage 2 (Balancing) Trade-offs**:
- Loses **47% of data** (23,189 → 12,216 samples)
- Result: 79.44% accuracy (2.91% below Focal Loss)
- Conclusion: Data loss hurts performance; **avoid undersampling** when possible

## 🏗️ Kiến trúc 4-Stage Pipeline

```
Stage 1: Weak Supervision (Reddit Gaming, no labels)
        ↓
Stage 1-X: Cross-Evaluation (Stage 1 model → Kaggle test)
        ↓
Stage 2: Supervised Balanced (Kaggle Dataset + Undersampling)
        ↓
Stage 3a: Focal Loss (Full Imbalanced)  +  Stage 3b: Class Weighting (Full Imbalanced)
        ↓
Stage 4: Final Comparison (All Methods)
```
---
## 📁 Cấu trúc thư mục

```
sentiment-analysis-project/
├── Notebook/                       # 🎯 5 Jupyter notebooks (All Stages Complete ✅)
│   ├── stage1_weak_supervision.ipynb           # Stage 1: Weak Supervision (86.70% Reddit)
│   ├── stage2.ipynb                            # Stage 2: Balanced (79.44%)
│   ├── stage3a_supervised_focal_loss.ipynb     # Stage 3a: Focal Loss (82.35% BEST)
│   ├── Stage3b_supervised_class_weight.ipynb   # Stage 3b: Class Weighting (82.22%)
│   └── stage4_final_comparison.ipynb           # Stage 4: Final Comparison
│
├── data/                           # 📊 Dataset
│   ├── 23k_r_gaming_comments_sentiments.csv    # Kaggle dataset
│   └── cleaned_comments.csv                     # Stage 2,3 data
│
├── results/                        # 📈 JSON results (All Complete ✅)
│   ├── stage1_results.json                      # 86.70% (Reddit)
│   ├── stage1_cross_eval_results.json           # 47.88% (Kaggle cross-eval)
│   ├── stage2_results.json                      # 79.44% (Balanced)
│   ├── stage3a_results.json                     # 82.35% (Focal Loss)
│   ├── stage3b_results.json                     # 82.22% (Class Weighting)
│   └── stage4_analysis_summary.json             # Stage 4 summary
│   
│
├── documentation/                  # 📄 Analysis documents
│   ├── RESULTS_ANALYSIS.md                      # Detailed all stage analysis
│   └── PROJECT_STATUS.md                        # Project completion status
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


**📅 Last Updated**: December 3, 2025  
**🔬 Research Focus**: Weak Supervision vs Supervised Learning for Gaming Sentiment Analysis

---

*Built with ❤️ for the gaming community and NLP research*