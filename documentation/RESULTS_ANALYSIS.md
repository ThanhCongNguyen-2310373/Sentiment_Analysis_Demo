# 📊 Results Analysis: Stage 1 vs Stage 3a

## 🎯 Quick Comparison

| Metric | Stage 1: Weak | Stage 3a: Focal Loss | Δ |
|--------|---------------|----------------------|---|
| **Accuracy** | 68.88% | **86.75%** | +17.87% ⬆️ |
| **F1-Score** | 68.70% | **86.84%** | +18.14% ⬆️ |
| **Dataset** | 3,847 | 23,189 | +503% |
| **Training Time** | 8 min | 109 min | +13.6x |
| **Manual Labels** | ❌ No | ✅ Yes | - |

---

## 📈 Stage 1: Weak Supervision (Reddit Gaming)

### **🎮 Đặc Điểm:**
- **Phương pháp**: Weak Supervision với 8 signals
- **Dataset**: Reddit Gaming posts
- **Kích thước**: 3,847 posts (1,542 train / 331 val / 331 test)
- **Model Base**: `cardiffnlp/twitter-roberta-base-sentiment-latest`

### **🔍 Weak Labeling Strategy:**

**6 Signals Được Sử Dụng:**
1. **Awards** - Số lượng awards (weight: 4.0)
2. **Comments** - Engagement của cộng đồng (weight: 3.0)
3. **Upvote Ratio** - Tỷ lệ upvote (weight: 2.5)
4. **Score** - Điểm tổng thể (weight: 2.0)
5. **Gaming Text Features** - Phân tích văn bản gaming-specific (weight: 1.8)
6. **Sarcasm Detection** - Phát hiện châm biếm (flip sentiment)
7. **flair** - Post category indicators
8. **top comments** - Community response sentiment

**Chất Lượng Weak Labels:**
- ✅ Labeled samples: 2,204 / 3,847 (57.3%)
- ✅ Average confidence: **78.7%**
- ✅ Min confidence threshold: 60%

### **📊 Kết Quả Performance:**
- **Accuracy**: **68.88%**
- **F1-Score**: **68.70%**
- **Training Time**: 477 seconds (~8 phút)

### **🎯 Ưu Điểm:**
✅ **Không cần manual labeling** - Tự động tạo labels từ Reddit signals  
✅ **Gaming-domain specific** - Tận dụng gaming subreddits và keywords  
✅ **Fast training** - Chỉ 8 phút training time  
✅ **High confidence labels** - 78.7% average confidence  
✅ **Community signals** - Tận dụng upvotes, awards, comments từ cộng đồng gaming  

### **⚠️ Nhược Điểm:**
❌ **Accuracy thấp hơn** - 68.88% so với 86.75% của Stage 3  
❌ **Dataset nhỏ** - Chỉ 3,847 samples vs 23,189 của Stage 3  
❌ **Noisy labels** - Weak supervision có thể tạo ra labels không chính xác  
❌ **Limited by Reddit signals** - Phụ thuộc vào quality của Reddit community voting  

---

## 🔥 Stage 3a: Supervised + Focal Loss

### **Approach:**
- **Method**: Focal Loss (α=0.25, γ=2.0)
- **Dataset**: 23,189 Kaggle reviews (full imbalanced)
- **Imbalance**: 2.46:1 ratio (positive:negative)

### **Results:**
- **Accuracy**: 86.75%
- **F1-Score**: 86.84%
- **Training**: 109 min

### **🎯 Ưu Điểm:**
✅ **High accuracy** - 86.75% accuracy (cao hơn 17.87% so với Stage 1)  
✅ **Large dataset** - 23,189 pre-labeled samples  
✅ **Clean labels** - Human-labeled Kaggle dataset  
✅ **Focal Loss handles imbalance** - Tối ưu cho imbalanced classes  
✅ **Better generalization** - F1-score 86.84% cho thấy balanced performance  

### **⚠️ Nhược Điểm:**
❌ **Requires manual labels** - Cần pre-labeled dataset  
❌ **Long training time** - 109 phút (13.7x chậm hơn Stage 1)  
❌ **Not domain-specific** - General game reviews, không focus gaming community  
❌ **Class imbalance** - Negative class chỉ chiếm 17.6%  

---


## 🚀 Next Steps

### **Stage 3a/3b Comparison:**
- Compare Focal Loss vs Class Weighting
- Tune α, γ parameters
- Combine with Stage 1 gaming features

### **Stage 4 Final Analysis:**
- Include Stage 2 (balanced) results
- Include Stage 3b (class weighting) results
- Comprehensive 5-way comparison

---

**📅 Updated**: November 12, 2025  
**� Data**: stage1_results.json & stage3_results.json (Stage 3a)
