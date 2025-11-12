# 📋 Quick Reference - Gaming Sentiment Analysis

## 🎯 4-Stage Overview

```
Stage 1: Weak Supervision       →  Stage 2: Balanced Supervised
  68.88% | 8 min | No labels       84.67% | 10 min | Labeled
           ↓                                  ↓
Stage 3a: Focal Loss           →  Stage 3b: Class Weighting  
  86.75% | 109 min | Labeled       TBD | ~110 min | Labeled
```

---

## 📊 Results at a Glance

| Stage | Accuracy | Time | Dataset | Labels Needed | Best For |
|-------|----------|------|---------|---------------|----------|
| **1** | 68.88% | 8 min | 3,847 | ❌ No | Fast prototyping |
| **2** | 84.67% | 10 min | 12,216 | ✅ Yes | Balanced performance |
| **3a** | 86.75% | 109 min | 23,189 | ✅ Yes | Production (best accuracy) |
| **3b** | TBD | ~110 min | 23,189 | ✅ Yes | Alternative to Focal Loss |

---

## ⚡ When to Use Which Stage?

### 🚀 **Stage 1** - Use When:
- No labeled data
- Need results fast (< 10 min)
- Gaming community focus
- Prototyping/POC
- Budget constraints

### ⚖️ **Stage 2** - Use When:
- Need balanced class performance
- Fast training important
- Good accuracy sufficient (84-85%)
- Learning/experimentation
- Limited GPU

### 🎯 **Stage 3a (Focal Loss)** - Use When:
- Highest accuracy critical
- Imbalanced dataset
- Production deployment
- Have GPU & time
- Hard examples important

### 🔧 **Stage 3b (Class Weighting)** - Use When:
- Alternative to Focal Loss
- Simpler implementation
- Compare different approaches
- Imbalanced classes

---

## 💰 Cost-Benefit Analysis

**Accuracy Gain vs Training Time**:
```
Stage 1 → 2:  +15.79% accuracy for +2 min  (7.9%/min)  ⭐ Best value
Stage 2 → 3:  +2.08% accuracy for +99 min  (0.02%/min)
Stage 1 → 3:  +17.87% accuracy for +101 min (0.18%/min)
```

**Verdict**: Stage 2 offers **best cost-benefit ratio**

---

## 🎬 Quick Start Commands

### Run on Google Colab:
```python
# 1. Upload notebook to Colab
# 2. Upload dataset (if needed)
# 3. Run all cells
# 4. Download results JSON
```

### Run Locally:
```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook

# Open Notebook/ and run stage notebooks
```

---

## 📁 Files You Need

### **For Training**:
- ✅ `Notebook/stage1_weak_supervision.ipynb` - Complete
- ✅ `Notebook/stage2.ipynb` - Complete
- ✅ `Notebook/stage3_supervised_focal_loss.ipynb` - Complete
- 🔄 `Notebook/stage3_supervised_class_weighting.ipynb` - In Progress
- ✅ `Notebook/stage4_final_comparison.ipynb` - Ready

### **For Analysis**:
- ✅ `results/stage1_results.json` - Available
- ⚠️ `results/stage2_results.json` - **Need upload**
- ✅ `results/stage3_focal_results.json` - Available (rename from stage3_results.json)
- ⏳ `results/stage3_weighted_results.json` - Pending

### **Documentation**:
- ✅ `README.md` - Main documentation
- ✅ `RESULTS_ANALYSIS.md` - Stage 1 vs 3 comparison
- ✅ `results/stage2_analysis.md` - Stage 2 detailed analysis
- ✅ `PROJECT_STATUS.md` - Current status

---

## 🔥 Hot Tips

### **Tip 1: Start with Stage 2**
If you're new, start with Stage 2:
- Fast training (10 min)
- Good accuracy (84.67%)
- Clean implementation
- Easy to understand

### **Tip 2: GPU Matters**
Training times with GPU vs CPU:
- Stage 1: 8 min (GPU) vs 30 min (CPU)
- Stage 2: 10 min (GPU) vs 45 min (CPU)  
- Stage 3: 109 min (GPU) vs 6+ hours (CPU)

**→ Use Google Colab free GPU!**

### **Tip 3: Class Balance**
Check your dataset:
- **Balanced** (equal classes) → Stage 2
- **Imbalanced** (unequal classes) → Stage 3a or 3b
- **Unknown** → Try Stage 2 first

### **Tip 4: Batch Size**
If out of memory:
```python
# Reduce batch size
per_device_train_batch_size=8  # from 16
per_device_eval_batch_size=16  # from 32
```

### **Tip 5: Early Stopping**
Save time with early stopping:
```python
# Add to TrainingArguments
load_best_model_at_end=True
metric_for_best_model="eval_loss"
```

---

## 🐛 Common Issues & Fixes

### **Issue 1: CUDA Out of Memory**
```python
# Solution: Reduce batch size
training_args = TrainingArguments(
    per_device_train_batch_size=8,  # Reduce from 16
)
```

### **Issue 2: Reddit API Rate Limit (Stage 1)**
```python
# Solution: Add delays
import time
time.sleep(2)  # Between requests
```

### **Issue 3: File Upload in Colab**
```python
# Solution: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Then use Drive paths
df = pd.read_csv('/content/drive/MyDrive/data.csv')
```

### **Issue 4: Model Download Slow**
```python
# Solution: Use cache
import os
os.environ['TRANSFORMERS_CACHE'] = '/content/drive/MyDrive/transformers_cache'
```

### **Issue 5: Long Training Time**
```python
# Solution: Reduce dataset or epochs
df_train = df_train.sample(n=10000)  # Sample data
num_train_epochs=2  # Reduce from 3
```

---

## 📈 Expected Results Checklist

### **After Each Stage**:
- [ ] Accuracy > 60% (minimum viable)
- [ ] F1-Score close to Accuracy (balanced)
- [ ] Loss decreasing over epochs
- [ ] No overfitting (train vs val loss)
- [ ] Results JSON file generated
- [ ] Confusion matrix looks reasonable

### **Red Flags**:
- ⚠️ Accuracy < 50% (random guess level)
- ⚠️ F1-Score << Accuracy (imbalanced issues)
- ⚠️ Loss increasing (learning rate too high)
- ⚠️ Train accuracy >> Val accuracy (overfitting)

---

## 🎯 Next Actions

### **If You Have**:

**✅ All 3 Results (Stage 1, 2, 3a)**:
→ Upload Stage 2 JSON → Run Stage 4 comparison

**🔄 Only Stage 1 & 3a**:
→ Run Stage 2 on Colab → Upload results → Stage 4

**⚡ Just Starting**:
→ Run Stage 2 first (fastest good results)

**🚀 Want Best Results**:
→ Skip to Stage 3a (Focal Loss)

**🔬 Research Mode**:
→ Run all 4 stages → Complete comparison

---

## 📞 Need Help?

**Check**:
1. `README.md` - Full documentation
2. `PROJECT_STATUS.md` - Current progress
3. `RESULTS_ANALYSIS.md` - Detailed analysis
4. Notebook comments - Step-by-step guides

**Still Stuck?**:
- GitHub Issues
- Review error messages carefully
- Check GPU/memory availability
- Verify file paths

---

## 🎉 Success Criteria

**✅ Project Complete When**:
- [ ] All 4 stages trained
- [ ] All result JSONs collected
- [ ] Stage 4 comparison executed
- [ ] Visualizations generated
- [ ] Best approach identified
- [ ] Documentation updated

**Current Progress**: 75% (3/4 stages complete)

---

**Last Updated**: November 12, 2025  
**Quick Ref Version**: 1.0  
**For**: Gaming Sentiment Analysis Project
