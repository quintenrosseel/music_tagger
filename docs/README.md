# Music Tagger Documentation

This folder contains documentation for the music tagging machine learning project.

## Documentation Files

### 📊 [Threshold Optimization Guide](threshold_optimization.md)
**Comprehensive guide explaining how threshold optimization works**

Learn about:
- How models predict probabilities (0-1)
- What thresholds are and why they matter
- Why the default 0.5 threshold is suboptimal for imbalanced data
- How to optimize thresholds for multi-label classification
- Usage examples and expected improvements
- Why threshold optimization is especially important for XGBoost

**Perfect for understanding why your XGBoost only predicted 5/30 genres!**

### 🐻 [Polars Cheatsheet](polars_cheatsheet.md)
Quick reference for Polars DataFrame operations used throughout the project.

### 🔗 [Links](links.md)
Useful external resources and references.

---

## Quick Start

### For Model Training and Optimization:

1. **Read**: [Threshold Optimization Guide](threshold_optimization.md)
2. **Run**: `notebooks/07_hyperparameter_tuning.ipynb`
3. **Understand**: Why XGBoost benefits most from threshold optimization

### Key Concepts:

```python
# 1. Models predict probabilities (0-1)
probabilities = model.predict_proba(X)  # [0.35, 0.65, 0.28, ...]

# 2. Thresholds convert to binary predictions
predictions = (probabilities >= 0.5).astype(int)  # Default: 0.5

# 3. Optimize thresholds per label
optimal_thresholds = find_optimal_thresholds(model, X_val, y_val)
# Result: [0.30, 0.50, 0.25, 0.60, ...]  ← Different per label!

# 4. Use optimized thresholds
predictions = (probabilities >= optimal_thresholds).astype(int)
```

---

## Project Structure

```
music_tagger/
├── docs/                          # Documentation (you are here!)
│   ├── README.md                  # This file
│   ├── threshold_optimization.md # Threshold optimization guide
│   ├── polars_cheatsheet.md      # Polars reference
│   └── links.md                   # External resources
├── notebooks/                     # Jupyter notebooks
│   ├── 06_train.ipynb            # Basic training
│   └── 07_hyperparameter_tuning.ipynb  # Advanced tuning + thresholds
├── models.py                      # Model definitions and utilities
├── processing.py                  # Data preprocessing
└── utils.py                       # Helper functions
```

---

## Common Questions

### Q: Why does my XGBoost only predict a few classes?
**A:** The default 0.5 threshold is too high for imbalanced data. See [Threshold Optimization Guide](threshold_optimization.md#why-optimize-thresholds).

### Q: What's the difference between `scale_pos_weight` and threshold optimization?
**A:**
- `scale_pos_weight`: Adjusts training to handle class imbalance
- Threshold optimization: Adjusts decision boundaries after training
- **Use both together** for best results!

### Q: Should I optimize thresholds on the test set?
**A:** **No!** Always optimize on the validation set, then evaluate on test set. See [Usage Examples](threshold_optimization.md#usage-examples).

### Q: How much improvement should I expect?
**A:**
- Random Forest: +0.03 to +0.08 macro F1
- XGBoost: +0.10 to +0.20 macro F1 (especially if it's being too conservative)
- See [Expected Improvements](threshold_optimization.md#expected-improvements)

---

## Contributing

When adding new documentation:
1. Create a new `.md` file in this folder
2. Add it to this README
3. Use clear examples and code snippets
4. Include visual explanations when helpful

---

## Additional Resources

### In Code:
- `models.py` - See docstrings for all model functions
- `processing.py` - See docstrings for preprocessing functions
- Notebooks - Complete working examples

### External:
- [Sklearn Multi-label Classification](https://scikit-learn.org/stable/modules/multiclass.html#multilabel-classification)
- [Handling Imbalanced Data](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
