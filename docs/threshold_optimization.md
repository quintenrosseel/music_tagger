# Threshold Optimization for Multi-Label Classification

## Overview

This document explains how threshold optimization works in the music tagging project, why it's important, and how to use it effectively.

## Table of Contents

1. [How Probabilities Work](#how-probabilities-work)
2. [What Are Thresholds?](#what-are-thresholds)
3. [Why Optimize Thresholds?](#why-optimize-thresholds)
4. [How Threshold Optimization Works](#how-threshold-optimization-works)
5. [Usage Examples](#usage-examples)
6. [When to Use Threshold Optimization](#when-to-use-threshold-optimization)

---

## How Probabilities Work

### Models Predict Probabilities (0 to 1)

Both Random Forest and XGBoost have a `predict_proba()` method that returns probability estimates for each class:

```python
# For a single label classifier:
probabilities = model.predict_proba(X)
# Returns: [[prob_class_0, prob_class_1], ...]
#           [0.3,          0.7        ]  ← 70% chance of class 1
```

### Multi-Label Classification

For our multi-label music classification (using `MultiOutputClassifier`), we get probabilities for each label:

```python
probabilities_list = model.predict_proba(X)
# Returns a LIST of arrays, one per label
# [
#   array([[0.8, 0.2], [0.4, 0.6], ...]),  # Label 1 (Soul) probabilities
#   array([[0.7, 0.3], [0.3, 0.7], ...]),  # Label 2 (House) probabilities
#   ...
# ]
```

### Extracting Positive Class Probabilities

We extract the probability of the positive class (class 1 = "has this tag") for each label:

```python
# Extract probability of positive class (class 1) for each label
probabilities = np.column_stack([proba[:, 1] for proba in probabilities_list])
# Shape: (n_samples, n_labels)
```

This gives us a probability matrix:
- **Rows** = songs
- **Columns** = labels (genres)
- **Values** = probability between 0.0 and 1.0 that the song has that label

**Example for one song:**
```
       Soul  House  Bass  Funk  Techno ...
Song1: 0.35  0.65   0.28  0.72  0.15   ...
```

---

## What Are Thresholds?

A **threshold** is the cutoff value used to convert probabilities into binary predictions (0 or 1).

### Default Threshold: 0.5

By default, scikit-learn uses a threshold of **0.5** for all labels:

```python
predictions = (probabilities >= 0.5).astype(int)
```

**Example:**
```
Probabilities:  [0.35, 0.65, 0.28, 0.72, 0.15]
                 Soul  House Bass  Funk  Techno
Threshold:       0.5 (same for all labels)
Predictions:    [0,    1,    0,    1,    0   ]
```

### Per-Label Thresholds

Instead of using 0.5 for all labels, we can use **different thresholds** for each label:

```python
thresholds = [0.30, 0.50, 0.20, 0.60, 0.40]  # One per label
predictions = (probabilities >= thresholds).astype(int)
```

**Example:**
```
Probabilities:     [0.35, 0.65, 0.28, 0.72, 0.15]
                    Soul  House Bass  Funk  Techno
Thresholds:        [0.30, 0.50, 0.20, 0.60, 0.40]
Predictions:       [1,    1,    1,    1,    0   ]
                    ↑                ↑
              Now predicts!    Now predicts!
```

---

## Why Optimize Thresholds?

### Problem: Class Imbalance

In multi-label music classification, we have severe class imbalance:
- **Common genres** (House, Soul): Many training examples
- **Rare genres** (Drum & Bass, Electro): Few training examples

Models learn to predict **higher probabilities** for common classes and **lower probabilities** for rare classes, even when the song actually belongs to the rare class.

### The 0.5 Threshold Problem

With a fixed 0.5 threshold:
- **Common genres**: Get predicted correctly (probabilities often > 0.5)
- **Rare genres**: Get ignored (probabilities rarely exceed 0.5)

**Real example from your XGBoost results:**
```
XGBoost predictions (default 0.5 threshold):
- Only 5 out of 30 genres predicted
- Many genres: 0.0 F1 score
- High precision, terrible recall
```

The model was actually outputting reasonable probabilities, but the 0.5 threshold was too high!

### Solution: Per-Label Thresholds

By optimizing thresholds for each label independently, we can:
- **Lower thresholds** for rare classes (e.g., 0.25) → better recall
- **Keep higher thresholds** for common classes (e.g., 0.55) → maintain precision
- **Maximize F1 score** for each label independently

---

## How Threshold Optimization Works

### The Process

For **each label independently**:

1. **Get predictions on validation set**: Use `predict_proba()` to get probabilities
2. **Try many thresholds**: Evaluate performance at different threshold values
3. **Calculate metric**: Compute F1, precision, or recall at each threshold
4. **Select best threshold**: Choose the threshold that maximizes the metric

### Implementation

From `models.py`:

```python
def find_optimal_thresholds(model, X_val, y_val, metric='f1'):
    """Find optimal threshold for each label independently."""

    # Get probabilities for all labels
    probabilities_list = model.predict_proba(X_val)
    probabilities = np.column_stack([proba[:, 1] for proba in probabilities_list])

    optimal_thresholds = []

    for i in range(y_val.shape[1]):  # For each label
        # Get precision-recall curve
        precision, recall, thresholds = precision_recall_curve(
            y_val[:, i],           # True labels for this label
            probabilities[:, i]     # Predicted probabilities for this label
        )

        # Calculate F1 scores at each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

        # Find threshold that maximizes F1
        optimal_idx = np.argmax(f1_scores)
        optimal_thresholds.append(thresholds[optimal_idx])

    return np.array(optimal_thresholds)
```

### Visual Example

For the "Soul" label:

```
                                     F1 Score
                                        ▲
                                        │
                    ┌──────────────────┐│
                    │                  ││
                    │                  ││
                    │                  ││
        ┌───────────┤                  ││
        │           │                  ││
        │           │                  ││
────────┼───────────┼──────────────────┼┼──────────► Threshold
       0.0         0.3                0.5           1.0
                    ↑
              Optimal = 0.3
           (Lower than default 0.5!)
```

The optimal threshold is 0.3, not 0.5, because:
- Soul is relatively common, but not dominant
- Lowering the threshold to 0.3 increases recall without hurting precision too much
- Overall F1 score is maximized at 0.3

---

## Usage Examples

### Basic Usage

```python
from models import optimize_prediction_thresholds, predict_with_threshold

# 1. Train your model
model = get_xgboost_model(n_estimators=200, scale_pos_weight=5)
model.fit(X_train, y_train)

# 2. Optimize thresholds on validation set
threshold_results = optimize_prediction_thresholds(
    model=model,
    X_val=X_val,
    y_val=y_val,
    tags=tags,
    metric='f1',  # or 'precision', 'recall'
    verbose=True
)

# 3. Get optimized thresholds
optimal_thresholds = threshold_results['thresholds']
# Example: array([0.30, 0.50, 0.25, 0.60, 0.35, ...])

# 4. Make predictions on test set with optimized thresholds
_, test_probs = predict_with_threshold(model, X_test, threshold=0.5)
y_pred = (test_probs >= optimal_thresholds).astype(int)
```

### Comparing Before and After

```python
# Before: Default 0.5 threshold
y_pred_default = model.predict(X_test)
f1_default = f1_score(y_test, y_pred_default, average='macro')
print(f"Default threshold F1: {f1_default:.4f}")

# After: Optimized thresholds
_, test_probs = predict_with_threshold(model, X_test, threshold=0.5)
y_pred_optimized = (test_probs >= optimal_thresholds).astype(int)
f1_optimized = f1_score(y_test, y_pred_optimized, average='macro')
print(f"Optimized threshold F1: {f1_optimized:.4f}")
print(f"Improvement: {f1_optimized - f1_default:+.4f}")
```

### Comparing Multiple Models

```python
# Tune both Random Forest and XGBoost
rf_model = get_random_forest_model(n_estimators=200)
xgb_model = get_xgboost_model(n_estimators=200, scale_pos_weight=5)

rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Optimize thresholds for both
rf_thresholds = optimize_prediction_thresholds(
    rf_model, X_val, y_val, tags, metric='f1'
)
xgb_thresholds = optimize_prediction_thresholds(
    xgb_model, X_val, y_val, tags, metric='f1'
)

# Compare on validation set
print(f"RF F1: {rf_thresholds['optimized_metrics']['macro_f1']:.4f}")
print(f"XGB F1: {xgb_thresholds['optimized_metrics']['macro_f1']:.4f}")
```

---

## When to Use Threshold Optimization

### Always Use When:

✅ **Class imbalance**: Different labels have very different frequencies
✅ **Multi-label classification**: Each label needs its own threshold
✅ **Imbalanced precision/recall**: Model is too conservative or too aggressive
✅ **Low recall on rare classes**: Model isn't predicting minority classes

### Especially Important For:

🔥 **XGBoost with `scale_pos_weight`**:
- `scale_pos_weight` adjusts training, but doesn't fix decision threshold
- Threshold optimization is critical for good recall
- Expected improvement: +0.10 to +0.20 macro F1

🔥 **Production systems**:
- Different labels may have different business requirements
- Some labels need high precision (fewer false positives)
- Other labels need high recall (catch everything)

### Workflow:

1. **Train/Val/Test Split**: Use `preprocess_tag_group()` with `val_size > 0`
2. **Train model** on training set
3. **Optimize thresholds** on validation set
4. **Evaluate** on test set with optimized thresholds

**Important:** Never optimize thresholds on the test set! This would leak information and overestimate performance.

---

## Expected Improvements

Based on your dataset:

| Scenario | Before | After | Improvement |
|----------|--------|-------|-------------|
| Random Forest (baseline) | ~0.35 F1 | ~0.40 F1 | +0.05 F1 |
| XGBoost (conservative) | ~0.13 F1 | ~0.35 F1 | **+0.22 F1** |
| Combined with hyperparameter tuning | ~0.30 F1 | ~0.50 F1 | +0.20 F1 |

### Why XGBoost Benefits More:

Your original XGBoost results showed only 5/30 genres predicted because:
1. **Class imbalance** → model outputs lower probabilities for rare classes
2. **`scale_pos_weight=10`** → adjusts training but not threshold
3. **Default 0.5 threshold** → too high for rare classes
4. **Result**: Good probabilities, but bad predictions

**Solution**: Threshold optimization finds per-label thresholds (e.g., 0.25 for rare genres, 0.55 for common genres)

---

## Key Takeaways

1. **Models always predict probabilities** (0-1), not binary labels
2. **Thresholds convert probabilities → predictions** (default: 0.5)
3. **Default 0.5 is suboptimal** for imbalanced multi-label data
4. **Per-label thresholds** are found by maximizing F1 on validation set
5. **No retraining needed** - just changes how we interpret probabilities
6. **Especially critical for XGBoost** with class imbalance handling

---

## References

- `models.py:find_optimal_thresholds()` - Core implementation
- `models.py:optimize_prediction_thresholds()` - User-friendly wrapper
- `models.py:predict_with_threshold()` - Make predictions with custom thresholds
- `notebooks/07_hyperparameter_tuning.ipynb` - Complete example workflow

---

## Further Reading

- [Sklearn: Precision-Recall Curves](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)
- [Dealing with Imbalanced Data](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/)
- [Multi-label Classification](https://scikit-learn.org/stable/modules/multiclass.html#multilabel-classification)
