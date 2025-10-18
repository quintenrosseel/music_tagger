# Threshold Tuning Playbook

## Overview

This playbook provides step-by-step instructions for diagnosing and fixing threshold issues for individual labels in your multi-label music classification system. Use this when you notice specific labels having too many false positives (threshold too low) or false negatives (threshold too high).

## Table of Contents

1. [Quick Diagnosis](#quick-diagnosis)
2. [Problem: Too Many False Positives](#problem-too-many-false-positives-threshold-too-low)
3. [Problem: Too Many False Negatives](#problem-too-many-false-negatives-threshold-too-high)
4. [Manual Threshold Tuning](#manual-threshold-tuning)
5. [Re-optimizing Thresholds](#re-optimizing-thresholds-from-scratch)
6. [Testing Your Changes](#testing-your-changes)
7. [Saving Updated Thresholds](#saving-updated-thresholds)

---

## Quick Diagnosis

When you suspect a label has threshold issues, ask yourself:

**Too many FALSE POSITIVES?** (Label appears when it shouldn't)

- Songs are getting tagged with the label incorrectly
- Example: "Motown" appears on non-Motown songs
- **Solution**: Threshold is too LOW → Increase it

**Too many FALSE NEGATIVES?** (Label missing when it should be there)

- Songs that should have the label don't get it
- Example: Clear techno tracks aren't tagged as "Techno"
- **Solution**: Threshold is too HIGH → Decrease it

---

## Problem: Too Many False Positives (Threshold Too Low)

### Symptoms

- Label appears on many songs where it doesn't belong
- High recall but low precision for this label
- Example: "Motown" tag shows up on House, Techno, and other non-Motown genres

### Diagnosis Steps

#### 1. Check Current Threshold and Performance

```python
import polars as pl
from models import load_model

# Load your model
model_data = load_model("models/xgboost_Genre_20251015_232027_model.pkl")

# Find the label index
label_name = "Motown"
label_idx = model_data['tags'].index(label_name)

# Check current threshold
current_threshold = model_data['thresholds'][label_idx]
print(f"Current threshold for '{label_name}': {current_threshold:.4f}")

# If you have validation metrics stored, check them
if 'per_label_metrics' in model_data:
    metrics = model_data['per_label_metrics']
    label_metrics = metrics.filter(pl.col("tag") == label_name)
    print(f"\nCurrent metrics:\n{label_metrics}")
```

#### 2. Analyze Predictions on Sample Data

```python
from processing import predict_with_optimized_thresholds
from utils import get_clean_songs
from pyrekordbox import Rekordbox6Database

# Get songs and features
db = Rekordbox6Database()
songs_df = get_clean_songs(db, rename=True)
features_df = pl.read_parquet("data/song_features.parquet")

# Get scaled features (same as training)
X_test = features_df.select(["song_id"] + feature_cols)
X_test_scaled = model_data['scaler'].transform(X_test.drop("song_id"))
X_test_scaled = pl.concat([X_test.select("song_id"), X_test_scaled], how="horizontal")

# Get predictions with current thresholds
predictions = predict_with_optimized_thresholds(
    models=model_data['model'],
    X_test=X_test_scaled,
    tags=model_data['tags'],
    thresholds=model_data['thresholds'],
    songs_df=songs_df
)

# Find songs predicted as Motown
motown_predictions = predictions.filter(
    pl.col("predicted_tags").list.contains("Motown")
)

print(f"\nSongs predicted as '{label_name}': {len(motown_predictions)}")
print(motown_predictions.select(["song_title", "artist_name", "predicted_tags"]))
```

#### 3. Check Prediction Probabilities

```python
import numpy as np
from models import predict_with_threshold

# Get probabilities for all predictions
_, probs = predict_with_threshold(
    model_data['model'],
    X_test_scaled.drop("song_id"),
    threshold=0.5
)

# Get probabilities for our label
label_probs = probs[:, label_idx]

# Add to dataframe
prob_df = predictions.with_columns(
    pl.lit(label_probs).alias(f"{label_name}_probability")
)

# Look at songs with Motown predictions and their probabilities
motown_prob_df = (
    prob_df
    .filter(pl.col("predicted_tags").list.contains(label_name))
    .select(["song_title", "artist_name", f"{label_name}_probability", "predicted_tags"])
    .sort(f"{label_name}_probability", descending=True)
)

print(f"\nSongs predicted as '{label_name}' with probabilities:")
print(motown_prob_df)

# Find the probability range
print(f"\nProbability range for '{label_name}' predictions:")
print(f"  Min: {motown_prob_df[f'{label_name}_probability'].min():.4f}")
print(f"  Max: {motown_prob_df[f'{label_name}_probability'].max():.4f}")
print(f"  Mean: {motown_prob_df[f'{label_name}_probability'].mean():.4f}")
print(f"  Current threshold: {current_threshold:.4f}")
```

### Solution: Increase the Threshold

Based on your analysis, determine a better threshold:

```python
# Look at the probability distribution
# If many false positives are in range 0.25-0.40 and current threshold is 0.25,
# try increasing to 0.35 or 0.40

# Try different thresholds and see the effect
test_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]

for test_threshold in test_thresholds:
    # Apply test threshold
    test_thresholds_array = model_data['thresholds'].copy()
    test_thresholds_array[label_idx] = test_threshold

    # Make predictions
    test_preds = predict_with_optimized_thresholds(
        models=model_data['model'],
        X_test=X_test_scaled,
        tags=model_data['tags'],
        thresholds=test_thresholds_array,
        songs_df=songs_df
    )

    # Count predictions
    n_predictions = len(
        test_preds.filter(pl.col("predicted_tags").list.contains(label_name))
    )

    print(f"Threshold {test_threshold:.2f}: {n_predictions} predictions")
```

**Choose a threshold that:**

- Reduces false positives significantly
- Still catches true positives (check a few known Motown songs)
- Is based on a clear gap in the probability distribution

**Example Decision:**
```python
# After analysis, you decide 0.40 is better than 0.25
new_threshold = 0.40

# Update the thresholds array
updated_thresholds = model_data['thresholds'].copy()
updated_thresholds[label_idx] = new_threshold

print(f"\nUpdated '{label_name}' threshold: {current_threshold:.4f} → {new_threshold:.4f}")
```

---

## Problem: Too Many False Negatives (Threshold Too High)

### Symptoms
- Label is missing from songs where it should appear
- High precision but low recall for this label
- Example: Clear "Techno" tracks aren't getting the "Techno" tag

### Diagnosis Steps

Same as above, but look for:
- Songs that SHOULD have the label but don't
- Check if their probabilities are just below the current threshold
- Check manually tagged ground truth if available

```python
# Find songs with ground truth labels (from your Rekordbox tags)
ground_truth_label = songs_df.filter(
    pl.col("tag_names").list.contains(label_name)
)

# Get predictions for these songs
gt_with_preds = ground_truth_label.join(
    prob_df.select(["song_id", "predicted_tags", f"{label_name}_probability"]),
    on="song_id",
    how="inner"
)

# Find songs that SHOULD have the label but don't
missed_songs = gt_with_preds.filter(
    ~pl.col("predicted_tags").list.contains(label_name)
)

print(f"\nMissed predictions for '{label_name}':")
print(f"Total ground truth: {len(ground_truth_label)}")
print(f"Correctly predicted: {len(gt_with_preds) - len(missed_songs)}")
print(f"Missed: {len(missed_songs)}")

print(f"\nMissed songs with probabilities:")
print(
    missed_songs
    .select(["song_title", "artist_name", f"{label_name}_probability"])
    .sort(f"{label_name}_probability", descending=True)
)
```

### Solution: Decrease the Threshold

```python
# If many true positives have probabilities around 0.35-0.45
# and current threshold is 0.50, try lowering to 0.40

# Try different thresholds
test_thresholds = [0.50, 0.45, 0.40, 0.35, 0.30]

for test_threshold in test_thresholds:
    test_thresholds_array = model_data['thresholds'].copy()
    test_thresholds_array[label_idx] = test_threshold

    test_preds = predict_with_optimized_thresholds(
        models=model_data['model'],
        X_test=X_test_scaled,
        tags=model_data['tags'],
        thresholds=test_thresholds_array,
        songs_df=songs_df
    )

    # Calculate recall on ground truth
    gt_preds = ground_truth_label.join(
        test_preds.select(["song_id", "predicted_tags"]),
        on="song_id",
        how="inner"
    )

    recalled = len(
        gt_preds.filter(pl.col("predicted_tags").list.contains(label_name))
    )
    recall = recalled / len(gt_preds) if len(gt_preds) > 0 else 0

    print(f"Threshold {test_threshold:.2f}: Recall = {recall:.2%} ({recalled}/{len(gt_preds)})")
```

**Choose a threshold that:**
- Increases recall (catches more true positives)
- Doesn't introduce too many false positives
- Maximizes F1 score if you can calculate it

---

## Manual Threshold Tuning

If you want fine-grained control, you can manually tune thresholds for any label:

```python
from models import save_model
import numpy as np

# Load model
model_data = load_model("models/xgboost_Genre_20251015_232027_model.pkl")

# Print current thresholds
print("Current thresholds:")
for tag, threshold in zip(model_data['tags'], model_data['thresholds']):
    print(f"  {tag:20s}: {threshold:.4f}")

# Update specific thresholds
updated_thresholds = model_data['thresholds'].copy()

# Increase Motown threshold (too many false positives)
motown_idx = model_data['tags'].index("Motown")
updated_thresholds[motown_idx] = 0.45  # was 0.25

# Decrease Techno threshold (missing too many)
techno_idx = model_data['tags'].index("Techno")
updated_thresholds[techno_idx] = 0.35  # was 0.50

# Save with updated thresholds
model_data['thresholds'] = updated_thresholds

save_model(
    model=model_data['model'],
    tags=model_data['tags'],
    model_name=model_data['model_name'],
    tag_group=model_data['tag_group'],
    save_dir="models",
    thresholds=updated_thresholds,
    scaler=model_data['scaler'],
    pca=model_data.get('pca'),
    metrics=model_data.get('metrics')
)

print("\n✅ Model saved with updated thresholds!")
```

---

## Re-optimizing Thresholds from Scratch

If many labels have issues, you might want to re-run threshold optimization with different settings:

```python
from models import optimize_prediction_thresholds, save_model
from processing import preprocess_tag_group
from utils import get_base_dataset

# Load data
base_df = get_base_dataset(db, min_tag_count=10)
features_df = pl.read_parquet("data/song_features.parquet")

# Preprocess with validation set
result = preprocess_tag_group(
    base_df=base_df,
    features_df=features_df,
    tag_group="Genre",
    test_size=0.2,
    val_size=0.2,  # Important: need validation set for threshold tuning
    apply_scaling=True,
    apply_pca=False,
    min_train_count=10
)

# Load your trained model
model_data = load_model("models/xgboost_Genre_20251015_232027_model.pkl")
model = model_data['model']

# Re-optimize thresholds on validation set
print("Re-optimizing thresholds on validation set...")
threshold_results = optimize_prediction_thresholds(
    model=model,
    X_val=result.X_val,
    y_val=result.y_val,
    tags=result.tags,
    metric='f1',  # Can also try 'precision' or 'recall'
    verbose=True
)

# Check the new thresholds
print("\nOld vs New Thresholds:")
for tag, old_thresh, new_thresh in zip(
    result.tags,
    model_data['thresholds'],
    threshold_results['thresholds']
):
    change = "↑" if new_thresh > old_thresh else "↓" if new_thresh < old_thresh else "="
    print(f"  {tag:20s}: {old_thresh:.4f} → {new_thresh:.4f} {change}")

# Save model with new thresholds
save_model(
    model=model,
    tags=result.tags,
    model_name=model_data['model_name'],
    tag_group="Genre",
    save_dir="models",
    thresholds=threshold_results['thresholds'],
    scaler=result.scaler,
    pca=result.pca,
    metrics=threshold_results['optimized_metrics']
)

print("\n✅ Model re-saved with optimized thresholds!")
```

### Alternative Optimization Metrics

Try different metrics depending on your goal:

```python
# For high precision (fewer false positives)
threshold_results_precision = optimize_prediction_thresholds(
    model=model,
    X_val=result.X_val,
    y_val=result.y_val,
    tags=result.tags,
    metric='precision',  # Prioritize precision
    verbose=True
)

# For high recall (catch everything)
threshold_results_recall = optimize_prediction_thresholds(
    model=model,
    X_val=result.X_val,
    y_val=result.y_val,
    tags=result.tags,
    metric='recall',  # Prioritize recall
    verbose=True
)

# Compare the differences
print("\nThreshold comparison by optimization metric:")
for tag, f1_t, prec_t, rec_t in zip(
    result.tags,
    threshold_results['thresholds'],
    threshold_results_precision['thresholds'],
    threshold_results_recall['thresholds']
):
    print(f"{tag:20s}: F1={f1_t:.2f}, Prec={prec_t:.2f}, Rec={rec_t:.2f}")
```

---

## Testing Your Changes

After updating thresholds, always test on real data:

```python
from processing import predict_with_optimized_thresholds

# Make predictions with updated thresholds
predictions = predict_with_optimized_thresholds(
    models=model_data['model'],
    X_test=X_test_scaled,
    tags=model_data['tags'],
    thresholds=updated_thresholds,  # Your new thresholds
    songs_df=songs_df
)

# Spot check specific cases
print("\n=== Motown Predictions (should be reduced) ===")
motown_preds = predictions.filter(pl.col("predicted_tags").list.contains("Motown"))
print(f"Count: {len(motown_preds)}")
print(motown_preds.select(["song_title", "artist_name", "predicted_tags"]).head(20))

# Check known ground truth songs
print("\n=== Known Motown Songs (should still be tagged) ===")
known_motown = songs_df.filter(pl.col("tag_names").list.contains("Motown"))
known_motown_preds = known_motown.join(predictions, on="song_id", how="inner")
recalled = len(
    known_motown_preds.filter(pl.col("predicted_tags").list.contains("Motown"))
)
print(f"Recall: {recalled}/{len(known_motown)} = {recalled/len(known_motown):.1%}")
print(
    known_motown_preds
    .select(["song_title", "artist_name", "tag_names", "predicted_tags"])
    .head(20)
)
```

### Quality Checks

Create a validation checklist:

```python
# 1. Check overall prediction counts
for tag in model_data['tags']:
    count = len(predictions.filter(pl.col("predicted_tags").list.contains(tag)))
    print(f"{tag:20s}: {count:4d} predictions")

# 2. Check songs with many tags (might indicate too many low thresholds)
predictions_with_count = predictions.with_columns(
    pl.col("predicted_tags").list.len().alias("n_tags")
)
print(f"\nSongs with 5+ tags: {len(predictions_with_count.filter(pl.col('n_tags') >= 5))}")

# 3. Check songs with no tags (might indicate too many high thresholds)
print(f"Songs with 0 tags: {len(predictions_with_count.filter(pl.col('n_tags') == 0))}")

# 4. Manually review a random sample
sample = predictions.sample(n=20, seed=42)
print("\nRandom sample predictions:")
print(sample.select(["song_title", "artist_name", "predicted_tags"]))
```

---

## Saving Updated Thresholds

Once you're happy with your thresholds, save them with the model:

```python
from models import save_model
from datetime import datetime

# Update thresholds in model_data
model_data['thresholds'] = updated_thresholds

# Option 1: Overwrite existing model
save_model(
    model=model_data['model'],
    tags=model_data['tags'],
    model_name=model_data['model_name'],
    tag_group=model_data['tag_group'],
    save_dir="models",
    thresholds=updated_thresholds,
    scaler=model_data['scaler'],
    pca=model_data.get('pca'),
    metrics=model_data.get('metrics')
)

# Option 2: Save as new version (safer)
new_model_name = f"{model_data['model_name']}_tuned"
save_model(
    model=model_data['model'],
    tags=model_data['tags'],
    model_name=new_model_name,
    tag_group=model_data['tag_group'],
    save_dir="models",
    thresholds=updated_thresholds,
    scaler=model_data['scaler'],
    pca=model_data.get('pca'),
    metrics=model_data.get('metrics')
)

print(f"\n✅ Model saved: models/{new_model_name}_Genre_*.pkl")
```

---

## Quick Reference

### Common Threshold Adjustments

| Problem | Symptom | Action | Typical Change |
|---------|---------|--------|----------------|
| Too many false positives | Label appears everywhere | ↑ Increase threshold | +0.10 to +0.20 |
| Too many false negatives | Label rarely appears | ↓ Decrease threshold | -0.10 to -0.15 |
| Model too conservative | Very few predictions overall | ↓ Decrease all thresholds | -0.05 to -0.10 |
| Model too aggressive | Too many tags per song | ↑ Increase all thresholds | +0.05 to +0.10 |

### Typical Threshold Ranges

- **Common labels** (House, Techno, Soul): 0.40 - 0.60
- **Rare labels** (Motown, Electro, DnB): 0.20 - 0.40
- **Very rare labels**: 0.15 - 0.30

### Decision Framework

```
1. Identify the problem label
2. Check current threshold and metrics
3. Analyze prediction probabilities
4. Test 3-5 candidate thresholds
5. Choose threshold that maximizes F1 (or your target metric)
6. Validate on sample data
7. Save and document changes
```

---

## Additional Resources

- [threshold_optimization.md](threshold_optimization.md) - Deep dive into how threshold optimization works
- [models.py](../models.py) - Implementation of `find_optimal_thresholds()` and `optimize_prediction_thresholds()`
- [notebooks/07_hyperparameter_tuning.ipynb](../notebooks/07_hyperparameter_tuning.ipynb) - Example workflow

---

## Troubleshooting

### "I keep getting false positives even with high thresholds"

The model might not have learned good features for that label. Check:
1. Training data quality - do you have enough examples?
2. Feature relevance - are audio features capturing the right characteristics?
3. Label definition - is the label too subjective or ambiguous?

### "All my thresholds are very low (< 0.30)"

This suggests:
1. Model is under-confident (try adjusting `scale_pos_weight` in training)
2. Severe class imbalance (consider oversampling rare classes)
3. Features might not be predictive enough

### "Threshold tuning doesn't help"

The problem might be in the model, not the thresholds:
1. Retrain with different hyperparameters
2. Try a different model architecture
3. Add or improve features
4. Get more training data for problematic labels

---

**Last Updated**: 2025-10-17
