"""
Example script demonstrating how to train and evaluate models for music tagging.

This shows the complete workflow:
1. Load and preprocess data
2. Train multiple models
3. Evaluate and compare
4. Convert predictions to labels
"""

import polars as pl
from pyrekordbox import Rekordbox6Database
from sklearn.model_selection import train_test_split

from models import (
    compare_models,
    evaluate_model,
    get_decision_tree_model,
    get_knn_model,
    get_linear_model,
    get_random_forest_model,
    get_xgboost_model,
    predictions_to_dataframe,
    predictions_to_labels,
)
from processing import (
    PolarsPCA,
    PolarsStandardScaler,
    filter_rare_labels,
    prepare_multilabel_data,
)

db = Rekordbox6Database()
from utils import get_base_dataset

base_df = get_base_dataset(db, min_tag_count=5)

# ============================================================================
# CONFIGURATION
# ============================================================================
test_size = 0.2
random_state = 42
min_train_count = 10
pca_variance = 0.95

# ============================================================================
# LOAD AND PREPARE DATA (assuming you already have base_df)
# ============================================================================
# db = Rekordbox6Database()
# base_df = get_base_dataset(db, min_tag_count=5)
features_df = pl.read_parquet("./data/song_features.parquet")

# Define features to exclude
exclude_cols = [
    "song_path",
    "harmonic_percussive_ratio",
    "percussive_strength",
    "tonnetz_mean_0",
    "tonnetz_mean_1",
    "tonnetz_mean_2",
    "tonnetz_mean_3",
    "tonnetz_mean_4",
    "tonnetz_mean_5",
    "tonnetz_std_0",
    "tonnetz_std_1",
    "tonnetz_std_2",
    "tonnetz_std_3",
    "tonnetz_std_4",
    "tonnetz_std_5",
]

feature_cols = [
    col
    for col in features_df.columns
    if col not in exclude_cols + ["song_id", "song_path"]
]

features_df = features_df.filter(pl.col("energy_increase_ratio").is_not_null())

# ============================================================================
# PREPROCESS ONE TAG GROUP (e.g., Genre)
# ============================================================================
print("=" * 70)
print("PREPARING DATA FOR GENRE CLASSIFICATION")
print("=" * 70)

X, y, tags = prepare_multilabel_data(
    base_df=base_df,
    features_df=features_df,
    tag_group="Genre",
    feature_columns=feature_cols,
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X.drop("song_id"), y, test_size=test_size, random_state=random_state
)

# Filter rare labels
X_train, y_train, X_test, y_test, tags = filter_rare_labels(
    X_train, y_train, X_test, y_test, tags, min_count=min_train_count
)

# Normalize
scaler = PolarsStandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PolarsPCA(n_components=pca_variance)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("\nFinal dataset:")
print(f"  Train: {X_train_pca.shape}, Test: {X_test_pca.shape}")
print(f"  Labels: {len(tags)}")
print(f"  Tags: {', '.join(tags)}\n")

# ============================================================================
# TRAIN AND EVALUATE INDIVIDUAL MODELS
# ============================================================================

# Example 1: Train a single model (Decision Tree)
print("\n" + "=" * 70)
print("EXAMPLE 1: TRAIN DECISION TREE")
print("=" * 70)

dt_model = get_decision_tree_model(max_depth=15, min_samples_split=20)
dt_model.fit(X_train_pca.to_numpy(), y_train.to_numpy())

# Evaluate
dt_metrics = evaluate_model(dt_model, X_test_pca, y_test, tags, verbose=True)

# Convert predictions to labels
y_pred = dt_metrics["predictions"]
predicted_labels = predictions_to_labels(y_pred, tags)

print("\nExample predictions (first 5 test samples):")
for i in range(min(5, len(predicted_labels))):
    print(f"  Sample {i + 1}: {predicted_labels[i]}")

# ============================================================================
# COMPARE MULTIPLE MODELS
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 2: COMPARE MULTIPLE MODELS")
print("=" * 70)

# Define models to compare
models = {
    "Linear": get_linear_model(C=1.0),
    "KNN (k=10)": get_knn_model(n_neighbors=10, weights="distance"),
    "Decision Tree": get_decision_tree_model(max_depth=15, min_samples_split=20),
    "Random Forest": get_random_forest_model(
        n_estimators=100, max_depth=15, min_samples_split=20
    ),
}

# Add XGBoost if available
try:
    models["XGBoost"] = get_xgboost_model(n_estimators=100, learning_rate=0.1)
except ImportError:
    print("⚠️  XGBoost not available, skipping...")

# Compare all models
comparison = compare_models(models, X_train_pca, y_train, X_test_pca, y_test, tags)

print("\nBest model by F1-score:")
best_model_name = comparison.row(0, named=True)["model"]
best_f1 = comparison.row(0, named=True)["macro_f1"]
print(f"  {best_model_name}: F1 = {best_f1:.4f}")

# ============================================================================
# CONVERT PREDICTIONS TO DATAFRAME
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 3: CONVERT PREDICTIONS TO DATAFRAME")
print("=" * 70)

# Train best model
best_model = get_random_forest_model(n_estimators=200, max_depth=20)
best_model.fit(X_train_pca.to_numpy(), y_train.to_numpy())

# Predict on test set
y_pred = best_model.predict(X_test_pca.to_numpy())

# Create song IDs (in practice, you'd have real song IDs)
test_song_ids = [f"song_{i}" for i in range(len(y_pred))]

# Convert to DataFrame
predictions_df = predictions_to_dataframe(y_pred, tags, test_song_ids)

print("\nPredictions DataFrame:")
print(predictions_df.head(10))

# Get true labels for comparison
y_true_labels = predictions_to_labels(y_test.to_numpy(), tags)
true_labels_df = pl.DataFrame({"song_id": test_song_ids, "true_tags": y_true_labels})

# Combine predictions and true labels
comparison_df = predictions_df.join(true_labels_df, on="song_id")
print("\nPredictions vs True Labels:")
print(comparison_df.head(10))

# ============================================================================
# FEATURE IMPORTANCE (for tree-based models)
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 4: FEATURE IMPORTANCE (Random Forest)")
print("=" * 70)

rf_model = get_random_forest_model(n_estimators=100, max_depth=15)
rf_model.fit(X_train_pca.to_numpy(), y_train.to_numpy())

# Get feature importances for each label
print("\nTop 3 most important features per label:")
for i, tag in enumerate(tags[:5]):  # Show first 5 labels
    estimator = rf_model.estimators_[i]
    importances = estimator.feature_importances_

    # Get top 3 features
    top_3_indices = importances.argsort()[-3:][::-1]

    print(f"\n{tag}:")
    for idx in top_3_indices:
        # Feature names are PC1, PC2, etc. after PCA
        feature_name = f"PC{idx + 1}"
        print(f"  {feature_name}: {importances[idx]:.4f}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
