"""
Multi-label classification models for music tagging.

This module provides sklearn-compatible models and utilities for training
and evaluating multi-label classifiers on music data.

Models implemented:
- Linear Model (Logistic Regression)
- K-Nearest Neighbors (KNN)
- Decision Tree (highly explainable)
- Random Forest (ensemble)
- XGBoost (gradient boosting)
"""

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")


def get_linear_model(C=1.0, max_iter=1000, random_state=42, balanced=True):
    """Get a Linear Model (Logistic Regression) for multi-label classification.

    Logistic Regression is a simple linear model that's fast to train and provides
    probabilistic predictions. Works well when classes are linearly separable.

    Parameters
    ----------
    C : float, default=1.0
        Inverse of regularization strength. Smaller values = stronger regularization.
    max_iter : int, default=1000
        Maximum number of iterations for solver convergence.
    random_state : int, default=42
        Random state for reproducibility.
    balanced : bool, default=True
        If True, use class_weight='balanced' to handle imbalanced data.
        Recommended to keep True for multi-label classification.

    Returns
    -------
    model : MultiOutputClassifier
        Multi-label classifier wrapping LogisticRegression.

    Examples
    --------
    >>> # Balanced model (default)
    >>> model = get_linear_model(C=0.5)
    >>> model.fit(X_train.to_numpy(), y_train.to_numpy())
    >>>
    >>> # Unbalanced model (not recommended)
    >>> model = get_linear_model(C=0.5, balanced=False)
    >>> model.fit(X_train.to_numpy(), y_train.to_numpy())
    """
    base_model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        class_weight="balanced" if balanced else None,
        n_jobs=-1,
    )
    return MultiOutputClassifier(base_model, n_jobs=-1)


def get_knn_model(n_neighbors=5, weights="distance", metric="minkowski"):
    """Get a K-Nearest Neighbors model for multi-label classification.

    KNN classifies based on the k nearest training examples. Non-parametric
    and doesn't require training, but can be slow for large datasets.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    weights : {'uniform', 'distance'}, default='distance'
        Weight function:
        - 'uniform': All neighbors weighted equally
        - 'distance': Closer neighbors have more influence
    metric : str, default='minkowski'
        Distance metric. 'minkowski' with p=2 is Euclidean distance.

    Returns
    -------
    model : MultiOutputClassifier
        Multi-label classifier wrapping KNeighborsClassifier.

    Examples
    --------
    >>> model = get_knn_model(n_neighbors=10, weights='distance')
    >>> model.fit(X_train.to_numpy(), y_train.to_numpy())
    >>> y_pred = model.predict(X_test.to_numpy())
    """
    base_model = KNeighborsClassifier(
        n_neighbors=n_neighbors, weights=weights, metric=metric, n_jobs=-1
    )
    return MultiOutputClassifier(base_model, n_jobs=-1)


def get_decision_tree_model(
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    balanced=True,
):
    """Get a Decision Tree model for multi-label classification.

    Decision Trees are highly interpretable - you can visualize and understand
    the exact decision rules. Prone to overfitting without regularization.

    Parameters
    ----------
    max_depth : int, default=10
        Maximum depth of the tree. Controls model complexity.
    min_samples_split : int, default=10
        Minimum samples required to split a node. Higher = simpler tree.
    min_samples_leaf : int, default=5
        Minimum samples required at leaf node. Higher = simpler tree.
    random_state : int, default=42
        Random state for reproducibility.
    balanced : bool, default=True
        If True, use class_weight='balanced' to handle imbalanced data.
        Recommended to keep True for multi-label classification.

    Returns
    -------
    model : MultiOutputClassifier
        Multi-label classifier wrapping DecisionTreeClassifier.

    Examples
    --------
    >>> # Standard balanced model (recommended for multi-label)
    >>> model = get_decision_tree_model(max_depth=15, min_samples_split=20)
    >>> model.fit(X_train.to_numpy(), y_train.to_numpy())
    >>>
    >>> # Unbalanced model (not recommended for multi-label)
    >>> model = get_decision_tree_model(max_depth=15, balanced=False)
    >>> model.fit(X_train.to_numpy(), y_train.to_numpy())
    """
    base_model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced" if balanced else None,
        random_state=random_state,
    )
    return MultiOutputClassifier(base_model, n_jobs=-1)


def get_random_forest_model(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
):
    """Get a Random Forest model for multi-label classification.

    Random Forest is an ensemble of decision trees. Generally provides better
    performance than single decision trees and is less prone to overfitting.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest. More trees = better performance but slower.
    max_depth : int, default=15
        Maximum depth of each tree.
    min_samples_split : int, default=10
        Minimum samples required to split a node.
    min_samples_leaf : int, default=5
        Minimum samples required at leaf node.
    random_state : int, default=42
        Random state for reproducibility.

    Returns
    -------
    model : MultiOutputClassifier
        Multi-label classifier wrapping RandomForestClassifier.

    Examples
    --------
    >>> model = get_random_forest_model(n_estimators=200, max_depth=20)
    >>> model.fit(X_train.to_numpy(), y_train.to_numpy())
    >>> y_pred = model.predict(X_test.to_numpy())
    >>>
    >>> # Get feature importances for each label
    >>> for i, estimator in enumerate(model.estimators_):
    ...     importances = estimator.feature_importances_
    ...     print(f"Label {i}: Top feature importance = {importances.max():.3f}")
    """
    base_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    return MultiOutputClassifier(base_model, n_jobs=-1)


def get_xgboost_model(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=None,
):
    """Get an XGBoost model for multi-label classification.

    XGBoost is a powerful gradient boosting framework that often achieves
    state-of-the-art results. Can be slower to train but very effective.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds.
    max_depth : int, default=6
        Maximum depth of each tree.
    learning_rate : float, default=0.1
        Step size shrinkage to prevent overfitting. Lower = more robust but slower.
    subsample : float, default=0.8
        Fraction of samples to use for each tree. Helps prevent overfitting.
    colsample_bytree : float, default=0.8
        Fraction of features to use for each tree. Helps prevent overfitting.
    random_state : int, default=42
        Random state for reproducibility.
    scale_pos_weight : float | None, default=None
        Balancing of positive and negative weights. Use higher values (e.g., 5-10)
        for imbalanced data to give more importance to minority class.
        If None, no balancing is applied.

    Returns
    -------
    model : MultiOutputClassifier
        Multi-label classifier wrapping XGBClassifier.

    Raises
    ------
    ImportError
        If XGBoost is not installed.

    Examples
    --------
    >>> # Standard model
    >>> model = get_xgboost_model(n_estimators=200, learning_rate=0.05)
    >>> model.fit(X_train.to_numpy(), y_train.to_numpy())
    >>>
    >>> # Balanced model for imbalanced data
    >>> model = get_xgboost_model(n_estimators=200, scale_pos_weight=5)
    >>> model.fit(X_train.to_numpy(), y_train.to_numpy())
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost is not installed. Install with: uv add xgboost")

    base_model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
    )
    return MultiOutputClassifier(base_model, n_jobs=-1)


def predictions_to_labels(y_pred: np.ndarray, tags: list[str]) -> list[list[str]]:
    """Convert binary prediction matrix to list of tag names.

    Converts the binary prediction matrix (shape: n_samples x n_labels) back
    to human-readable tag names for each sample.

    Parameters
    ----------
    y_pred : np.ndarray
        Binary prediction matrix of shape (n_samples, n_labels).
        Values should be 0 or 1.
    tags : list[str]
        Ordered list of tag names corresponding to columns in y_pred.

    Returns
    -------
    predictions : list[list[str]]
        List of predicted tags for each sample.
        Each element is a list of tag names that were predicted (value=1).

    Examples
    --------
    >>> y_pred = np.array([
    ...     [1, 0, 1],  # Song 1: Rock, Upbeat
    ...     [0, 1, 0],  # Song 2: Jazz
    ...     [1, 1, 1],  # Song 3: Rock, Jazz, Upbeat
    ... ])
    >>> tags = ["Rock", "Jazz", "Upbeat"]
    >>> predictions = predictions_to_labels(y_pred, tags)
    >>> print(predictions)
    [['Rock', 'Upbeat'], ['Jazz'], ['Rock', 'Jazz', 'Upbeat']]
    """
    predictions = []
    for row in y_pred:
        # Get indices where prediction is 1
        predicted_indices = np.where(row == 1)[0]
        # Map indices to tag names
        predicted_tags = [tags[i] for i in predicted_indices]
        predictions.append(predicted_tags)
    return predictions


def predictions_to_dataframe(
    y_pred: np.ndarray, tags: list[str], song_ids: list[str] | None = None
) -> pl.DataFrame:
    """Convert binary predictions to a Polars DataFrame with song IDs and tags.

    Parameters
    ----------
    y_pred : np.ndarray
        Binary prediction matrix of shape (n_samples, n_labels).
    tags : list[str]
        Ordered list of tag names corresponding to columns in y_pred.
    song_ids : list[str] | None, optional
        List of song IDs. If None, will use integer indices.

    Returns
    -------
    predictions_df : pl.DataFrame
        DataFrame with columns:
        - song_id: Song identifier
        - predicted_tags: List of predicted tag names

    Examples
    --------
    >>> y_pred = model.predict(X_test.to_numpy())
    >>> song_ids = X_test_with_ids["song_id"].to_list()
    >>> predictions_df = predictions_to_dataframe(y_pred, genre_tags, song_ids)
    >>> print(predictions_df)
    shape: (100, 2)
    song_id      predicted_tags
    ---        ---
    str        list[str]
    123456      ["Rock", "Upbeat"]
    234567      ["Jazz", "Chill"]
    ...         ...
    """
    predictions = predictions_to_labels(y_pred, tags)

    if song_ids is None:
        song_ids = [str(i) for i in range(len(predictions))]

    return pl.DataFrame({"song_id": song_ids, "predicted_tags": predictions})


def predict_with_threshold(
    model,
    X: pl.DataFrame | np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict with custom decision threshold.

    Instead of using the default 0.5 threshold, you can adjust the threshold
    to control the precision/recall trade-off.

    Parameters
    ----------
    model : MultiOutputClassifier
        Trained multi-label classifier.
    X : pl.DataFrame or np.ndarray
        Features to predict on.
    threshold : float, default=0.5
        Decision threshold. Higher values = more conservative (higher precision,
        lower recall). Lower values = more aggressive (lower precision, higher recall).
        - threshold=0.3: More tags predicted (catches more, but more false alarms)
        - threshold=0.5: Default balanced threshold
        - threshold=0.7: Fewer tags predicted (more confident, misses some)

    Returns
    -------
    predictions : np.ndarray
        Binary prediction matrix (n_samples, n_labels).
    probabilities : np.ndarray
        Probability matrix (n_samples, n_labels).

    Examples
    --------
    >>> # Conservative predictions (high precision)
    >>> y_pred, probs = predict_with_threshold(model, X_test, threshold=0.7)
    >>>
    >>> # Aggressive predictions (high recall)
    >>> y_pred, probs = predict_with_threshold(model, X_test, threshold=0.3)
    >>>
    >>> # Use probabilities for further analysis
    >>> print(f"Average confidence: {probs.mean():.3f}")
    """
    # Convert to numpy if needed
    if isinstance(X, pl.DataFrame):
        X = X.to_numpy()

    # Get probabilities for each label
    # predict_proba returns a list of arrays, one per label
    # Each array has shape (n_samples, 2) for [prob_class_0, prob_class_1]
    probabilities_list = model.predict_proba(X)

    # Extract probability of positive class (class 1) for each label
    # Stack into matrix of shape (n_samples, n_labels)
    probabilities = np.column_stack([proba[:, 1] for proba in probabilities_list])

    # Apply threshold
    predictions = (probabilities >= threshold).astype(int)

    return predictions, probabilities


def find_optimal_thresholds(
    model,
    X_val: pl.DataFrame | np.ndarray,
    y_val: pl.DataFrame | np.ndarray,
    metric: str = "f1",
) -> np.ndarray:
    """Find optimal threshold for each label independently.

    Analyzes precision-recall curves to find the threshold that maximizes
    a specific metric for each label. Different labels may have different
    optimal thresholds.

    Parameters
    ----------
    model : MultiOutputClassifier
        Trained multi-label classifier.
    X_val : pl.DataFrame or np.ndarray
        Validation features.
    y_val : pl.DataFrame or np.ndarray
        True validation labels.
    metric : {'f1', 'precision', 'recall'}, default='f1'
        Which metric to optimize:
        - 'f1': Maximize F1-score (balanced precision/recall)
        - 'precision': Maximize precision (minimize false alarms)
        - 'recall': Maximize recall (catch as many as possible)

    Returns
    -------
    optimal_thresholds : np.ndarray
        Array of optimal thresholds, one per label.

    Examples
    --------
    >>> # Find F1-optimal thresholds on validation set
    >>> thresholds = find_optimal_thresholds(model, X_val, y_val, metric='f1')
    >>> print(f"Thresholds range: {thresholds.min():.3f} to {thresholds.max():.3f}")
    >>>
    >>> # Apply these thresholds to test set
    >>> _, probabilities = predict_with_threshold(model, X_test, threshold=0.5)
    >>> y_pred = (probabilities >= thresholds).astype(int)
    >>>
    >>> # Compare per-label thresholds
    >>> for tag, thresh in zip(tags, thresholds):
    ...     print(f"{tag:20s}: {thresh:.3f}")
    """
    # Convert to numpy if needed
    if isinstance(X_val, pl.DataFrame):
        X_val = X_val.to_numpy()
    if isinstance(y_val, pl.DataFrame):
        y_val = y_val.to_numpy()

    # Get probabilities
    probabilities_list = model.predict_proba(X_val)
    probabilities = np.column_stack([proba[:, 1] for proba in probabilities_list])

    optimal_thresholds = []

    for i in range(y_val.shape[1]):
        # Get precision-recall curve for this label
        precision, recall, thresholds = precision_recall_curve(
            y_val[:, i], probabilities[:, i]
        )

        # Calculate metric scores
        if metric == "f1":
            # F1 = 2 * (precision * recall) / (precision + recall)
            scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        elif metric == "precision":
            scores = precision
        elif metric == "recall":
            scores = recall
        else:
            raise ValueError(
                f"Unknown metric: {metric}. Choose from 'f1', 'precision', 'recall'"
            )

        # Find threshold that maximizes the metric
        optimal_idx = np.argmax(scores)

        # Handle edge case where optimal_idx is beyond thresholds array
        if optimal_idx < len(thresholds):
            optimal_thresholds.append(thresholds[optimal_idx])
        else:
            # Default to 0.5 if no clear optimum
            optimal_thresholds.append(0.5)

    return np.array(optimal_thresholds)


def evaluate_model(
    model,
    X_test: pl.DataFrame | np.ndarray,
    y_test: pl.DataFrame | np.ndarray,
    tags: list[str],
    verbose: bool = True,
    optimize_thresholds: bool = False,
    threshold_metric: str = "f1",
    X_train: pl.DataFrame | np.ndarray | None = None,
    y_train: pl.DataFrame | np.ndarray | None = None,
) -> dict:
    """Evaluate a trained multi-label classifier.

    Computes various metrics including per-label and overall performance.
    Optionally finds and applies optimal thresholds for each label.

    Parameters
    ----------
    model : MultiOutputClassifier
        Trained multi-label classifier.
    X_test : pl.DataFrame or np.ndarray
        Test features.
    y_test : pl.DataFrame or np.ndarray
        True labels (binary matrix).
    tags : list[str]
        List of tag names.
    verbose : bool, default=True
        If True, print evaluation results.
    optimize_thresholds : bool, default=False
        If True, find optimal thresholds for each label and apply them.
        Thresholds are found on the training set (X_train, y_train) and applied
        to the test set. This is a pragmatic approach that works well in practice.
    threshold_metric : {'f1', 'precision', 'recall'}, default='f1'
        Metric to optimize when finding thresholds (only used if optimize_thresholds=True).
    X_train : pl.DataFrame or np.ndarray, optional
        Training features for threshold optimization. Required if optimize_thresholds=True.
    y_train : pl.DataFrame or np.ndarray, optional
        Training labels for threshold optimization. Required if optimize_thresholds=True.

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'hamming_loss': Fraction of incorrectly predicted labels
        - 'exact_match_accuracy': Fraction of samples with all labels correct
        - 'per_label_metrics': DataFrame with precision, recall, F1 per label
        - 'predictions': Binary prediction matrix
        - 'optimal_thresholds': Array of optimal thresholds (if optimize_thresholds=True)
        - 'threshold_comparison': DataFrame comparing default vs optimal (if optimize_thresholds=True)

    Raises
    ------
    ValueError
        If optimize_thresholds=True but X_train or y_train is not provided.

    Examples
    --------
    >>> # Standard evaluation
    >>> model = get_random_forest_model()
    >>> model.fit(X_train.to_numpy(), y_train.to_numpy())
    >>> metrics = evaluate_model(model, X_test, y_test, genre_tags)
    >>>
    >>> # With threshold optimization
    >>> metrics = evaluate_model(
    ...     model, X_test, y_test, genre_tags,
    ...     optimize_thresholds=True,
    ...     threshold_metric='f1',
    ...     X_train=X_train,
    ...     y_train=y_train
    ... )
    >>> print(f"Optimal thresholds: {metrics['optimal_thresholds']}")
    >>> print(f"Improvement: {metrics['threshold_comparison']}")
    """
    # Convert to numpy if needed
    if isinstance(X_test, pl.DataFrame):
        X_test = X_test.to_numpy()
    if isinstance(y_test, pl.DataFrame):
        y_test_np = y_test.to_numpy()
    else:
        y_test_np = y_test

    # Make predictions with default threshold (0.5)
    y_pred = model.predict(X_test)

    # Compute metrics with default threshold
    hamming = hamming_loss(y_test_np, y_pred)
    exact_match = accuracy_score(y_test_np, y_pred)

    # Per-label metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test_np, y_pred, average=None, zero_division=0
    )

    per_label_df = pl.DataFrame(
        {
            "tag": tags,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        }
    ).sort("f1_score", descending=True)

    # Macro-averaged metrics (all labels weighted equally)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test_np, y_pred, average="macro", zero_division=0
    )

    # Weighted-averaged metrics (labels weighted by support)
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_test_np, y_pred, average="weighted", zero_division=0
        )
    )

    # Threshold optimization (if requested)
    optimal_thresholds = None
    threshold_comparison = None
    if optimize_thresholds:
        # Validate that training set is provided
        if X_train is None or y_train is None:
            raise ValueError(
                "When optimize_thresholds=True, you must provide X_train and y_train.\n"
                "The thresholds will be tuned on the training set and applied to the test set.\n\n"
                "Correct usage:\n"
                "  metrics = evaluate_model(model, X_test, y_test, tags,\n"
                "                          optimize_thresholds=True,\n"
                "                          X_train=X_train, y_train=y_train)"
            )

        # Convert training set to numpy if needed
        if isinstance(X_train, pl.DataFrame):
            X_train_np = X_train.to_numpy()
        else:
            X_train_np = X_train

        if isinstance(y_train, pl.DataFrame):
            y_train_np = y_train.to_numpy()
        else:
            y_train_np = y_train

        # Find optimal thresholds on TRAINING set
        optimal_thresholds = find_optimal_thresholds(
            model, X_train_np, y_train_np, metric=threshold_metric
        )

        # Get predictions with optimal thresholds on TEST set
        _, probabilities = predict_with_threshold(model, X_test, threshold=0.5)
        y_pred_optimized = (probabilities >= optimal_thresholds).astype(int)

        # Compute metrics with optimized thresholds
        opt_precision, opt_recall, opt_f1, _ = precision_recall_fscore_support(
            y_test_np, y_pred_optimized, average=None, zero_division=0
        )

        opt_macro_precision, opt_macro_recall, opt_macro_f1, _ = (
            precision_recall_fscore_support(
                y_test_np, y_pred_optimized, average="macro", zero_division=0
            )
        )

        opt_weighted_precision, opt_weighted_recall, opt_weighted_f1, _ = (
            precision_recall_fscore_support(
                y_test_np, y_pred_optimized, average="weighted", zero_division=0
            )
        )

        # Create comparison DataFrame
        threshold_comparison = pl.DataFrame(
            {
                "tag": tags,
                "threshold": optimal_thresholds,
                "default_f1": f1,
                "optimized_f1": opt_f1,
                "f1_improvement": opt_f1 - f1,
            }
        ).sort("f1_improvement", descending=True)

        # Update predictions to use optimized thresholds
        y_pred = y_pred_optimized
        per_label_df = pl.DataFrame(
            {
                "tag": tags,
                "precision": opt_precision,
                "recall": opt_recall,
                "f1_score": opt_f1,
                "support": support,
            }
        ).sort("f1_score", descending=True)
        macro_precision, macro_recall, macro_f1 = (
            opt_macro_precision,
            opt_macro_recall,
            opt_macro_f1,
        )
        weighted_precision, weighted_recall, weighted_f1 = (
            opt_weighted_precision,
            opt_weighted_recall,
            opt_weighted_f1,
        )

    if verbose:
        print(f"\n{'=' * 70}")
        print("MODEL EVALUATION RESULTS")
        if optimize_thresholds:
            print(f" (Using optimized thresholds for {threshold_metric})")
        print(f"{'=' * 70}")
        print(f"Hamming Loss: {hamming:.4f}")
        print("  (Average fraction of labels incorrectly predicted per sample)")
        print(f"\nExact Match Accuracy: {exact_match:.4f}")
        print("  (Fraction of samples with ALL labels predicted correctly)")
        print("\nMacro-Averaged Metrics:")
        print("  (All labels weighted equally)")
        print(f"  Precision: {macro_precision:.4f}")
        print(f"  Recall:    {macro_recall:.4f}")
        print(f"  F1-Score:  {macro_f1:.4f}")
        print("\nWeighted-Averaged Metrics:")
        print("  (Labels weighted by number of samples)")
        print(f"  Precision: {weighted_precision:.4f}")
        print(f"  Recall:    {weighted_recall:.4f}")
        print(f"  F1-Score:  {weighted_f1:.4f}")

        if (
            optimize_thresholds
            and threshold_comparison is not None
            and optimal_thresholds is not None
        ):
            print("\nThreshold Optimization Results:")
            print(f"  Optimized for: {threshold_metric}")
            print("  Thresholds found on: training set")
            print("  Thresholds applied to: test set")
            print(
                f"  Threshold range: {optimal_thresholds.min():.3f} - {optimal_thresholds.max():.3f}"
            )
            print("\nPer-Label Threshold Comparison (sorted by improvement):")
            print(threshold_comparison)

        print("\nPer-Label Metrics (sorted by F1-score):")
        print(per_label_df)
        print(f"{'=' * 70}\n")

    result = {
        "hamming_loss": hamming,
        "exact_match_accuracy": exact_match,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_label_metrics": per_label_df,
        "predictions": y_pred,
    }

    # Add threshold optimization results if available
    if optimize_thresholds:
        result["optimal_thresholds"] = optimal_thresholds
        result["threshold_comparison"] = threshold_comparison

    return result


def train_and_compare_models(
    models_dict: dict,
    X_train: pl.DataFrame | np.ndarray,
    y_train: pl.DataFrame | np.ndarray,
    X_test: pl.DataFrame | np.ndarray,
    y_test: pl.DataFrame | np.ndarray,
    tags: list[str],
) -> pl.DataFrame:
    """Train and compare multiple models.

    Parameters
    ----------
    models_dict : dict
        Dictionary mapping model names to model instances.
        Example: {"Linear": get_linear_model(), "RF": get_random_forest_model()}
    X_train, y_train : pl.DataFrame or np.ndarray
        Training data.
    X_test, y_test : pl.DataFrame or np.ndarray
        Test data.
    tags : list[str]
        List of tag names.

    Returns
    -------
    comparison_df : pl.DataFrame
        DataFrame comparing all models with metrics.

    Examples
    --------
    >>> models = {
    ...     "Linear": get_linear_model(),
    ...     "KNN": get_knn_model(n_neighbors=10),
    ...     "Decision Tree": get_decision_tree_model(max_depth=15),
    ...     "Random Forest": get_random_forest_model(n_estimators=100),
    ...     "XGBoost": get_xgboost_model(n_estimators=100),
    ... }
    >>> comparison = compare_models(models, X_train, y_train, X_test, y_test, genre_tags)
    >>> print(comparison.sort("macro_f1", descending=True))
    """
    # Convert to numpy if needed
    if isinstance(X_train, pl.DataFrame):
        X_train = X_train.to_numpy()
    if isinstance(y_train, pl.DataFrame):
        y_train = y_train.to_numpy()
    if isinstance(X_test, pl.DataFrame):
        X_test = X_test.to_numpy()
    if isinstance(y_test, pl.DataFrame):
        y_test = y_test.to_numpy()

    results = []

    for model_name, model in models_dict.items():
        print(f"\n{'=' * 70}")
        print(f"Training: {model_name}")
        print(f"{'=' * 70}")

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, tags, verbose=False)

        results.append(
            {
                "model": model_name,
                "hamming_loss": metrics["hamming_loss"],
                "exact_match_accuracy": metrics["exact_match_accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_precision": metrics["weighted_precision"],
                "weighted_recall": metrics["weighted_recall"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )

        print(f" {model_name} trained successfully")
        print(
            f"  Hamming Loss: {metrics['hamming_loss']:.4f}, "
            f"Exact Match: {metrics['exact_match_accuracy']:.4f}, "
            f"Macro F1: {metrics['macro_f1']:.4f}, "
            f"Weighted F1: {metrics['weighted_f1']:.4f}"
        )

    comparison_df = pl.DataFrame(results).sort("macro_f1", descending=True)

    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print("\nSorted by Macro F1-Score:")
    print(comparison_df)
    print(f"{'=' * 70}\n")

    return comparison_df
