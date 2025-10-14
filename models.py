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


def get_linear_model(C=1.0, max_iter=1000, random_state=42):
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

    Returns
    -------
    model : MultiOutputClassifier
        Multi-label classifier wrapping LogisticRegression.

    Examples
    --------
    >>> model = get_linear_model(C=0.5)
    >>> model.fit(X_train.to_numpy(), y_train.to_numpy())
    >>> y_pred = model.predict(X_test.to_numpy())
    """
    base_model = LogisticRegression(
        C=C, max_iter=max_iter, random_state=random_state, n_jobs=-1
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
    max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42
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

    Returns
    -------
    model : MultiOutputClassifier
        Multi-label classifier wrapping DecisionTreeClassifier.

    Examples
    --------
    >>> model = get_decision_tree_model(max_depth=15, min_samples_split=20)
    >>> model.fit(X_train.to_numpy(), y_train.to_numpy())
    >>> y_pred = model.predict(X_test.to_numpy())
    >>>
    >>> # Access individual trees for each label
    >>> for i, estimator in enumerate(model.estimators_):
    ...     print(f"Label {i}: Tree depth = {estimator.get_depth()}")
    """
    base_model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
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
    >>> model = get_xgboost_model(n_estimators=200, learning_rate=0.05)
    >>> model.fit(X_train.to_numpy(), y_train.to_numpy())
    >>> y_pred = model.predict(X_test.to_numpy())
    >>>
    >>> # Get feature importances
    >>> for i, estimator in enumerate(model.estimators_):
    ...     importances = estimator.feature_importances_
    ...     print(f"Label {i}: Top feature importance = {importances.max():.3f}")
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")

    base_model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
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


def evaluate_model(
    model,
    X_test: pl.DataFrame | np.ndarray,
    y_test: pl.DataFrame | np.ndarray,
    tags: list[str],
    verbose: bool = True,
) -> dict:
    """Evaluate a trained multi-label classifier.

    Computes various metrics including per-label and overall performance.

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

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'hamming_loss': Fraction of incorrectly predicted labels
        - 'exact_match_accuracy': Fraction of samples with all labels correct
        - 'per_label_metrics': DataFrame with precision, recall, F1 per label
        - 'predictions': Binary prediction matrix

    Examples
    --------
    >>> model = get_random_forest_model()
    >>> model.fit(X_train.to_numpy(), y_train.to_numpy())
    >>> metrics = evaluate_model(model, X_test, y_test, genre_tags)
    >>>
    >>> # Access specific metrics
    >>> print(f"Hamming Loss: {metrics['hamming_loss']:.3f}")
    >>> print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.3f}")
    >>> print(metrics['per_label_metrics'])
    """
    # Convert to numpy if needed
    if isinstance(X_test, pl.DataFrame):
        X_test = X_test.to_numpy()
    if isinstance(y_test, pl.DataFrame):
        y_test_np = y_test.to_numpy()
    else:
        y_test_np = y_test

    # Make predictions
    y_pred = model.predict(X_test)

    # Compute metrics
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

    # Average metrics
    avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
        y_test_np, y_pred, average="macro", zero_division=0
    )

    if verbose:
        print(f"\n{'=' * 70}")
        print("MODEL EVALUATION RESULTS")
        print(f"{'=' * 70}")
        print(f"Hamming Loss: {hamming:.4f}")
        print("  (Average fraction of labels incorrectly predicted per sample)")
        print(f"\nExact Match Accuracy: {exact_match:.4f}")
        print("  (Fraction of samples with ALL labels predicted correctly)")
        print("\nMacro-Averaged Metrics:")
        print(f"  Precision: {avg_precision:.4f}")
        print(f"  Recall:    {avg_recall:.4f}")
        print(f"  F1-Score:  {avg_f1:.4f}")
        print("\nPer-Label Metrics (sorted by F1-score):")
        print(per_label_df)
        print(f"{'=' * 70}\n")

    return {
        "hamming_loss": hamming,
        "exact_match_accuracy": exact_match,
        "macro_precision": avg_precision,
        "macro_recall": avg_recall,
        "macro_f1": avg_f1,
        "per_label_metrics": per_label_df,
        "predictions": y_pred,
    }


def compare_models(
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
            }
        )

        print(f" {model_name} trained successfully")
        print(
            f"  Hamming Loss: {metrics['hamming_loss']:.4f}, "
            f"Exact Match: {metrics['exact_match_accuracy']:.4f}, "
            f"F1: {metrics['macro_f1']:.4f}"
        )

    comparison_df = pl.DataFrame(results).sort("macro_f1", descending=True)

    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(comparison_df)
    print(f"{'=' * 70}\n")

    return comparison_df
