"""Preprocessing utilities for music tagging ML pipelines.

This module provides sklearn-compatible transformers for feature preprocessing
including normalization and dimensionality reduction via PCA.

MULTI-LABEL CLASSIFICATION IMPORTANT NOTES:
============================================

For multi-label classification (where each song can have multiple tags), the
preprocessing workflow is different from single-label classification:

1. DATA STRUCTURE:
   - Raw data: Song-tag pairs (one row per song-tag combination)
   - After prep: One row per song with binary label matrix

   Example transformation:
   FROM (song-tag pairs):
       song_id | tag_name | features...
       123     | Rock     | 0.5, 0.2, ...
       123     | Upbeat   | 0.5, 0.2, ...
       456     | Jazz     | 0.1, 0.8, ...

   TO (multi-label format):
       X (features):           y (binary labels):
       song_id | features...   Rock | Upbeat | Jazz
       123     | 0.5, 0.2...   1    | 1      | 0
       456     | 0.1, 0.8...   0    | 0      | 1

2. PREPROCESSING ORDER:
   a) Use prepare_multilabel_data() to convert song-tag pairs to song-level data
   b) Split into train/test at the SONG level (not tag level)
   c) Fit scaler/PCA on training songs only
   d) Transform both train and test sets

3. AVOIDING DATA LEAKAGE:
   - Must deduplicate songs before fitting scaler/PCA
   - Never fit preprocessing on song-tag pairs directly
   - Always fit on unique songs in training set only

4. TRAINING:
   - Use sklearn.multioutput.MultiOutputClassifier for classifiers that
     don't natively support multi-label (e.g., DecisionTree, RandomForest)
   - Or use classifiers with native multi-label support (e.g.,
     MultiOutputClassifier, ClassifierChain)

Usage examples
```python
from processing import PolarsStandardScaler, PolarsPCA, create_preprocessing_pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Option 1: Use individual transformers
scaler = PolarsStandardScaler()
X_scaled = scaler.fit_transform(X_train)

pca = PolarsPCA(n_components=0.95)  # Keep 95% variance
X_reduced = pca.fit_transform(X_scaled)

# Option 2: Use in sklearn pipeline
pipeline = Pipeline([
    ("scaler", PolarsStandardScaler()),
    ("pca", PolarsPCA(n_components=50)),
    ("classifier", LogisticRegression())
])
pipeline.fit(X_train, y_train)

# Option 3: Use convenience function
steps = create_preprocessing_pipeline(
    use_pca=True,
    n_components=0.95,
    normalize=True
)
pipeline = Pipeline(steps + [("classifier", LogisticRegression())])

# MULTI-LABEL WORKFLOW EXAMPLE:
from processing import prepare_multilabel_data, PolarsStandardScaler, PolarsPCA
from utils import get_base_dataset
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier

# 1. Get base dataset and features
base_df = get_base_dataset(db, min_tag_count=10)
features_df = pl.read_parquet("song_features.parquet")

# 2. Prepare multi-label data (song-level, not song-tag pairs)
X, y, tags = prepare_multilabel_data(base_df, features_df, "Genre")

# 3. Split at SONG level
X_train, X_test, y_train, y_test = train_test_split(
    X.drop("song_id"), y, test_size=0.3, random_state=42
)

# 4. Fit preprocessing on training data only
scaler = PolarsStandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Optional: Apply PCA
pca = PolarsPCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 6. Train multi-label classifier
clf = MultiOutputClassifier(DecisionTreeClassifier(max_depth=10))
clf.fit(X_train_pca.to_numpy(), y_train.to_numpy())

# 7. Predict and evaluate
y_pred = clf.predict(X_test_pca.to_numpy())
```
"""

from dataclasses import dataclass

import polars as pl
import pyrekordbox
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from typing import Union, Any


class PolarsStandardScaler(BaseEstimator, TransformerMixin):
    """StandardScaler that works with Polars DataFrames.

    Normalizes features by removing the mean and scaling to unit variance.
    This is essential for gradient descent-based models like logistic regression,
    neural networks, and SVM.

    The standard score of a sample x is calculated as:
        z = (x - u) / s
    where u is the mean of the training samples and s is the standard deviation.

    Parameters
    ----------
    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.
    with_mean : bool, default=True
        If True, center the data before scaling.
    with_std : bool, default=True
        If True, scale the data to unit variance.

    Attributes
    ----------
    scaler_ : StandardScaler
        The underlying sklearn StandardScaler instance.
    feature_names_ : list[str]
        Names of features seen during fit.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> import polars as pl
    >>> from processing import PolarsStandardScaler
    >>>
    >>> # Create sample data
    >>> df = pl.DataFrame({
    ...     "feature1": [1.0, 2.0, 3.0, 4.0],
    ...     "feature2": [10.0, 20.0, 30.0, 40.0]
    ... })
    >>>
    >>> # Fit and transform
    >>> scaler = PolarsStandardScaler()
    >>> scaled_df = scaler.fit_transform(df)
    >>> print(scaled_df)
    >>>
    >>> # Use in sklearn pipeline
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model import LogisticRegression
    >>>
    >>> pipeline = Pipeline([
    ...     ("scaler", PolarsStandardScaler()),
    ...     ("classifier", LogisticRegression())
    ... ])
    """

    def __init__(
        self, copy: bool = True, with_mean: bool = True, with_std: bool = True
    ):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X: pl.DataFrame, y=None) -> "PolarsStandardScaler":
        """Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : pl.DataFrame
            Training data to fit the scaler.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : PolarsStandardScaler
            Fitted scaler.
        """
        self.feature_names_ = X.columns
        self.n_features_in_ = len(X.columns)

        # Initialize and fit sklearn's StandardScaler
        self.scaler_ = StandardScaler(
            copy=self.copy, with_mean=self.with_mean, with_std=self.with_std
        )
        self.scaler_.fit(X.to_numpy())

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Scale features of X according to feature_range.

        Parameters
        ----------
        X : pl.DataFrame
            Data to transform.

        Returns
        -------
        X_scaled : pl.DataFrame
            Transformed data.
        """
        if not hasattr(self, "scaler_"):
            raise ValueError(
                "This PolarsStandardScaler instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using 'transform'."
            )

        X_scaled = self.scaler_.transform(X.to_numpy())
        return pl.DataFrame(X_scaled, schema=X.columns)

    def inverse_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Scale back the data to the original representation.

        Parameters
        ----------
        X : pl.DataFrame
            Data to inverse transform.

        Returns
        -------
        X_original : pl.DataFrame
            Data in original scale.
        """
        if not hasattr(self, "scaler_"):
            raise ValueError(
                "This PolarsStandardScaler instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using 'inverse_transform'."
            )

        X_original = self.scaler_.inverse_transform(X.to_numpy())
        return pl.DataFrame(X_original, schema=X.columns)


class PolarsPCA(BaseEstimator, TransformerMixin):
    """Principal Component Analysis (PCA) that works with Polars DataFrames.

    Linear dimensionality reduction using Singular Value Decomposition (SVD)
    to project the data to a lower dimensional space. Useful for:
    - Reducing feature dimensionality
    - Removing multicollinearity
    - Noise reduction
    - Visualization (2D/3D projections)

    Parameters
    ----------
    n_components : int, float, or None, default=None
        Number of components to keep:
        - If int: number of components to keep
        - If float (0 < n_components < 1): select number of components such that
          the amount of variance explained is greater than the percentage specified
        - If None: keep all components (min(n_samples, n_features))
    whiten : bool, default=False
        When True, the components_ vectors are multiplied by sqrt(n_samples) and
        divided by the singular values to ensure uncorrelated outputs with unit
        component-wise variances. Whitening removes some information but can
        improve predictive accuracy of downstream classifiers.
    random_state : int, RandomState instance or None, default=None
        Used when the 'arpack' or 'randomized' solvers are used.

    Attributes
    ----------
    pca_ : PCA
        The underlying sklearn PCA instance.
    feature_names_in_ : list[str]
        Names of features seen during fit.
    n_features_in_ : int
        Number of features seen during fit.
    n_components_ : int
        Number of components kept.
    explained_variance_ : ndarray
        The amount of variance explained by each of the selected components.
    explained_variance_ratio_ : ndarray
        Percentage of variance explained by each of the selected components.

    Examples
    --------
    >>> import polars as pl
    >>> from processing import PolarsPCA
    >>>
    >>> # Create sample data with many features
    >>> df = pl.DataFrame({
    ...     "feature1": [1.0, 2.0, 3.0, 4.0],
    ...     "feature2": [10.0, 20.0, 30.0, 40.0],
    ...     "feature3": [100.0, 200.0, 300.0, 400.0]
    ... })
    >>>
    >>> # Reduce to 2 components
    >>> pca = PolarsPCA(n_components=2)
    >>> reduced_df = pca.fit_transform(df)
    >>> print(f"Original shape: {df.shape}")
    >>> print(f"Reduced shape: {reduced_df.shape}")
    >>> print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    >>>
    >>> # Keep 95% of variance
    >>> pca = PolarsPCA(n_components=0.95)
    >>> reduced_df = pca.fit_transform(df)
    >>>
    >>> # Use in sklearn pipeline
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model import LogisticRegression
    >>>
    >>> pipeline = Pipeline([
    ...     ("pca", PolarsPCA(n_components=0.95)),
    ...     ("classifier", LogisticRegression())
    ... ])
    """

    def __init__(
        self,
        n_components: int | float | None = None,
        whiten: bool = False,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state

    def fit(self, X: pl.DataFrame, y=None) -> "PolarsPCA":
        """Fit the PCA model with X.

        Parameters
        ----------
        X : pl.DataFrame
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : PolarsPCA
            Fitted PCA instance.
        """
        self.feature_names_in_ = X.columns
        self.n_features_in_ = len(X.columns)

        # Initialize and fit sklearn's PCA
        self.pca_ = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            random_state=self.random_state,
        )
        self.pca_.fit(X.to_numpy())

        # Expose useful attributes
        self.n_components_ = self.pca_.n_components_
        self.explained_variance_ = self.pca_.explained_variance_
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_

        print(
            f"\nPCA fitted: {self.n_features_in_} features -> {self.n_components_} components"
        )
        print(
            f"Explained variance ratio: {self.explained_variance_ratio_[:5]}"
        )  # Show first 5
        print(f"Total variance explained: {self.explained_variance_ratio_.sum():.4f}\n")

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Apply dimensionality reduction to X.

        Parameters
        ----------
        X : pl.DataFrame
            Data to transform.

        Returns
        -------
        X_transformed : pl.DataFrame
            Transformed data with reduced dimensionality.
        """
        if not hasattr(self, "pca_"):
            raise ValueError(
                "This PolarsPCA instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using 'transform'."
            )

        X_transformed = self.pca_.transform(X.to_numpy())

        # Generate column names for principal components
        component_names = [f"PC{i + 1}" for i in range(X_transformed.shape[1])]

        return pl.DataFrame(X_transformed, schema=component_names)

    def inverse_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform data back to its original space.

        Parameters
        ----------
        X : pl.DataFrame
            Data in PCA space.

        Returns
        -------
        X_original : pl.DataFrame
            Data transformed back to original space.
        """
        if not hasattr(self, "pca_"):
            raise ValueError(
                "This PolarsPCA instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using 'inverse_transform'."
            )

        X_original = self.pca_.inverse_transform(X.to_numpy())
        return pl.DataFrame(X_original, schema=self.feature_names_in_)


def create_preprocessing_pipeline(
    use_pca: bool = False,
    n_components: int | float | None = None,
    normalize: bool = True,
) -> list:
    """Create a list of preprocessing steps for sklearn Pipeline.

    This is a convenience function to quickly create common preprocessing
    configurations for your training pipelines.

    Parameters
    ----------
    use_pca : bool, default=False
        Whether to include PCA dimensionality reduction.
    n_components : int, float, or None, default=None
        Number of PCA components. Only used if use_pca=True.
        - If int: number of components to keep
        - If float: percentage of variance to preserve (e.g., 0.95 for 95%)
        - If None: keep all components
    normalize : bool, default=True
        Whether to normalize features using StandardScaler.

    Returns
    -------
    steps : list
        List of (name, transformer) tuples for sklearn Pipeline.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model import LogisticRegression
    >>> from processing import create_preprocessing_pipeline
    >>>
    >>> # Create pipeline with normalization only
    >>> steps = create_preprocessing_pipeline(normalize=True)
    >>> pipeline = Pipeline(steps + [("classifier", LogisticRegression())])
    >>>
    >>> # Create pipeline with PCA (95% variance) and normalization
    >>> steps = create_preprocessing_pipeline(
    ...     use_pca=True,
    ...     n_components=0.95,
    ...     normalize=True
    ... )
    >>> pipeline = Pipeline(steps + [("classifier", LogisticRegression())])
    >>>
    >>> # Create pipeline with PCA (50 components) only
    >>> steps = create_preprocessing_pipeline(
    ...     use_pca=True,
    ...     n_components=50,
    ...     normalize=False
    ... )
    >>> pipeline = Pipeline(steps + [("classifier", LogisticRegression())])
    """
    steps = []

    if normalize:
        steps.append(("scaler", PolarsStandardScaler()))

    if use_pca:
        steps.append(("pca", PolarsPCA(n_components=n_components)))

    return steps


def prepare_multilabel_data(
    base_df: pl.DataFrame,
    features_df: pl.DataFrame,
    tag_group: str,
    feature_columns: list[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Prepare data for multi-label classification from song-tag pairs.

    This function handles the key difference for multi-label data:
    - Features are aggregated at the SONG level (one row per unique song)
    - Labels are converted to binary matrix format (one column per tag)

    IMPORTANT: In multi-label classification, each song can have multiple tags.
    The dataset must be structured with:
    - X: One row per song (not per song-tag pair)
    - y: Binary matrix with one column per tag (1 if song has tag, 0 otherwise)

    Parameters
    ----------
    base_df : pl.DataFrame
        Base dataset from get_base_dataset() with song-tag pairs.
        Must include columns: song_id, tag_name, tag_group
    features_df : pl.DataFrame
        Audio features dataframe with song_id and feature columns.
    tag_group : str
        The tag group to prepare data for (e.g., "Genre", "Mood", "Situation").
    feature_columns : list[str] | None, optional
        List of feature column names to use. If None, all columns except
        song_id and song_path will be used as features.

    Returns
    -------
    X : pl.DataFrame
        Feature matrix with one row per unique song.
        Columns: song_id + all feature columns
    y : pl.DataFrame
        Binary label matrix with one row per song and one column per tag.
        Column names are the tag names. Values are 1 (has tag) or 0 (no tag).
    tags : list[str]
        Ordered list of tag names corresponding to columns in y.

    Examples
    --------
    >>> from utils import get_base_dataset
    >>> from processing import prepare_multilabel_data
    >>>
    >>> # Get base dataset and features
    >>> base_df = get_base_dataset(db, min_tag_count=10)
    >>> features_df = pl.read_parquet("song_features.parquet")
    >>>
    >>> # Prepare data for Genre classification
    >>> X, y, tags = prepare_multilabel_data(
    ...     base_df=base_df,
    ...     features_df=features_df,
    ...     tag_group="Genre"
    ... )
    >>>
    >>> print(f"Songs: {len(X)}")
    >>> print(f"Features: {len(X.columns) - 1}")  # -1 for song_id
    >>> print(f"Tags: {len(tags)}")
    >>> print(f"y shape: {y.shape}")
    >>>
    >>> # Now split and train
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.multioutput import MultiOutputClassifier
    >>> from sklearn.tree import DecisionTreeClassifier
    >>>
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X.drop("song_id"), y, test_size=0.3, random_state=42
    ... )
    >>>
    >>> clf = MultiOutputClassifier(DecisionTreeClassifier())
    >>> clf.fit(X_train.to_numpy(), y_train.to_numpy())
    """
    # Filter to the specific tag group
    group_df = base_df.filter(pl.col("tag_group") == tag_group)

    if len(group_df) == 0:
        raise ValueError(f"No tags found for tag_group='{tag_group}'")

    # Get unique songs and their tags
    song_tags = (
        group_df.group_by("song_id")
        .agg(pl.col("tag_name").alias("tags"))
        .sort("song_id")
    )

    # Get all unique tags for this group
    tags = sorted(group_df["tag_name"].unique().to_list())

    print(f"\nPreparing multi-label data for '{tag_group}':")
    print(f"{'=' * 60}")
    print(f"Unique songs: {len(song_tags)}")
    print(f"Unique tags: {len(tags)}")
    print(f"Tags: {', '.join(tags)}")
    print(f"{'=' * 60}\n")

    # Create binary label matrix
    # For each song, create a row with 1 if tag is present, 0 otherwise
    label_data = []
    for row in song_tags.iter_rows(named=True):
        song_id = row["song_id"]
        song_tag_list = row["tags"]
        # Create binary vector: 1 if tag in song_tag_list, else 0
        label_row = {"song_id": song_id}
        for tag in tags:
            label_row[tag] = 1 if tag in song_tag_list else 0
        label_data.append(label_row)

    y = pl.DataFrame(label_data)

    # Prepare feature matrix (one row per unique song)
    if feature_columns is None:
        # Use all columns except song_id and song_path
        feature_columns = [
            col for col in features_df.columns if col not in ["song_id", "song_path"]
        ]

    X = (
        features_df.select(["song_id"] + feature_columns)
        .join(y.select("song_id"), on="song_id", how="inner")
        .sort("song_id")
    )

    # Ensure X and y have same song order
    y = y.join(X.select("song_id"), on="song_id", how="inner").drop("song_id")

    # Calculate average tags per song
    avg_tags = y.select(pl.sum_horizontal(pl.all()).mean()).item()

    print("Final dataset shape:")
    print(f"  X: {X.shape} (song_id + {len(feature_columns)} features)")
    print(f"  y: {y.shape} ({len(tags)} binary labels)")
    print(f"  Average tags per song: {avg_tags:.2f}\n")

    return X, y, tags


def get_multilabel_stats(y: pl.DataFrame, tags: list[str]) -> pl.DataFrame:
    """Calculate statistics for multi-label dataset.

    Parameters
    ----------
    y : pl.DataFrame
        Binary label matrix from prepare_multilabel_data()
    tags : list[str]
        List of tag names corresponding to columns in y

    Returns
    -------
    stats : pl.DataFrame
        DataFrame with columns:
        - tag: Tag name
        - count: Number of songs with this tag
        - percentage: Percentage of songs with this tag

    Examples
    --------
    >>> X, y, tags = prepare_multilabel_data(base_df, features_df, "Genre")
    >>> stats = get_multilabel_stats(y, tags)
    >>> print(stats)
    """
    stats_data = []
    total_songs = len(y)

    for tag in tags:
        count = y[tag].sum()
        percentage = (count / total_songs) * 100
        stats_data.append({"tag": tag, "count": count, "percentage": percentage})

    return pl.DataFrame(stats_data).sort("count", descending=True)


def filter_rare_labels(
    X_train: pl.DataFrame,
    y_train: pl.DataFrame,
    X_test: pl.DataFrame,
    y_test: pl.DataFrame,
    tags: list[str],
    min_count: int = 5,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, list[str]]:
    """Filter out labels that have fewer than min_count occurrences in training set.

    IMPORTANT: This should be called AFTER train/test split to avoid data leakage.
    Tags are filtered based on their occurrence in the training set only, then
    the same filtering is applied to the test set.

    Parameters
    ----------
    X_train : pl.DataFrame
        Training features
    y_train : pl.DataFrame
        Training labels (binary matrix)
    X_test : pl.DataFrame
        Test features
    y_test : pl.DataFrame
        Test labels (binary matrix)
    tags : list[str]
        List of tag names corresponding to columns in y
    min_count : int, default=5
        Minimum number of occurrences required in training set

    Returns
    -------
    X_train : pl.DataFrame
        Training features (unchanged)
    y_train : pl.DataFrame
        Training labels with rare tags removed
    X_test : pl.DataFrame
        Test features (unchanged)
    y_test : pl.DataFrame
        Test labels with rare tags removed
    filtered_tags : list[str]
        List of tag names after filtering

    Examples
    --------
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> X_train, y_train, X_test, y_test, tags = filter_rare_labels(
    ...     X_train, y_train, X_test, y_test, tags, min_count=10
    ... )
    """
    # Calculate counts in training set
    train_counts = []
    for tag in tags:
        count = y_train[tag].sum()
        train_counts.append({"tag": tag, "count": count})

    counts_df = pl.DataFrame(train_counts)

    # Find tags below threshold
    rare_tags = counts_df.filter(pl.col("count") < min_count)
    valid_tags = counts_df.filter(pl.col("count") >= min_count)["tag"].to_list()

    if len(rare_tags) > 0:
        print(
            f"\nFiltering rare labels with < {min_count} occurrences in training set:"
        )
        print(f"{'=' * 60}")
        for row in rare_tags.iter_rows(named=True):
            print(f"  - {row['tag']}: {row['count']} occurrence(s)")
        print(f"{'=' * 60}")
        print(f"Removed {len(rare_tags)} tags, kept {len(valid_tags)} tags\n")

        # Filter both train and test to keep only valid tags
        y_train = y_train.select(valid_tags)
        y_test = y_test.select(valid_tags)
    else:
        print(f"All {len(tags)} tags meet the minimum count threshold ({min_count})\n")

    return X_train, y_train, X_test, y_test, valid_tags


@dataclass
class ProcessingResult:
    """Result container for preprocess_tag_group function.

    This dataclass provides a clean, typed interface for accessing
    preprocessing results with named attributes instead of messy tuple unpacking.

    Attributes
    ----------
    X_train : pl.DataFrame
        Preprocessed training features
    X_val : pl.DataFrame | None
        Preprocessed validation features (None if val_size=0)
    X_test : pl.DataFrame
        Preprocessed test features
    y_train : pl.DataFrame
        Training labels (binary matrix)
    y_val : pl.DataFrame | None
        Validation labels (binary matrix, None if val_size=0)
    y_test : pl.DataFrame
        Test labels (binary matrix)
    tags : list[str]
        List of tag names after filtering
    song_ids_train : pl.DataFrame
        Song IDs for training set
    song_ids_val : pl.DataFrame | None
        Song IDs for validation set (None if val_size=0)
    song_ids_test : pl.DataFrame
        Song IDs for test set
    scaler : PolarsStandardScaler | None
        Fitted scaler if apply_scaling=True, else None
    pca : PolarsPCA | None
        Fitted PCA if apply_pca=True, else None

    Examples
    --------
    >>> from processing import preprocess_tag_group
    >>>
    >>> # With validation set
    >>> result = preprocess_tag_group(
    ...     base_df, features_df, "Genre",
    ...     test_size=0.2, val_size=0.2,
    ...     apply_scaling=True,
    ...     apply_pca=True
    ... )
    >>>
    >>> # Access attributes by name (much cleaner!)
    >>> print(result.X_train.shape)
    >>> print(result.X_val.shape)
    >>> print(result.tags)
    >>> model.fit(result.X_train.to_numpy(), result.y_train.to_numpy())
    >>>
    >>> # Use the scaler for inference
    >>> if result.scaler:
    ...     new_features_scaled = result.scaler.transform(new_features)
    >>>
    >>> # Still supports tuple unpacking if needed (backward compatibility)
    >>> X_train, X_val, X_test, y_train, y_val, y_test, tags, *_ = result
    """

    X_train: "pl.DataFrame"
    X_val: "pl.DataFrame | None"
    X_test: "pl.DataFrame"
    y_train: "pl.DataFrame"
    y_val: "pl.DataFrame | None"
    y_test: "pl.DataFrame"
    tags: list[str]
    song_ids_train: "pl.DataFrame"
    song_ids_val: "pl.DataFrame | None"
    song_ids_test: "pl.DataFrame"
    scaler: "PolarsStandardScaler | None"
    pca: "PolarsPCA | None"

    def __iter__(self):
        """Allow tuple unpacking for backward compatibility."""
        return iter(
            (
                self.X_train,
                self.X_val,
                self.X_test,
                self.y_train,
                self.y_val,
                self.y_test,
                self.tags,
                self.song_ids_train,
                self.song_ids_val,
                self.song_ids_test,
                self.scaler,
                self.pca,
            )
        )


def preprocess_tag_group(
    base_df,
    features_df,
    tag_group,
    feature_cols=None,
    test_size=0.2,
    val_size=0.0,
    random_state=42,
    min_train_count=None,
    apply_scaling=True,
    apply_pca=False,
    pca_variance=0.95,
    verbose=True,
) -> ProcessingResult:
    """
    Flexible preprocessing pipeline for one tag group.

    This function allows you to configure which preprocessing steps to apply,
    making it easy to experiment with different configurations.

    Parameters
    ----------
    base_df : pl.DataFrame
        Base dataset with song-tag pairs from get_base_dataset()
    features_df : pl.DataFrame
        Audio features dataframe with song_id and feature columns
    tag_group : str
        Tag group to preprocess (e.g., "Genre", "Mood", "Situation")
    feature_cols : list[str] | None, default=None
        Feature columns to use. If None, uses all except song_id/song_path
    test_size : float, default=0.2
        Fraction of data to use for test set
    val_size : float, default=0.0
        Fraction of TRAINING data to use for validation set.
        If 0, no validation set is created.
        Example: test_size=0.2, val_size=0.2 means 60% train, 20% val, 20% test
    random_state : int, default=42
        Random seed for reproducible splits
    min_train_count : int | None, default=None
        Minimum occurrences in training set to keep a label.
        If None, no filtering is applied.
    apply_scaling : bool, default=True
        Whether to apply StandardScaler normalization
    apply_pca : bool, default=False
        Whether to apply PCA dimensionality reduction
    pca_variance : float | int, default=0.95
        PCA components: float for variance ratio (e.g., 0.95),
        int for fixed number of components. Only used if apply_pca=True.
    verbose : bool, default=True
        Whether to print progress and statistics

    Returns
    -------
    ProcessingResult
        A dataclass containing all preprocessing results with the following attributes:
        - X_train: Preprocessed training features
        - X_val: Preprocessed validation features (None if val_size=0)
        - X_test: Preprocessed test features
        - y_train: Training labels (binary matrix)
        - y_val: Validation labels (binary matrix, None if val_size=0)
        - y_test: Test labels (binary matrix)
        - tags: List of tag names after filtering
        - song_ids_train: Song IDs for train set
        - song_ids_val: Song IDs for validation set (None if val_size=0)
        - song_ids_test: Song IDs for test set
        - scaler: Fitted scaler if apply_scaling=True, else None
        - pca: Fitted PCA if apply_pca=True, else None

    Examples
    --------
    >>> # With validation set for hyperparameter tuning
    >>> result = preprocess_tag_group(
    ...     base_df, features_df, "Genre",
    ...     test_size=0.2, val_size=0.2,
    ...     apply_scaling=True,
    ...     apply_pca=True
    ... )
    >>> print(result.X_train.shape)  # 60% of data
    >>> print(result.X_val.shape)    # 20% of data
    >>> print(result.X_test.shape)   # 20% of data
    >>> model.fit(result.X_train.to_numpy(), result.y_train.to_numpy())
    >>>
    >>> # Without validation set (standard train/test split)
    >>> mood_result = preprocess_tag_group(
    ...     base_df, features_df, "Mood",
    ...     test_size=0.25,
    ...     val_size=0.0,
    ...     min_train_count=10,
    ...     apply_scaling=True,
    ...     apply_pca=True,
    ...     pca_variance=0.95
    ... )
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"PREPROCESSING: {tag_group.upper()}")
        print(f"{'=' * 70}")

    # 1. Prepare multi-label data
    X, y, tags = prepare_multilabel_data(
        base_df=base_df,
        features_df=features_df,
        tag_group=tag_group,
        feature_columns=feature_cols,
    )

    # Show initial label distribution
    if verbose:
        print(f"\n{tag_group} - Initial label statistics:")
        print(get_multilabel_stats(y, tags))

    # IMPORTANT: Save song IDs before dropping them
    song_ids = X.select("song_id")
    X_features = X.drop("song_id")

    # 2. Split train/test at SONG level
    # First split: separate test set
    X_train, X_test, y_train, y_test, song_ids_train, song_ids_test = train_test_split(
        X_features, y, song_ids, test_size=test_size, random_state=random_state
    )

    # Second split: create validation set from training data (if requested)
    X_val = None
    y_val = None
    song_ids_val = None

    if val_size > 0:
        # Calculate validation size relative to the remaining training data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val, song_ids_train, song_ids_val = train_test_split(
            X_train, y_train, song_ids_train,
            test_size=val_size_adjusted,
            random_state=random_state
        )

    # 3. Filter rare labels based on training set (optional)
    if min_train_count is not None:
        X_train, y_train, X_test, y_test, tags = filter_rare_labels(
            X_train, y_train, X_test, y_test, tags, min_count=min_train_count
        )
        # Also filter validation set if it exists
        if X_val is not None and y_val is not None:
            y_val = y_val.select(tags)
    elif verbose:
        print("Skipping rare label filtering (min_train_count=None)\n")

    # Track the final feature matrices
    X_train_final: pl.DataFrame = X_train
    X_val_final: pl.DataFrame | None = X_val
    X_test_final: pl.DataFrame = X_test
    scaler: PolarsStandardScaler | None = None
    pca: PolarsPCA | None = None

    # 4. Normalize features (optional)
    if apply_scaling:
        if verbose:
            print("Normalizing features...")
        scaler = PolarsStandardScaler()
        X_train_final = scaler.fit_transform(X_train_final)
        X_test_final = scaler.transform(X_test_final)
        if X_val_final is not None:
            X_val_final = scaler.transform(X_val_final)
    elif verbose:
        print("Skipping feature normalization (apply_scaling=False)\n")

    # 5. Apply PCA (optional)
    if apply_pca:
        if verbose:
            print(f"Applying PCA with n_components={pca_variance}...")
        pca = PolarsPCA(n_components=pca_variance)
        X_train_final = pca.fit_transform(X_train_final)
        X_test_final = pca.transform(X_test_final)
        if X_val_final is not None:
            X_val_final = pca.transform(X_val_final)
    elif verbose:
        print("Skipping PCA (apply_pca=False)\n")

    # Summary
    if verbose:
        print(f"\n{tag_group} - Final dataset summary:")
        print(f"  Original features: {X_train.shape[1]}")
        if apply_pca:
            print(f"  PCA components: {X_train_final.shape[1]}")
            print(f"  Variance preserved: {pca.explained_variance_ratio_.sum():.2%}")
        else:
            print(f"  Final features: {X_train_final.shape[1]}")
        print(f"  Train samples: {X_train_final.shape[0]}")
        if X_val_final is not None:
            print(f"  Validation samples: {X_val_final.shape[0]}")
        print(f"  Test samples: {X_test_final.shape[0]}")
        print(f"  Active labels: {len(tags)}")
        print(
            f"  Avg tags/song (train): {y_train.select(pl.sum_horizontal(pl.all()).mean()).item():.2f}"
        )
        print("  Preprocessing applied:")
        print(f"    - Scaling: {apply_scaling}")
        print(f"    - PCA: {apply_pca}")
        print(f"    - Validation split: {val_size > 0}")
        print(f"    - Rare label filtering: {min_train_count is not None}")
        print(f"{'=' * 70}\n")

    return ProcessingResult(
        X_train=X_train_final,
        X_val=X_val_final,
        X_test=X_test_final,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        tags=tags,
        song_ids_train=song_ids_train,
        song_ids_val=song_ids_val,
        song_ids_test=song_ids_test,
        scaler=scaler,
        pca=pca,
    )


def predict_with_metadata(
    processing_result: ProcessingResult,
    model: MultiOutputClassifier | dict[str, MultiOutputClassifier],
    db: pyrekordbox.Rekordbox6Database = None,
    songs_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Generate predictions with song metadata from a ProcessingResult.

    This is a convenience function that automates the common workflow of:
    1. Making predictions on test data
    2. Converting predictions to labels
    3. Joining with song metadata (artist, title)

    Parameters
    ----------
    processing_result : ProcessingResult
        The result from preprocess_tag_group()
    model : MultiOutputClassifier or dict[str, MultiOutputClassifier]
        Either:
        - A single fitted sklearn multi-label classifier
        - A dictionary mapping model names to fitted classifiers
          (e.g., {"Linear": model1, "Random Forest": model2})
    db : Rekordbox6Database, optional
        Database instance to fetch song metadata. Either db or songs_df must be provided.
    songs_df : pl.DataFrame, optional
        Pre-fetched songs dataframe with metadata. Either db or songs_df must be provided.
        Must contain columns: song_id, artist_name, song_title

    Returns
    -------
    pl.DataFrame
        If single model:
            DataFrame with columns: song_id, artist_name, song_title, predicted_tags
        If multiple models (dict):
            DataFrame with columns: song_id, artist_name, song_title,
            model1_predicted_tags, model2_predicted_tags, ...
            (one column per model with format: {model_name}_predicted_tags)

    Examples
    --------
    >>> from processing import preprocess_tag_group, predict_with_metadata
    >>> from models import get_random_forest_model, get_linear_model
    >>>
    >>> # Preprocess data
    >>> result = preprocess_tag_group(
    ...     base_df, features_df, "Genre",
    ...     apply_scaling=True,
    ...     apply_pca=True
    ... )
    >>>
    >>> # Single model
    >>> model = get_random_forest_model(n_estimators=200, max_depth=20)
    >>> model.fit(result.X_train.to_numpy(), result.y_train.to_numpy())
    >>> predictions = predict_with_metadata(result, model, db)
    >>> print(predictions)
    >>>
    >>> # Multiple models for comparison
    >>> models = {
    ...     "Linear": get_linear_model(),
    ...     "Random Forest": get_random_forest_model()
    ... }
    >>> for model in models.values():
    ...     model.fit(result.X_train.to_numpy(), result.y_train.to_numpy())
    >>> predictions = predict_with_metadata(result, models, db)
    >>> print(predictions)  # Shows predictions from both models side-by-side
    """
    # Validate inputs
    if db is None and songs_df is None:
        raise ValueError("Either db or songs_df must be provided")

    # Get test song IDs
    test_song_ids = processing_result.song_ids_test["song_id"].to_list()
    X_test_np = processing_result.X_test.to_numpy()

    # Check if model is a dictionary (multiple models) or single model
    is_multi_model = isinstance(model, dict)

    if is_multi_model:
        # Multiple models: create predictions for each model
        all_predictions = {}

        for model_name, single_model in model.items():
            # Get predictions for this model
            y_pred = single_model.predict(X_test_np)  # type: ignore

            # Convert predictions to labels
            model_predictions = []
            for i in range(len(y_pred)):  # type: ignore
                pred_vector = y_pred[i]  # type: ignore
                predicted_tags = [
                    tag for tag, pred in zip(processing_result.tags, pred_vector) if pred == 1
                ]
                model_predictions.append(predicted_tags)

            # Store with model name
            all_predictions[model_name] = model_predictions

        # Create DataFrame with all models' predictions
        predictions_data = []
        for i, song_id in enumerate(test_song_ids):
            row = {"song_id": song_id}
            for model_name, predictions in all_predictions.items():
                # Use snake_case for column names
                col_name = f"{model_name.lower().replace(' ', '_')}_predicted_tags"
                row[col_name] = predictions[i]
            predictions_data.append(row)

        predictions_df = pl.DataFrame(predictions_data)
    else:
        # Single model: original behavior
        y_pred = model.predict(X_test_np)  # type: ignore

        # Convert predictions to labels
        predictions_data = []
        for i, song_id in enumerate(test_song_ids):
            pred_vector = y_pred[i]  # type: ignore
            predicted_tags = [
                tag for tag, pred in zip(processing_result.tags, pred_vector) if pred == 1
            ]
            predictions_data.append({"song_id": song_id, "predicted_tags": predicted_tags})

        predictions_df = pl.DataFrame(predictions_data)

    # Fetch song metadata if needed
    if songs_df is None:
        if db is None:
            raise ValueError("Either db or songs_df must be provided")
        from utils import get_clean_songs

        songs_df = get_clean_songs(db, rename=True)

    # Join with metadata
    predictions_with_metadata = predictions_df.join(
        songs_df.select(["song_id", "artist_name", "song_title"]),
        on="song_id",
        how="left",
    )

    # Reorder columns: song_id, artist_name, song_title, then prediction columns
    if is_multi_model:
        prediction_cols = [col for col in predictions_with_metadata.columns
                          if col.endswith("_predicted_tags")]
        predictions_with_metadata = predictions_with_metadata.select(
            ["song_id", "artist_name", "song_title"] + prediction_cols
        )
    else:
        predictions_with_metadata = predictions_with_metadata.select(
            ["song_id", "artist_name", "song_title", "predicted_tags"]
        )

    return predictions_with_metadata


def predict_with_optimized_thresholds(
    models: Union[dict[str, Any], Any],
    X_test: pl.DataFrame,
    tags: list[str],
    thresholds: Union[dict[str, "np.ndarray"], "np.ndarray"],
    songs_df: pl.DataFrame,
    song_id_col: str = "song_id",
) -> pl.DataFrame:
    """Make predictions using optimized thresholds and join with song metadata.

    Convenience function for inference that:
    1. Gets probability predictions from model(s)
    2. Applies optimized thresholds (not 0.5!)
    3. Converts to tag lists
    4. Joins with song metadata

    Parameters
    ----------
    models : dict[str, MultiOutputClassifier] or MultiOutputClassifier
        Either:
        - Single model
        - Dictionary of models (e.g., {"Random Forest": model1, "XGBoost": model2})
    X_test : pl.DataFrame
        Test features (must include song_id column)
    tags : list[str]
        List of tag names
    thresholds : dict[str, np.ndarray] or np.ndarray
        Either:
        - Single array of thresholds (if single model)
        - Dictionary of thresholds matching model keys (if multiple models)
    songs_df : pl.DataFrame
        Song metadata DataFrame
    song_id_col : str, default="song_id"
        Name of the song ID column

    Returns
    -------
    pl.DataFrame
        Predictions with song metadata
        If single model: columns include "predicted_tags"
        If multiple models: columns include "{model_name}_predicted_tags" for each model

    Examples
    --------
    >>> from models import load_model
    >>> from processing import predict_with_optimized_thresholds
    >>>
    >>> # Single model
    >>> model_data = load_model("models/xgboost_Genre_model.pkl")
    >>> predictions = predict_with_optimized_thresholds(
    ...     models=model_data['model'],
    ...     X_test=X_test_scaled,
    ...     tags=model_data['tags'],
    ...     thresholds=model_data['thresholds'],
    ...     songs_df=songs_df
    ... )
    >>>
    >>> # Multiple models at once
    >>> rf_data = load_model("models/rf_model.pkl")
    >>> xgb_data = load_model("models/xgb_model.pkl")
    >>> predictions = predict_with_optimized_thresholds(
    ...     models={"Random Forest": rf_data['model'], "XGBoost": xgb_data['model']},
    ...     X_test=X_test_scaled,
    ...     tags=rf_data['tags'],
    ...     thresholds={"Random Forest": rf_data['thresholds'], "XGBoost": xgb_data['thresholds']},
    ...     songs_df=songs_df
    ... )
    """
    from models import predict_with_threshold
    import numpy as np

    # Get song IDs before dropping
    song_ids = X_test.select(song_id_col)
    X_features = X_test.drop(song_id_col)

    # Check if multiple models or single model
    is_multi_model = isinstance(models, dict)

    if is_multi_model:
        # Multiple models
        all_predictions = {}

        for model_name, model in models.items():
            # Get probabilities and apply thresholds
            _, probs = predict_with_threshold(model, X_features, threshold=0.5)
            pred_binary = (probs >= thresholds[model_name]).astype(int)

            # Convert to tag lists
            predicted_tags = []
            for pred_vector in pred_binary:
                tag_list = [tags[i] for i in range(len(tags)) if pred_vector[i] == 1]
                predicted_tags.append(tag_list if tag_list else None)

            all_predictions[f"{model_name}_predicted_tags"] = predicted_tags

        # Create predictions dataframe
        pred_df = pl.DataFrame({
            song_id_col: song_ids[song_id_col],
            **all_predictions
        })

    else:
        # Single model
        _, probs = predict_with_threshold(models, X_features, threshold=0.5)
        pred_binary = (probs >= thresholds).astype(int)

        # Convert to tag lists
        predicted_tags = []
        for pred_vector in pred_binary:
            tag_list = [tags[i] for i in range(len(tags)) if pred_vector[i] == 1]
            predicted_tags.append(tag_list if tag_list else None)

        pred_df = pl.DataFrame({
            song_id_col: song_ids[song_id_col],
            "predicted_tags": predicted_tags
        })

    # Join with metadata
    predictions = songs_df.join(pred_df, on=song_id_col, how="inner")

    return predictions
