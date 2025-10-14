"""Preprocessing utilities for music tagging ML pipelines.

This module provides sklearn-compatible transformers for feature preprocessing
including normalization and dimensionality reduction via PCA.
"""

import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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

    def fit(self, X: pl.DataFrame, y=None):
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

    def fit(self, X: pl.DataFrame, y=None):
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

        print(f"\nPCA fitted: {self.n_features_in_} features -> {self.n_components_} components")
        print(f"Explained variance ratio: {self.explained_variance_ratio_[:5]}")  # Show first 5
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
        component_names = [f"PC{i+1}" for i in range(X_transformed.shape[1])]

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
