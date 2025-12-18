"""
Encoding feature generator
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


class EncodingGenerator:
    """
    Generate encoding features for categorical variables

    Creates encoded features using:
    - Target encoding (mean target per category with CV)
    - Frequency encoding (category frequency)
    - Count encoding (category counts)
    - Rank encoding (rank by frequency)

    Parameters
    ----------
    X : pd.DataFrame
        Input features
    y : pd.Series
        Target variable
    task : str
        Task type ('classification' or 'regression')
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, task: str):
        self.X = X
        self.y = y
        self.task = task

    def generate(self, max_features: int = 20) -> pd.DataFrame:
        """
        Generate encoded features

        Parameters
        ----------
        max_features : int, default=20
            Maximum number of features to generate

        Returns
        -------
        pd.DataFrame
            Generated encoding features
        """
        categorical_cols = self.X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if len(categorical_cols) == 0:
            return pd.DataFrame(index=self.X.index)

        encoded_features = {}
        count = 0

        # Process each categorical column
        for col in categorical_cols:
            if count >= max_features:
                break

            # Check cardinality
            n_unique = self.X[col].nunique()

            # Skip columns with too many or too few unique values
            if n_unique > 100 or n_unique < 2:
                continue

            try:
                # Target encoding with K-fold
                if count < max_features:
                    target_enc = self._target_encode(col)
                    encoded_features[f"{col}_target_enc"] = target_enc
                    count += 1

                # Frequency encoding
                if count < max_features:
                    freq_enc = self._frequency_encode(col)
                    encoded_features[f"{col}_freq_enc"] = freq_enc
                    count += 1

                # Count encoding
                if count < max_features:
                    count_enc = self._count_encode(col)
                    encoded_features[f"{col}_count_enc"] = count_enc
                    count += 1

                # Rank encoding
                if count < max_features:
                    rank_enc = self._rank_encode(col)
                    encoded_features[f"{col}_rank_enc"] = rank_enc
                    count += 1

            except Exception as e:
                # Skip this column if error
                continue

        if len(encoded_features) == 0:
            return pd.DataFrame(index=self.X.index)

        return pd.DataFrame(encoded_features, index=self.X.index)

    def _target_encode(self, col: str, n_splits: int = 5) -> pd.Series:
        """
        Target encoding with cross-validation to prevent overfitting

        Parameters
        ----------
        col : str
            Column name to encode
        n_splits : int, default=5
            Number of CV folds

        Returns
        -------
        pd.Series
            Target encoded values
        """
        encoded = pd.Series(index=self.X.index, dtype=float)

        # Handle NaN in categorical column
        X_col = self.X[col].fillna("__missing__")

        # Use KFold to prevent overfitting
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        try:
            for train_idx, val_idx in kf.split(self.X):
                # Calculate mean target for each category in train set
                target_means = (
                    self.y.iloc[train_idx].groupby(X_col.iloc[train_idx]).mean()
                )

                # Apply to validation set
                encoded.iloc[val_idx] = X_col.iloc[val_idx].map(target_means)

            # Fill remaining NaN with global mean
            encoded.fillna(self.y.mean(), inplace=True)

        except Exception:
            # If CV fails, use simple encoding with smoothing
            global_mean = self.y.mean()
            counts = X_col.value_counts()
            target_sum = self.y.groupby(X_col).sum()

            # Smoothing parameter
            alpha = 10
            smoothed_means = (target_sum + alpha * global_mean) / (counts + alpha)
            encoded = X_col.map(smoothed_means).fillna(global_mean)

        return encoded

    def _frequency_encode(self, col: str) -> pd.Series:
        """
        Frequency encoding (normalized counts)

        Parameters
        ----------
        col : str
            Column name to encode

        Returns
        -------
        pd.Series
            Frequency encoded values
        """
        X_col = self.X[col].fillna("__missing__")
        freq = X_col.value_counts(normalize=True)
        return X_col.map(freq)

    def _count_encode(self, col: str) -> pd.Series:
        """
        Count encoding (absolute counts)

        Parameters
        ----------
        col : str
            Column name to encode

        Returns
        -------
        pd.Series
            Count encoded values
        """
        X_col = self.X[col].fillna("__missing__")
        counts = X_col.value_counts()
        return X_col.map(counts)

    def _rank_encode(self, col: str) -> pd.Series:
        """
        Rank encoding (rank by frequency)

        Parameters
        ----------
        col : str
            Column name to encode

        Returns
        -------
        pd.Series
            Rank encoded values
        """
        X_col = self.X[col].fillna("__missing__")

        # Get value counts and create rank mapping
        value_counts = X_col.value_counts()
        rank_map = {val: rank + 1 for rank, val in enumerate(value_counts.index)}

        return X_col.map(rank_map)
