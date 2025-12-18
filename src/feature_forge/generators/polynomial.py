"""
Polynomial feature generator
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from typing import List


class PolynomialGenerator:
    """
    Generate polynomial and power features

    Creates polynomial features up to a specified degree, including
    interaction terms between features.

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

    def generate(self, max_features: int = 20, degree: int = 2) -> pd.DataFrame:
        """
        Generate polynomial features

        Parameters
        ----------
        max_features : int, default=20
            Maximum number of features to generate
        degree : int, default=2
            Maximum degree of polynomial features

        Returns
        -------
        pd.DataFrame
            Generated polynomial features
        """
        # Select only numeric columns
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) == 0:
            return pd.DataFrame(index=self.X.index)

        # Limit number of columns to prevent explosion
        # Select most important columns based on correlation
        n_cols = min(len(numeric_cols), 5)
        selected_cols = self._select_important_cols(numeric_cols, n_cols)

        X_numeric = self.X[selected_cols]

        # Generate polynomial features
        poly = PolynomialFeatures(
            degree=degree, include_bias=False, interaction_only=False
        )

        try:
            X_poly = poly.fit_transform(X_numeric)
        except Exception as e:
            print(f"Warning: Error in polynomial generation: {e}")
            return pd.DataFrame(index=self.X.index)

        # Get feature names
        feature_names = poly.get_feature_names_out(selected_cols)

        # Remove original features (they're already in X)
        original_features = set(selected_cols)
        new_indices = []
        new_names = []

        for i, name in enumerate(feature_names):
            # Skip if it's an original feature
            if name not in original_features:
                new_indices.append(i)
                # Create readable name
                new_name = self._create_feature_name(name)
                new_names.append(new_name)

        if len(new_indices) == 0:
            return pd.DataFrame(index=self.X.index)

        X_poly_new = X_poly[:, new_indices]

        # Limit to max_features
        if len(new_names) > max_features:
            # Keep features with highest variance
            variances = np.var(X_poly_new, axis=0)
            top_indices = np.argsort(variances)[-max_features:]
            X_poly_new = X_poly_new[:, top_indices]
            new_names = [new_names[i] for i in top_indices]

        return pd.DataFrame(X_poly_new, columns=new_names, index=self.X.index)

    def _select_important_cols(self, cols: List[str], n: int) -> List[str]:
        """
        Select most important columns based on correlation with target

        Parameters
        ----------
        cols : list of str
            Column names to select from
        n : int
            Number of columns to select

        Returns
        -------
        list of str
            Selected column names
        """
        correlations = {}

        for col in cols:
            try:
                # Handle NaN values
                mask = ~(self.X[col].isna() | self.y.isna())
                if mask.sum() < len(self.X) * 0.5:  # Skip if too many NaN
                    continue

                corr = abs(self.X.loc[mask, col].corr(self.y[mask]))
                if not np.isnan(corr):
                    correlations[col] = corr
            except Exception:
                continue

        if len(correlations) == 0:
            # If no correlations computed, return first n columns
            return cols[:n]

        # Sort by correlation and return top n
        sorted_cols = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        return [col for col, _ in sorted_cols[:n]]

    def _create_feature_name(self, poly_name: str) -> str:
        """
        Create readable feature name from sklearn polynomial name

        Parameters
        ----------
        poly_name : str
            Polynomial feature name from sklearn

        Returns
        -------
        str
            Readable feature name
        """
        # Example: "x0^2" -> "feature1_squared"
        # Example: "x0 x1" -> "feature1_x_feature2"

        if "^" in poly_name:
            # Power feature
            return f"poly_{poly_name.replace('^', '_pow_').replace(' ', '_x_')}"
        elif " " in poly_name:
            # Interaction feature
            return f"poly_{poly_name.replace(' ', '_x_')}"
        else:
            return f"poly_{poly_name}"
