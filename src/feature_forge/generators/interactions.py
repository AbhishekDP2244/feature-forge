"""
Feature interaction generator
"""

import pandas as pd
import numpy as np
from itertools import combinations
from typing import List


class InteractionGenerator:
    """
    Generate feature interactions

    Creates mathematical interactions between features including:
    - Multiplication (x1 * x2)
    - Division (x1 / x2)
    - Addition (x1 + x2)
    - Subtraction (x1 - x2)
    - Difference ratios

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
        Generate interaction features

        Parameters
        ----------
        max_features : int, default=20
            Maximum number of features to generate

        Returns
        -------
        pd.DataFrame
            Generated interaction features
        """
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return pd.DataFrame(index=self.X.index)

        # Rank features by importance
        important_features = self._rank_by_importance(numeric_cols)[:10]

        interactions = {}
        count = 0

        # Generate pairwise interactions
        for col1, col2 in combinations(important_features, 2):
            if count >= max_features:
                break

            try:
                # Multiplication
                if count < max_features:
                    interactions[f"{col1}_times_{col2}"] = self.X[col1] * self.X[col2]
                    count += 1

                # Division (with safe handling)
                if count < max_features:
                    # Add small epsilon to avoid division by zero
                    denominator = self.X[col2].replace(0, np.nan)
                    interactions[f"{col1}_div_{col2}"] = self.X[col1] / (
                        denominator + 1e-10
                    )
                    # Replace inf with nan
                    interactions[f"{col1}_div_{col2}"].replace(
                        [np.inf, -np.inf], np.nan, inplace=True
                    )
                    count += 1

                # Addition
                if count < max_features:
                    interactions[f"{col1}_plus_{col2}"] = self.X[col1] + self.X[col2]
                    count += 1

                # Subtraction
                if count < max_features:
                    interactions[f"{col1}_minus_{col2}"] = self.X[col1] - self.X[col2]
                    count += 1

                # Ratio to sum
                if count < max_features:
                    total = self.X[col1] + self.X[col2]
                    interactions[f"{col1}_ratio_{col2}"] = self.X[col1] / (
                        total + 1e-10
                    )
                    interactions[f"{col1}_ratio_{col2}"].replace(
                        [np.inf, -np.inf], np.nan, inplace=True
                    )
                    count += 1

            except Exception as e:
                # Skip this pair if there's an error
                continue

        if len(interactions) == 0:
            return pd.DataFrame(index=self.X.index)

        # Create dataframe
        interactions_df = pd.DataFrame(interactions, index=self.X.index)

        # Remove features with too many NaN or inf values
        valid_features = []
        for col in interactions_df.columns:
            nan_ratio = interactions_df[col].isna().sum() / len(interactions_df)
            if nan_ratio < 0.5:  # Keep features with less than 50% NaN
                valid_features.append(col)

        return interactions_df[valid_features]

    def _rank_by_importance(self, cols: List[str]) -> List[str]:
        """
        Rank features by correlation with target

        Parameters
        ----------
        cols : list of str
            Column names to rank

        Returns
        -------
        list of str
            Ranked column names
        """
        correlations = {}

        for col in cols:
            try:
                # Handle NaN values
                mask = ~(self.X[col].isna() | self.y.isna())
                if mask.sum() < len(self.X) * 0.5:
                    continue

                corr = abs(self.X.loc[mask, col].corr(self.y[mask]))
                if not np.isnan(corr):
                    correlations[col] = corr
            except Exception:
                continue

        if len(correlations) == 0:
            return cols

        # Sort by correlation (highest first)
        sorted_cols = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        return [col for col, _ in sorted_cols]
