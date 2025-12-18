"""
Aggregation feature generator
"""

import pandas as pd
import numpy as np
from typing import List


class AggregationGenerator:
    """
    Generate aggregation features

    Creates features based on groupby aggregations for categorical columns.
    Useful when you have categorical features that can be grouped.

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
        Generate aggregation features

        Parameters
        ----------
        max_features : int, default=20
            Maximum number of features to generate

        Returns
        -------
        pd.DataFrame
            Generated aggregation features
        """
        # Get categorical and numeric columns
        categorical_cols = self.X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()

        if len(categorical_cols) == 0 or len(numeric_cols) == 0:
            return pd.DataFrame(index=self.X.index)

        aggregations = {}
        count = 0

        # Limit to prevent too many features
        categorical_cols = categorical_cols[:3]  # Max 3 categorical features
        numeric_cols = numeric_cols[:5]  # Max 5 numeric features

        # Generate aggregations
        for cat_col in categorical_cols:
            if count >= max_features:
                break

            # Check cardinality
            n_unique = self.X[cat_col].nunique()
            if n_unique > 100 or n_unique < 2:
                continue

            for num_col in numeric_cols:
                if count >= max_features:
                    break

                try:
                    # Mean aggregation
                    if count < max_features:
                        grouped_mean = self.X.groupby(cat_col)[num_col].transform(
                            "mean"
                        )
                        aggregations[f"{cat_col}_mean_{num_col}"] = grouped_mean
                        count += 1

                    # Median aggregation
                    if count < max_features:
                        grouped_median = self.X.groupby(cat_col)[num_col].transform(
                            "median"
                        )
                        aggregations[f"{cat_col}_median_{num_col}"] = grouped_median
                        count += 1

                    # Std aggregation
                    if count < max_features:
                        grouped_std = self.X.groupby(cat_col)[num_col].transform("std")
                        aggregations[f"{cat_col}_std_{num_col}"] = grouped_std
                        count += 1

                    # Min aggregation
                    if count < max_features:
                        grouped_min = self.X.groupby(cat_col)[num_col].transform("min")
                        aggregations[f"{cat_col}_min_{num_col}"] = grouped_min
                        count += 1

                    # Max aggregation
                    if count < max_features:
                        grouped_max = self.X.groupby(cat_col)[num_col].transform("max")
                        aggregations[f"{cat_col}_max_{num_col}"] = grouped_max
                        count += 1

                    # Difference from group mean
                    if count < max_features:
                        diff_from_mean = self.X[num_col] - grouped_mean
                        aggregations[f"{cat_col}_diff_mean_{num_col}"] = diff_from_mean
                        count += 1

                except Exception as e:
                    # Skip if error occurs
                    continue

        if len(aggregations) == 0:
            return pd.DataFrame(index=self.X.index)

        # Create dataframe
        agg_df = pd.DataFrame(aggregations, index=self.X.index)

        # Remove features with too many NaN
        valid_features = []
        for col in agg_df.columns:
            nan_ratio = agg_df[col].isna().sum() / len(agg_df)
            if nan_ratio < 0.5:
                valid_features.append(col)

        return agg_df[valid_features]
