"""
Visualization utilities for feature analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional


class Visualizer:
    """
    Visualization utilities for feature engineering

    Provides methods to visualize feature importance, correlations,
    and other feature engineering insights.
    """

    @staticmethod
    def plot_feature_importance(
        importance_scores: Dict[str, float],
        top_n: int = 20,
        figsize: tuple = (10, 8),
        title: str = "Feature Importance",
    ):
        """
        Plot feature importance scores

        Parameters
        ----------
        importance_scores : dict
            Feature importance scores
        top_n : int, default=20
            Number of top features to plot
        figsize : tuple, default=(10, 8)
            Figure size
        title : str
            Plot title
        """
        # Sort by importance
        sorted_features = sorted(
            importance_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_features = sorted_features[:top_n]

        features, scores = zip(*top_features)

        plt.figure(figsize=figsize)
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel("Importance Score")
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_correlation_matrix(
        X: pd.DataFrame, figsize: tuple = (12, 10), top_n: Optional[int] = None
    ):
        """
        Plot correlation matrix heatmap

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        figsize : tuple, default=(12, 10)
            Figure size
        top_n : int, optional
            Plot only top n features by variance
        """
        # Select only numeric columns
        X_numeric = X.select_dtypes(include=[np.number])

        if top_n is not None and len(X_numeric.columns) > top_n:
            # Select top features by variance
            variances = X_numeric.var().sort_values(ascending=False)
            top_cols = variances.head(top_n).index
            X_numeric = X_numeric[top_cols]

        # Calculate correlation matrix
        corr_matrix = X_numeric.corr()

        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            annot=False,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
        )
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_feature_distribution(
        X: pd.DataFrame,
        feature: str,
        y: Optional[pd.Series] = None,
        figsize: tuple = (10, 6),
    ):
        """
        Plot feature distribution

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        feature : str
            Feature name to plot
        y : pd.Series, optional
            Target variable for conditional distribution
        figsize : tuple, default=(10, 6)
            Figure size
        """
        plt.figure(figsize=figsize)

        if pd.api.types.is_numeric_dtype(X[feature]):
            if y is not None:
                # Plot distribution by target
                for target_val in y.unique()[:5]:  # Max 5 classes
                    mask = y == target_val
                    plt.hist(
                        X.loc[mask, feature].dropna(),
                        alpha=0.5,
                        label=f"Target={target_val}",
                        bins=30,
                    )
                plt.legend()
            else:
                plt.hist(X[feature].dropna(), bins=30)
            plt.xlabel(feature)
            plt.ylabel("Frequency")
        else:
            # Categorical feature
            value_counts = X[feature].value_counts().head(20)
            plt.bar(range(len(value_counts)), value_counts.values)
            plt.xticks(
                range(len(value_counts)), value_counts.index, rotation=45, ha="right"
            )
            plt.xlabel(feature)
            plt.ylabel("Count")

        plt.title(f"Distribution of {feature}")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_missing_values(X: pd.DataFrame, figsize: tuple = (10, 6)):
        """
        Plot missing values heatmap

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        figsize : tuple, default=(10, 6)
            Figure size
        """
        missing = X.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing) == 0:
            print("No missing values found!")
            return

        missing_pct = (missing / len(X)) * 100

        plt.figure(figsize=figsize)
        plt.barh(range(len(missing)), missing_pct.values)
        plt.yticks(range(len(missing)), missing.index)
        plt.xlabel("Missing Values (%)")
        plt.title("Missing Values by Feature")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
