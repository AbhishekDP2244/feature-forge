"""
Correlation-based feature selector
"""

import pandas as pd
import numpy as np
from typing import List


class RedundancyRemover:
    """
    Remove redundant features based on correlation

    Identifies and removes highly correlated features to reduce redundancy
    in the feature set. When two features are highly correlated, keeps the
    one with higher correlation to the target.

    Parameters
    ----------
    X : pd.DataFrame
        Input features
    y : pd.Series
        Target variable
    threshold : float, default=0.95
        Correlation threshold for considering features redundant
    method : str, default='correlation'
        Method for detecting redundancy: 'correlation' or 'mutual_information'
    verbose : bool, default=True
        Whether to print progress
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.95,
        method: str = "correlation",
        verbose: bool = True,
    ):
        self.X = X
        self.y = y
        self.threshold = threshold
        self.method = method
        self.verbose = verbose

    def select(self) -> List[str]:
        """
        Select non-redundant features

        Returns
        -------
        list of str
            Selected feature names
        """
        if self.method == "correlation":
            return self._select_by_correlation()
        elif self.method == "mutual_information":
            return self._select_by_mutual_info()
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _select_by_correlation(self) -> List[str]:
        """
        Select features by correlation-based redundancy removal

        Returns
        -------
        list of str
            Selected feature names
        """
        # Select only numeric columns
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) == 0:
            return self.X.columns.tolist()

        X_numeric = self.X[numeric_cols]

        # Calculate correlation matrix
        corr_matrix = X_numeric.corr().abs()

        # Calculate correlation with target
        target_corr = {}
        for col in numeric_cols:
            try:
                mask = ~(X_numeric[col].isna() | self.y.isna())
                if mask.sum() > 0:
                    corr = abs(X_numeric.loc[mask, col].corr(self.y[mask]))
                    target_corr[col] = corr if not np.isnan(corr) else 0
                else:
                    target_corr[col] = 0
            except:
                target_corr[col] = 0

        # Find redundant features
        redundant = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]

                # Check if correlation exceeds threshold
                if corr_matrix.iloc[i, j] > self.threshold:
                    # Keep the feature with higher target correlation
                    if target_corr[col_i] >= target_corr[col_j]:
                        redundant.add(col_j)
                    else:
                        redundant.add(col_i)

        # Keep all non-numeric features and non-redundant numeric features
        non_numeric_cols = [col for col in self.X.columns if col not in numeric_cols]
        selected_numeric = [col for col in numeric_cols if col not in redundant]

        selected_features = non_numeric_cols + selected_numeric

        if self.verbose:
            print(f"    Correlation threshold: {self.threshold}")
            print(f"    Removed {len(redundant)} redundant features")

        return selected_features

    def _select_by_mutual_info(self) -> List[str]:
        """
        Select features by mutual information-based redundancy removal

        Returns
        -------
        list of str
            Selected feature names
        """
        from sklearn.feature_selection import (
            mutual_info_classif,
            mutual_info_regression,
        )

        # Select only numeric columns
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) == 0:
            return self.X.columns.tolist()

        X_numeric = self.X[numeric_cols].fillna(0)

        # Calculate mutual information with target
        if self.y.dtype in ["object", "category", "bool"] or self.y.nunique() < 10:
            mi_scores = mutual_info_classif(X_numeric, self.y, random_state=42)
        else:
            mi_scores = mutual_info_regression(X_numeric, self.y, random_state=42)

        mi_dict = dict(zip(numeric_cols, mi_scores))

        # Calculate pairwise mutual information (simplified using correlation)
        corr_matrix = X_numeric.corr().abs()

        # Find redundant features
        redundant = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]

                # Check if correlation exceeds threshold
                if corr_matrix.iloc[i, j] > self.threshold:
                    # Keep the feature with higher MI score
                    if mi_dict[col_i] >= mi_dict[col_j]:
                        redundant.add(col_j)
                    else:
                        redundant.add(col_i)

        # Keep all non-numeric features and non-redundant numeric features
        non_numeric_cols = [col for col in self.X.columns if col not in numeric_cols]
        selected_numeric = [col for col in numeric_cols if col not in redundant]

        selected_features = non_numeric_cols + selected_numeric

        if self.verbose:
            print(f"    MI threshold: {self.threshold}")
            print(f"    Removed {len(redundant)} redundant features")

        return selected_features
