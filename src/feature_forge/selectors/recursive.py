"""
Recursive feature elimination selector
"""

import pandas as pd
import numpy as np
from typing import List
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class RecursiveSelector:
    """
    Recursive Feature Elimination (RFE) selector

    Uses RFE to recursively remove features and select the best subset.

    Parameters
    ----------
    X : pd.DataFrame
        Input features
    y : pd.Series
        Target variable
    task : str
        Task type ('classification' or 'regression')
    n_features : int
        Number of features to select
    step : int or float, default=1
        Number of features to remove at each iteration
    verbose : bool, default=True
        Whether to print progress
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: str,
        n_features: int,
        step: int = 1,
        verbose: bool = True,
    ):
        self.X = X
        self.y = y
        self.task = task
        self.n_features = n_features
        self.step = step
        self.verbose = verbose

    def select(self, model_type: str = "rf") -> List[str]:
        """
        Select features using RFE

        Parameters
        ----------
        model_type : str, default='rf'
            Model to use: 'rf' or 'tree'

        Returns
        -------
        list of str
            Selected feature names
        """
        # Prepare data
        X_prepared = self._prepare_features()

        # Select estimator
        if self.task == "classification":
            estimator = RandomForestClassifier(
                n_estimators=50, random_state=42, n_jobs=-1
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=50, random_state=42, n_jobs=-1
            )

        if self.verbose:
            print(f"    Running RFE to select {self.n_features} features...")

        # Run RFE
        selector = RFE(
            estimator=estimator, n_features_to_select=self.n_features, step=self.step
        )

        selector.fit(X_prepared, self.y)

        # Get selected features
        selected_features = X_prepared.columns[selector.support_].tolist()

        if self.verbose:
            print(f"    RFE selected {len(selected_features)} features")

        return selected_features

    def _prepare_features(self) -> pd.DataFrame:
        """
        Prepare features for RFE

        Returns
        -------
        pd.DataFrame
            Prepared features
        """
        X_prepared = self.X.copy()

        # Handle categorical columns
        cat_cols = X_prepared.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            X_prepared[col] = pd.Categorical(X_prepared[col]).codes

        # Handle missing values
        X_prepared = X_prepared.fillna(-999)

        return X_prepared
