"""
Importance-based feature selector
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class ImportanceSelector:
    """
    Select features based on importance scores

    Uses model-based feature importance to select top features.
    Supports various models and importance calculation methods.

    Parameters
    ----------
    X : pd.DataFrame
        Input features
    y : pd.Series
        Target variable
    task : str
        Task type ('classification' or 'regression')
    n_features : int, optional
        Number of features to select
    threshold : float, optional
        Importance threshold for selection
    verbose : bool, default=True
        Whether to print progress
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: str,
        n_features: Optional[int] = None,
        threshold: Optional[float] = None,
        verbose: bool = True,
    ):
        self.X = X
        self.y = y
        self.task = task
        self.n_features = n_features
        self.threshold = threshold
        self.verbose = verbose

        if n_features is None and threshold is None:
            # Default: select top 50% of features
            self.n_features = max(1, len(X.columns) // 2)

    def select(self, model_type: str = "lgbm") -> List[str]:
        """
        Select features based on importance

        Parameters
        ----------
        model_type : str, default='lgbm'
            Model to use: 'lgbm', 'xgb', 'rf', or 'tree'

        Returns
        -------
        list of str
            Selected feature names
        """
        # Get feature importance
        importance_scores = self._calculate_importance(model_type)

        # Select features
        if self.threshold is not None:
            # Select by threshold
            selected = [
                feat
                for feat, score in importance_scores.items()
                if score >= self.threshold
            ]
        else:
            # Select top n_features
            sorted_features = sorted(
                importance_scores.items(), key=lambda x: x[1], reverse=True
            )
            selected = [feat for feat, _ in sorted_features[: self.n_features]]

        if self.verbose:
            print(f"    Selected {len(selected)} features by importance")

        return selected

    def _calculate_importance(self, model_type: str) -> dict:
        """
        Calculate feature importance using specified model

        Parameters
        ----------
        model_type : str
            Model type to use

        Returns
        -------
        dict
            Feature importance scores
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        # Prepare data (handle categorical and missing values)
        X_prepared = self._prepare_features()

        # Select model
        if model_type == "lgbm":
            try:
                import lightgbm as lgb

                if self.task == "classification":
                    model = lgb.LGBMClassifier(
                        n_estimators=100, random_state=42, verbose=-1
                    )
                else:
                    model = lgb.LGBMRegressor(
                        n_estimators=100, random_state=42, verbose=-1
                    )
            except ImportError:
                if self.verbose:
                    print("    LightGBM not available, using Random Forest")
                model_type = "rf"

        if model_type == "xgb":
            try:
                import xgboost as xgb

                if self.task == "classification":
                    model = xgb.XGBClassifier(n_estimators=100, random_state=42)
                else:
                    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            except ImportError:
                if self.verbose:
                    print("    XGBoost not available, using Random Forest")
                model_type = "rf"

        if model_type == "rf":
            if self.task == "classification":
                model = RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100, random_state=42, n_jobs=-1
                )

        elif model_type == "tree":
            if self.task == "classification":
                model = DecisionTreeClassifier(max_depth=10, random_state=42)
            else:
                model = DecisionTreeRegressor(max_depth=10, random_state=42)

        # Fit model
        model.fit(X_prepared, self.y)

        # Get feature importance
        importance = model.feature_importances_

        # Create importance dictionary
        importance_dict = dict(zip(X_prepared.columns, importance))

        return importance_dict

    def _prepare_features(self) -> pd.DataFrame:
        """
        Prepare features for model training

        Returns
        -------
        pd.DataFrame
            Prepared features
        """
        X_prepared = self.X.copy()

        # Handle categorical columns
        cat_cols = X_prepared.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            # Simple label encoding
            X_prepared[col] = pd.Categorical(X_prepared[col]).codes

        # Handle missing values
        X_prepared = X_prepared.fillna(-999)

        return X_prepared
