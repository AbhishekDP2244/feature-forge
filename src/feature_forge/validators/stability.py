"""
Feature stability checker
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from typing import Dict


class StabilityChecker:
    """
    Check stability of feature importance across CV folds

    Stable features maintain consistent importance across different
    data splits, indicating they are reliable predictors.

    Parameters
    ----------
    X : pd.DataFrame
        Input features
    y : pd.Series
        Target variable
    task : str
        Task type ('classification' or 'regression')
    n_splits : int, default=5
        Number of CV folds
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, task: str, n_splits: int = 5):
        self.X = X
        self.y = y
        self.task = task
        self.n_splits = n_splits

    def check_stability(self, model_type: str = "lgbm") -> pd.DataFrame:
        """
        Check feature importance stability across folds

        Parameters
        ----------
        model_type : str, default='lgbm'
            Model to use for importance calculation

        Returns
        -------
        pd.DataFrame
            Stability metrics for each feature
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        # Prepare data
        X_prepared = self._prepare_features()

        # Store importance across folds
        importance_folds = []

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for train_idx, _ in kf.split(X_prepared):
            X_train = X_prepared.iloc[train_idx]
            y_train = self.y.iloc[train_idx]

            # Train model
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

            model.fit(X_train, y_train)

            # Get importance
            importance = dict(zip(X_prepared.columns, model.feature_importances_))
            importance_folds.append(importance)

        # Calculate stability metrics
        stability_metrics = []

        for feature in X_prepared.columns:
            importances = [fold_imp.get(feature, 0) for fold_imp in importance_folds]

            stability_metrics.append(
                {
                    "feature": feature,
                    "mean_importance": np.mean(importances),
                    "std_importance": np.std(importances),
                    "cv_importance": np.std(importances)
                    / (np.mean(importances) + 1e-10),  # Coefficient of variation
                    "min_importance": np.min(importances),
                    "max_importance": np.max(importances),
                }
            )

        return pd.DataFrame(stability_metrics).sort_values(
            "mean_importance", ascending=False
        )

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
            X_prepared[col] = pd.Categorical(X_prepared[col]).codes

        # Handle missing values
        X_prepared = X_prepared.fillna(-999)

        return X_prepared
