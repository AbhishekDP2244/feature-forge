"""
Feature scoring and ranking
"""

import pandas as pd
import numpy as np
from typing import Dict
from sklearn.model_selection import cross_val_score


class FeatureScorer:
    """
    Score and rank features by importance

    Provides multiple methods for calculating feature importance scores.

    Parameters
    ----------
    X : pd.DataFrame
        Input features
    y : pd.Series
        Target variable
    task : str
        Task type ('classification' or 'regression')
    verbose : bool, default=True
        Whether to print progress
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, task: str, verbose: bool = True):
        self.X = X
        self.y = y
        self.task = task
        self.verbose = verbose

    def score_features(
        self, model_type: str = "lgbm", method: str = "importance", cv: int = 5
    ) -> Dict[str, float]:
        """
        Score all features

        Parameters
        ----------
        model_type : str, default='lgbm'
            Model to use: 'lgbm', 'xgb', 'rf', 'tree'
        method : str, default='importance'
            Scoring method: 'importance', 'permutation', or 'correlation'
        cv : int, default=5
            Number of cross-validation folds

        Returns
        -------
        dict
            Feature scores
        """
        if method == "importance":
            return self._score_by_importance(model_type)
        elif method == "permutation":
            return self._score_by_permutation(model_type, cv)
        elif method == "correlation":
            return self._score_by_correlation()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _score_by_importance(self, model_type: str) -> Dict[str, float]:
        """
        Score features using model feature importance

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

        # Prepare data
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
                    model = xgb.XGBClassifier(
                        n_estimators=100, random_state=42, verbosity=0
                    )
                else:
                    model = xgb.XGBRegressor(
                        n_estimators=100, random_state=42, verbosity=0
                    )
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

        # Normalize to sum to 1
        importance = importance / importance.sum()

        # Create importance dictionary
        importance_dict = dict(zip(X_prepared.columns, importance))

        return importance_dict

    def _score_by_permutation(self, model_type: str, cv: int) -> Dict[str, float]:
        """
        Score features using permutation importance

        Parameters
        ----------
        model_type : str
            Model type to use
        cv : int
            Number of CV folds

        Returns
        -------
        dict
            Permutation importance scores
        """
        from sklearn.inspection import permutation_importance
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        # Prepare data
        X_prepared = self._prepare_features()

        # Select model
        if self.task == "classification":
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

        # Fit model
        model.fit(X_prepared, self.y)

        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X_prepared, self.y, n_repeats=10, random_state=42, n_jobs=-1
        )

        # Get mean importance
        importance = perm_importance.importances_mean

        # Normalize
        if importance.sum() > 0:
            importance = importance / importance.sum()

        # Create importance dictionary
        importance_dict = dict(zip(X_prepared.columns, importance))

        return importance_dict

    def _score_by_correlation(self) -> Dict[str, float]:
        """
        Score features by correlation with target

        Returns
        -------
        dict
            Correlation scores
        """
        # Select only numeric columns
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()

        correlations = {}

        for col in numeric_cols:
            try:
                mask = ~(self.X[col].isna() | self.y.isna())
                if mask.sum() > 0:
                    corr = abs(self.X.loc[mask, col].corr(self.y[mask]))
                    correlations[col] = corr if not np.isnan(corr) else 0
                else:
                    correlations[col] = 0
            except:
                correlations[col] = 0

        # Add non-numeric columns with score 0
        for col in self.X.columns:
            if col not in correlations:
                correlations[col] = 0

        # Normalize
        total = sum(correlations.values())
        if total > 0:
            correlations = {k: v / total for k, v in correlations.items()}

        return correlations

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
