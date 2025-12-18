"""
Data leakage detector
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, r2_score


class LeakDetector:
    """
    Detect potential data leakage in features

    A feature is considered potentially leaky if a simple decision tree
    can achieve unrealistically high performance using only that feature.

    Parameters
    ----------
    X : pd.DataFrame
        Input features
    y : pd.Series
        Target variable
    threshold : float, default=0.98
        Performance threshold above which a feature is considered leaky
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.98):
        self.X = X
        self.y = y
        self.threshold = threshold

    def is_leaky(self, feature: pd.Series) -> bool:
        """
        Check if a single feature is potentially leaky

        Parameters
        ----------
        feature : pd.Series
            Feature to check

        Returns
        -------
        bool
            True if feature is potentially leaky
        """
        # Remove NaN values
        mask = ~(feature.isna() | self.y.isna())

        if mask.sum() < 10:
            return True  # Not enough data, mark as suspicious

        X_clean = feature[mask].values.reshape(-1, 1)
        y_clean = self.y[mask]

        # Check if feature is constant
        if feature.nunique() <= 1:
            return True

        # Check if feature perfectly predicts target
        if len(X_clean) == y_clean.nunique():
            # If each value maps to unique target, likely leaky
            if feature.nunique() == y_clean.nunique():
                return True

        # Train simple decision tree
        try:
            if self._is_classification():
                model = DecisionTreeClassifier(max_depth=3, random_state=42)
                model.fit(X_clean, y_clean)

                # Check if binary classification
                if len(np.unique(y_clean)) == 2:
                    pred_proba = model.predict_proba(X_clean)[:, 1]
                    score = roc_auc_score(y_clean, pred_proba)
                else:
                    # Multi-class: use accuracy
                    score = model.score(X_clean, y_clean)
            else:
                model = DecisionTreeRegressor(max_depth=3, random_state=42)
                model.fit(X_clean, y_clean)
                pred = model.predict(X_clean)
                score = r2_score(y_clean, pred)
                # For regression, perfect R2 of 1.0 is suspicious
                if score > 0.99:
                    return True

            # Check against threshold
            return score > self.threshold

        except Exception:
            # If error occurs, mark as not leaky (be conservative)
            return False

    def check_all_features(self) -> pd.DataFrame:
        """
        Check all features for potential leakage

        Returns
        -------
        pd.DataFrame
            Results with feature names and leak status
        """
        results = []

        for col in self.X.columns:
            is_leaky = self.is_leaky(self.X[col])
            results.append({"feature": col, "is_leaky": is_leaky})

        return pd.DataFrame(results)

    def _is_classification(self) -> bool:
        """
        Check if task is classification

        Returns
        -------
        bool
            True if classification task
        """
        return (
            self.y.dtype in ["object", "category", "bool"]
            or self.y.nunique() / len(self.y) < 0.05
        )
