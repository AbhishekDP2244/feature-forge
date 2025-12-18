"""
DateTime feature generator
"""

import pandas as pd
import numpy as np
from typing import List


class DateTimeGenerator:
    """
    Generate datetime-based features

    Extracts temporal features from datetime columns including:
    - Year, month, day, day of week, hour, minute
    - Is weekend, is month start/end
    - Season, quarter
    - Time since epoch
    - Cyclical encodings

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
        Generate datetime features

        Parameters
        ----------
        max_features : int, default=20
            Maximum number of features to generate

        Returns
        -------
        pd.DataFrame
            Generated datetime features
        """
        # Find datetime columns
        datetime_cols = []

        for col in self.X.columns:
            if pd.api.types.is_datetime64_any_dtype(self.X[col]):
                datetime_cols.append(col)
            # Try to parse as datetime
            elif self.X[col].dtype == "object":
                try:
                    # Sample a few values to check
                    sample = self.X[col].dropna().head(10)
                    if len(sample) > 0:
                        pd.to_datetime(sample)
                        datetime_cols.append(col)
                except:
                    continue

        if len(datetime_cols) == 0:
            return pd.DataFrame(index=self.X.index)

        datetime_features = {}
        count = 0

        for col in datetime_cols[:2]:  # Limit to 2 datetime columns
            if count >= max_features:
                break

            try:
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(self.X[col]):
                    dt_col = pd.to_datetime(self.X[col], errors="coerce")
                else:
                    dt_col = self.X[col]

                # Skip if too many NaT values
                if dt_col.isna().sum() / len(dt_col) > 0.5:
                    continue

                # Year
                if count < max_features:
                    datetime_features[f"{col}_year"] = dt_col.dt.year
                    count += 1

                # Month
                if count < max_features:
                    datetime_features[f"{col}_month"] = dt_col.dt.month
                    count += 1

                # Day
                if count < max_features:
                    datetime_features[f"{col}_day"] = dt_col.dt.day
                    count += 1

                # Day of week (0=Monday, 6=Sunday)
                if count < max_features:
                    datetime_features[f"{col}_dayofweek"] = dt_col.dt.dayofweek
                    count += 1

                # Hour (if time component exists)
                if count < max_features and hasattr(dt_col.dt, "hour"):
                    hour = dt_col.dt.hour
                    if hour.notna().sum() > 0:
                        datetime_features[f"{col}_hour"] = hour
                        count += 1

                # Is weekend
                if count < max_features:
                    datetime_features[f"{col}_is_weekend"] = (
                        dt_col.dt.dayofweek >= 5
                    ).astype(int)
                    count += 1

                # Quarter
                if count < max_features:
                    datetime_features[f"{col}_quarter"] = dt_col.dt.quarter
                    count += 1

                # Is month start
                if count < max_features:
                    datetime_features[f"{col}_is_month_start"] = (
                        dt_col.dt.is_month_start.astype(int)
                    )
                    count += 1

                # Is month end
                if count < max_features:
                    datetime_features[f"{col}_is_month_end"] = (
                        dt_col.dt.is_month_end.astype(int)
                    )
                    count += 1

                # Days since epoch (Unix timestamp)
                if count < max_features:
                    epoch = pd.Timestamp("1970-01-01")
                    datetime_features[f"{col}_days_since_epoch"] = (
                        dt_col - epoch
                    ).dt.days
                    count += 1

                # Cyclical encoding for month (sin/cos)
                if count < max_features + 1:  # Need 2 features
                    month = dt_col.dt.month
                    datetime_features[f"{col}_month_sin"] = np.sin(
                        2 * np.pi * month / 12
                    )
                    datetime_features[f"{col}_month_cos"] = np.cos(
                        2 * np.pi * month / 12
                    )
                    count += 2

                # Cyclical encoding for day of week
                if count < max_features + 1:  # Need 2 features
                    dow = dt_col.dt.dayofweek
                    datetime_features[f"{col}_dow_sin"] = np.sin(2 * np.pi * dow / 7)
                    datetime_features[f"{col}_dow_cos"] = np.cos(2 * np.pi * dow / 7)
                    count += 2

            except Exception as e:
                # Skip this datetime column if error
                continue

        if len(datetime_features) == 0:
            return pd.DataFrame(index=self.X.index)

        return pd.DataFrame(datetime_features, index=self.X.index)
