"""
FeatureSmith: Intelligent Feature Engineering and Selection
============================================================

A powerful library for automated feature engineering, selection,
and validation in machine learning workflows.

Main Components:
    - FeatureSmith: Main class for feature engineering
    - Generators: Create new features (polynomial, interactions, encoding)
    - Selectors: Select best features (importance, correlation, RFE)
    - Validators: Detect data leakage and validate features
    - Evaluators: Score and rank features

Quick Start:
    >>> from feature_forge import FeatureSmith
    >>> import pandas as pd
    >>>
    >>> # Initialize
    >>> smith = FeatureSmith(X_train, y_train)
    >>>
    >>> # Generate features
    >>> X_augmented = smith.forge(strategies=['polynomial', 'interactions'])
    >>>
    >>> # Rank features
    >>> ranked = smith.rank_features()
    >>>
    >>> # Remove redundancy
    >>> optimal = smith.remove_redundancy()

For more details, visit: https://github.com/AbhishekDP2244/feature-forge
"""
from importlib.metadata import version

__version__ = version("featuresmith")
__author__ = "AbhishekDP2244"
__email__ = "abhishekpanigrahi.work@gmail.com"

from .smith import FeatureSmith

__all__ = ["FeatureSmith", "__version__"]
