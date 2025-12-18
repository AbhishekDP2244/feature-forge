"""
Main FeatureSmith class for feature engineering
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
import warnings


class FeatureSmith:
    """
    Intelligent feature engineering and selection

    The FeatureSmith class provides automated feature generation, selection,
    and validation capabilities for machine learning workflows.

    Parameters
    ----------
    X : pd.DataFrame
        Training features
    y : pd.Series or np.ndarray
        Target variable
    task : str, default='auto'
        Task type: 'classification', 'regression', or 'auto' (auto-detect)
    n_jobs : int, default=-1
        Number of parallel jobs for computation
    random_state : int, default=42
        Random state for reproducibility
    verbose : bool, default=True
        Whether to print progress messages

    Attributes
    ----------
    generated_features : dict
        Dictionary storing generated features by strategy
    feature_scores : dict
        Dictionary storing feature importance scores
    task : str
        Detected or specified task type

    Examples
    --------
    >>> from feature_forge import FeatureSmith
    >>> import pandas as pd
    >>>
    >>> # Create sample data
    >>> X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> y = pd.Series([0, 1, 0])
    >>>
    >>> # Initialize FeatureSmith
    >>> smith = FeatureSmith(X, y)
    >>>
    >>> # Generate features
    >>> X_new = smith.forge(strategies=['polynomial'])
    >>>
    >>> # Rank features
    >>> ranked = smith.rank_features()
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        task: str = "auto",
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: bool = True,
    ):
        # Input validation
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if len(X) == 0:
            raise ValueError("X cannot be empty")

        # Store data
        self.X = X.copy()
        self.y = y if isinstance(y, pd.Series) else pd.Series(y, index=X.index)

        # Check dimensions match
        if len(self.X) != len(self.y):
            raise ValueError(
                f"X and y must have same length. Got {len(self.X)} and {len(self.y)}"
            )

        # Parameters
        self.task = self._detect_task() if task == "auto" else task
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        # Storage for generated features and scores
        self.generated_features = {}
        self.feature_scores = {}
        self.X_augmented = None

        if self.verbose:
            print(f"FeatureSmith initialized")
            print(f"  - Task: {self.task}")
            print(f"  - Samples: {len(self.X)}")
            print(f"  - Original features: {len(self.X.columns)}")

    def _detect_task(self) -> str:
        """
        Auto-detect if classification or regression task

        Returns
        -------
        str
            'classification' or 'regression'
        """
        # Check data type
        if self.y.dtype in ["object", "category", "bool"]:
            return "classification"

        # Check unique ratio
        unique_ratio = self.y.nunique() / len(self.y)

        # If less than 5% unique values, likely classification
        if unique_ratio < 0.05:
            return "classification"

        # Check if target has integer values in small range
        if self.y.dtype in ["int32", "int64"]:
            if self.y.nunique() <= 20:
                return "classification"

        return "regression"

    def forge(
        self,
        strategies: List[str] = ["polynomial", "interactions"],
        max_features: int = 50,
        validate: bool = True,
    ) -> pd.DataFrame:
        """
        Generate new features using specified strategies

        Parameters
        ----------
        strategies : list of str, default=['polynomial', 'interactions']
            Feature generation strategies to use. Available strategies:
            - 'polynomial': Polynomial features (x^2, x^3, etc.)
            - 'interactions': Feature interactions (x1*x2, x1/x2, etc.)
            - 'aggregations': Group-by aggregations
            - 'datetime': DateTime feature extraction
            - 'encoding': Target and frequency encoding
        max_features : int, default=50
            Maximum number of new features to generate
        validate : bool, default=True
            Whether to validate features for data leakage

        Returns
        -------
        pd.DataFrame
            Original features plus generated features

        Examples
        --------
        >>> X_augmented = smith.forge(strategies=['polynomial', 'interactions'])
        >>> print(f"Generated {X_augmented.shape[1] - X.shape[1]} new features")
        """
        from .generators import (
            PolynomialGenerator,
            InteractionGenerator,
            AggregationGenerator,
            DateTimeGenerator,
            EncodingGenerator,
        )

        # Map strategy names to generator classes
        generator_map = {
            "polynomial": PolynomialGenerator,
            "interactions": InteractionGenerator,
            "aggregations": AggregationGenerator,
            "datetime": DateTimeGenerator,
            "encoding": EncodingGenerator,
        }

        if self.verbose:
            print(f"\nForging features with strategies: {strategies}")

        X_augmented = self.X.copy()
        feature_count = 0

        for strategy in strategies:
            if strategy not in generator_map:
                warnings.warn(f"Unknown strategy: {strategy}. Skipping.")
                continue

            if self.verbose:
                print(f"\n  Generating {strategy} features...")

            try:
                # Initialize generator
                generator = generator_map[strategy](self.X, self.y, self.task)

                # Generate features
                remaining_budget = max_features - feature_count
                new_features = generator.generate(max_features=remaining_budget)

                if len(new_features.columns) == 0:
                    if self.verbose:
                        print(f"    No features generated for {strategy}")
                    continue

                # Validate features for leakage if requested
                if validate:
                    new_features = self._validate_features(new_features)

                # Store generated feature names
                self.generated_features[strategy] = new_features.columns.tolist()

                # Add to augmented dataset
                X_augmented = pd.concat([X_augmented, new_features], axis=1)
                feature_count = len(X_augmented.columns) - len(self.X.columns)

                if self.verbose:
                    print(f"    Generated: {len(new_features.columns)} features")
                    print(f"    Total new features: {feature_count}")

                # Stop if we've reached max_features
                if feature_count >= max_features:
                    if self.verbose:
                        print(f"\n  Reached maximum feature limit ({max_features})")
                    break

            except Exception as e:
                warnings.warn(f"Error generating {strategy} features: {str(e)}")
                continue

        self.X_augmented = X_augmented

        if self.verbose:
            print(f"\n✓ Feature generation complete!")
            print(f"  - Original features: {len(self.X.columns)}")
            print(f"  - New features: {feature_count}")
            print(f"  - Total features: {len(X_augmented.columns)}")

        return X_augmented

    def rank_features(
        self,
        X: Optional[pd.DataFrame] = None,
        model_type: str = "lgbm",
        method: str = "importance",
        cv: int = 5,
    ) -> pd.DataFrame:
        """
        Rank features by importance

        Parameters
        ----------
        X : pd.DataFrame, optional
            Feature matrix to rank. If None, uses augmented features from forge()
        model_type : str, default='lgbm'
            Model to use for ranking: 'lgbm', 'xgb', 'rf', 'tree'
        method : str, default='importance'
            Ranking method: 'importance', 'permutation', 'shap'
        cv : int, default=5
            Number of cross-validation folds

        Returns
        -------
        pd.DataFrame
            Ranked features with scores

        Examples
        --------
        >>> ranked = smith.rank_features(model_type='lgbm')
        >>> print(ranked.head(10))
        """
        from .evaluators import FeatureScorer

        # Use augmented features if available, otherwise use original
        if X is None:
            if self.X_augmented is not None:
                X = self.X_augmented
            else:
                X = self.X

        if self.verbose:
            print(f"\nRanking {len(X.columns)} features using {model_type}...")

        scorer = FeatureScorer(X, self.y, self.task, verbose=self.verbose)
        scores = scorer.score_features(model_type, method, cv)

        self.feature_scores = scores

        # Create ranked dataframe
        ranked_df = (
            pd.DataFrame(
                {"feature": list(scores.keys()), "score": list(scores.values())}
            )
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

        if self.verbose:
            print(f"✓ Feature ranking complete!")
            print(f"\nTop 5 features:")
            for idx, row in ranked_df.head().iterrows():
                print(f"  {idx+1}. {row['feature']}: {row['score']:.4f}")

        return ranked_df

    def remove_redundancy(
        self,
        X: Optional[pd.DataFrame] = None,
        threshold: float = 0.95,
        method: str = "correlation",
    ) -> List[str]:
        """
        Remove redundant features based on correlation

        Parameters
        ----------
        X : pd.DataFrame, optional
            Feature matrix. If None, uses augmented features from forge()
        threshold : float, default=0.95
            Correlation threshold for considering features redundant
        method : str, default='correlation'
            Method for detecting redundancy: 'correlation' or 'mutual_information'

        Returns
        -------
        list of str
            Selected non-redundant feature names

        Examples
        --------
        >>> optimal = smith.remove_redundancy(threshold=0.95)
        >>> print(f"Selected {len(optimal)} non-redundant features")
        """
        from .selectors import RedundancyRemover

        # Use augmented features if available
        if X is None:
            if self.X_augmented is not None:
                X = self.X_augmented
            else:
                X = self.X

        if self.verbose:
            print(f"\nRemoving redundant features (threshold={threshold})...")

        remover = RedundancyRemover(X, self.y, threshold, method, verbose=self.verbose)
        selected_features = remover.select()

        if self.verbose:
            removed = len(X.columns) - len(selected_features)
            print(f"✓ Redundancy removal complete!")
            print(f"  - Original: {len(X.columns)} features")
            print(f"  - Removed: {removed} redundant features")
            print(f"  - Remaining: {len(selected_features)} features")

        return selected_features

    def _validate_features(self, new_features: pd.DataFrame) -> pd.DataFrame:
        """
        Validate features for potential data leakage

        Parameters
        ----------
        new_features : pd.DataFrame
            Features to validate

        Returns
        -------
        pd.DataFrame
            Validated features (leaky features removed)
        """
        from .validators import LeakDetector

        detector = LeakDetector(new_features, self.y, threshold=0.98)
        valid_features = []
        leaky_features = []

        for col in new_features.columns:
            if not detector.is_leaky(new_features[col]):
                valid_features.append(col)
            else:
                leaky_features.append(col)

        if leaky_features and self.verbose:
            print(f"    ⚠ Removed {len(leaky_features)} potentially leaky features")

        return new_features[valid_features]

    def generate_report(
        self, output_path: str = "feature_report.html", X: Optional[pd.DataFrame] = None
    ):
        """
        Generate comprehensive HTML feature engineering report

        Parameters
        ----------
        output_path : str, default='feature_report.html'
            Path to save the HTML report
        X : pd.DataFrame, optional
            Feature matrix to report on. If None, uses augmented features

        Examples
        --------
        >>> smith.generate_report('my_report.html')
        """
        from .utils import ReportGenerator

        if X is None:
            if self.X_augmented is not None:
                X = self.X_augmented
            else:
                X = self.X

        if self.verbose:
            print(f"\nGenerating feature report...")

        generator = ReportGenerator(
            X_original=self.X,
            X_augmented=X,
            y=self.y,
            generated_features=self.generated_features,
            feature_scores=self.feature_scores,
            task=self.task,
        )

        generator.create_html_report(output_path)

        if self.verbose:
            print(f"✓ Report saved to: {output_path}")

    def get_summary(self) -> Dict:
        """
        Get summary statistics of feature engineering process

        Returns
        -------
        dict
            Summary statistics
        """
        summary = {
            "task": self.task,
            "original_features": len(self.X.columns),
            "samples": len(self.X),
            "target_type": str(self.y.dtype),
            "strategies_used": list(self.generated_features.keys()),
        }

        if self.X_augmented is not None:
            summary["total_features"] = len(self.X_augmented.columns)
            summary["generated_features"] = len(self.X_augmented.columns) - len(
                self.X.columns
            )

        if self.feature_scores:
            summary["features_scored"] = len(self.feature_scores)

        return summary
