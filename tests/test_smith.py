"""
Tests for FeatureSmith class
"""

import pytest
import pandas as pd
import numpy as np
from feature_forge import FeatureSmith


@pytest.fixture
def sample_classification_data():
    """Create sample classification dataset"""
    np.random.seed(42)
    X = pd.DataFrame({
        'num1': np.random.randn(100),
        'num2': np.random.randn(100),
        'num3': np.random.randn(100),
        'cat1': np.random.choice(['A', 'B', 'C'], 100),
        'cat2': np.random.choice(['X', 'Y'], 100)
    })
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y


@pytest.fixture
def sample_regression_data():
    """Create sample regression dataset"""
    np.random.seed(42)
    X = pd.DataFrame({
        'num1': np.random.randn(100),
        'num2': np.random.randn(100),
        'num3': np.random.randn(100),
    })
    y = pd.Series(np.random.randn(100) * 10 + 50)
    return X, y


def test_initialization_classification(sample_classification_data):
    """Test FeatureSmith initialization with classification data"""
    X, y = sample_classification_data
    smith = FeatureSmith(X, y, task='auto')
    
    assert smith.task == 'classification'
    assert len(smith.X) == 100
    assert len(smith.y) == 100


def test_initialization_regression(sample_regression_data):
    """Test FeatureSmith initialization with regression data"""
    X, y = sample_regression_data
    smith = FeatureSmith(X, y, task='auto')
    
    assert smith.task == 'regression'
    assert len(smith.X) == 100


def test_forge_polynomial(sample_classification_data):
    """Test polynomial feature generation"""
    X, y = sample_classification_data
    smith = FeatureSmith(X, y, verbose=False)
    
    X_augmented = smith.forge(strategies=['polynomial'], max_features=10)
    
    assert X_augmented.shape[1] > X.shape[1]
    assert X_augmented.shape[0] == X.shape[0]
    assert 'polynomial' in smith.generated_features


def test_forge_interactions(sample_classification_data):
    """Test interaction feature generation"""
    X, y = sample_classification_data
    smith = FeatureSmith(X, y, verbose=False)
    
    X_augmented = smith.forge(strategies=['interactions'], max_features=10)
    
    assert X_augmented.shape[1] > X.shape[1]
    assert 'interactions' in smith.generated_features


def test_forge_encoding(sample_classification_data):
    """Test encoding feature generation"""
    X, y = sample_classification_data
    smith = FeatureSmith(X, y, verbose=False)
    
    X_augmented = smith.forge(strategies=['encoding'], max_features=10)
    
    assert X_augmented.shape[1] > X.shape[1]
    assert 'encoding' in smith.generated_features


def test_forge_multiple_strategies(sample_classification_data):
    """Test multiple feature generation strategies"""
    X, y = sample_classification_data
    smith = FeatureSmith(X, y, verbose=False)
    
    X_augmented = smith.forge(
        strategies=['polynomial', 'interactions', 'encoding'],
        max_features=20
    )
    
    assert X_augmented.shape[1] > X.shape[1]
    assert len(smith.generated_features) > 0


def test_rank_features(sample_classification_data):
    """Test feature ranking"""
    X, y = sample_classification_data
    smith = FeatureSmith(X, y, verbose=False)
    
    X_augmented = smith.forge(strategies=['polynomial'], max_features=5)
    ranked = smith.rank_features(model_type='rf')
    
    assert len(ranked) > 0
    assert 'feature' in ranked.columns
    assert 'score' in ranked.columns
    assert ranked['score'].iloc[0] >= ranked['score'].iloc[-1]  # Check sorted


def test_remove_redundancy(sample_classification_data):
    """Test redundancy removal"""
    X, y = sample_classification_data
    
    # Add redundant feature
    X['redundant'] = X['num1'] * 1.001
    
    smith = FeatureSmith(X, y, verbose=False)
    selected = smith.remove_redundancy(threshold=0.99)
    
    # Either 'redundant' or 'num1' should be removed
    assert 'redundant' not in selected or 'num1' not in selected


def test_max_features_limit(sample_classification_data):
    """Test that max_features limit is respected"""
    X, y = sample_classification_data
    smith = FeatureSmith(X, y, verbose=False)
    
    max_features = 15
    X_augmented = smith.forge(strategies=['polynomial', 'interactions'], max_features=max_features)
    
    n_generated = X_augmented.shape[1] - X.shape[1]
    assert n_generated <= max_features


def test_empty_dataframe():
    """Test handling of empty dataframe"""
    X = pd.DataFrame()
    y = pd.Series()
    
    with pytest.raises(ValueError):
        smith = FeatureSmith(X, y)


def test_mismatched_lengths():
    """Test handling of mismatched X and y lengths"""
    X = pd.DataFrame({'a': [1, 2, 3]})
    y = pd.Series([1, 2])
    
    with pytest.raises(ValueError):
        smith = FeatureSmith(X, y)


def test_get_summary(sample_classification_data):
    """Test summary generation"""
    X, y = sample_classification_data
    smith = FeatureSmith(X, y, verbose=False)
    
    X_augmented = smith.forge(strategies=['polynomial'], max_features=5)
    summary = smith.get_summary()
    
    assert 'task' in summary
    assert 'original_features' in summary
    assert 'samples' in summary
    assert summary['original_features'] == len(X.columns)


def test_validate_features(sample_classification_data):
    """Test feature validation for leakage"""
    X, y = sample_classification_data
    smith = FeatureSmith(X, y, verbose=False)
    
    # Generate features with validation enabled
    X_augmented = smith.forge(
        strategies=['polynomial'],
        max_features=10,
        validate=True
    )
    
    # Should successfully generate features without errors
    assert X_augmented.shape[1] >= X.shape[1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])