# FeatureForge 🔥

**Intelligent feature engineering and selection for machine learning competitions and projects.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub release](https://img.shields.io/github/v/release/YOUR-USERNAME/feature-forge)](https://github.com/AbhishekDP2244/feature-forge/releases)
[![GitHub stars](https://img.shields.io/github/stars/YOUR-USERNAME/feature-forge)](https://github.com/AbhishekDP2244/feature-forge/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/YOUR-USERNAME/feature-forge)](https://github.com/AbhishekDP2244/feature-forge/issues)

FeatureForge automates the tedious process of feature engineering, allowing you to focus on model building and experimentation. Perfect for Kaggle competitions, ML projects, and rapid prototyping.

## ✨ Features

- **🎯 Smart Feature Generation**
  - Polynomial features with automatic selection
  - Feature interactions (multiplication, division, addition, subtraction)
  - Target encoding with cross-validation to prevent overfitting
  - Frequency and count encoding
  - DateTime feature extraction
  - Group-by aggregations

- **📊 Intelligent Selection**
  - Multiple importance calculation methods (model-based, permutation, correlation)
  - Correlation-based redundancy removal
  - Recursive feature elimination (RFE)
  - Stability checking across CV folds

- **🛡️ Built-in Validation**
  - Automatic data leakage detection
  - Feature stability analysis
  - Quality checks for generated features

- **📈 Comprehensive Reporting**
  - Beautiful HTML reports with visualizations
  - Feature importance rankings
  - Generation statistics and insights

## 🚀 Quick Start

### Installation

```bash
pip install feature-forge
```

### Basic Usage

```python
from feature_forge import FeatureSmith
import pandas as pd

# Load your data
X_train = pd.read_csv('train.csv')
y_train = X_train['target']
X_train = X_train.drop('target', axis=1)

# Initialize FeatureSmith
smith = FeatureSmith(X_train, y_train, task='auto')

# Generate features
X_augmented = smith.forge(
    strategies=['polynomial', 'interactions', 'encoding'],
    max_features=50
)

# Rank features by importance
ranked_features = smith.rank_features(model_type='lgbm')
print(ranked_features.head(10))

# Remove redundant features
optimal_features = smith.remove_redundancy(threshold=0.95)

# Generate comprehensive report
smith.generate_report('feature_report.html')
```

## 📚 Documentation

### FeatureSmith Class

The main class for feature engineering operations.

**Parameters:**
- `X` (pd.DataFrame): Training features
- `y` (pd.Series): Target variable
- `task` (str): 'classification', 'regression', or 'auto' (default: 'auto')
- `n_jobs` (int): Number of parallel jobs (default: -1)
- `random_state` (int): Random seed (default: 42)
- `verbose` (bool): Print progress messages (default: True)

### Methods

#### `forge(strategies, max_features, validate)`

Generate new features using specified strategies.

```python
X_augmented = smith.forge(
    strategies=['polynomial', 'interactions', 'encoding'],
    max_features=50,
    validate=True  # Check for data leakage
)
```

**Available Strategies:**
- `'polynomial'`: Polynomial features (x², x³, interactions)
- `'interactions'`: Mathematical interactions (×, ÷, +, -)
- `'encoding'`: Target and frequency encoding for categorical features
- `'aggregations'`: Group-by aggregations
- `'datetime'`: DateTime feature extraction

#### `rank_features(model_type, method, cv)`

Rank features by importance.

```python
ranked = smith.rank_features(
    model_type='lgbm',  # 'lgbm', 'xgb', 'rf', 'tree'
    method='importance',  # 'importance', 'permutation', 'correlation'
    cv=5
)
```

#### `remove_redundancy(threshold, method)`

Remove highly correlated features.

```python
optimal = smith.remove_redundancy(
    threshold=0.95,
    method='correlation'  # 'correlation' or 'mutual_information'
)
```

#### `generate_report(output_path)`

Create a comprehensive HTML report.

```python
smith.generate_report('feature_report.html')
```

## 🎯 Advanced Examples

### Example 1: Kaggle Competition Pipeline

```python
from feature_forge import FeatureSmith
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier

# Initialize
smith = FeatureSmith(X_train, y_train, task='classification')

# Generate features
X_augmented = smith.forge(
    strategies=['polynomial', 'interactions', 'encoding'],
    max_features=100
)

# Get top features
ranked = smith.rank_features(model_type='lgbm')
top_50_features = ranked.head(50)['feature'].tolist()

# Train model with selected features
model = LGBMClassifier(n_estimators=1000)
scores = cross_val_score(
    model, 
    X_augmented[top_50_features], 
    y_train, 
    cv=5, 
    scoring='roc_auc'
)

print(f"CV AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Example 2: Feature Engineering for Time Series

```python
# For datetime features
smith = FeatureSmith(X_train, y_train)

# Extract datetime features
X_augmented = smith.forge(
    strategies=['datetime', 'aggregations'],
    max_features=30
)

# Generate report with insights
smith.generate_report('timeseries_features.html')
```

### Example 3: Handling High-Cardinality Categorical Features

```python
# Target encoding with validation
smith = FeatureSmith(X_train, y_train)

X_encoded = smith.forge(
    strategies=['encoding'],
    max_features=20,
    validate=True  # Detect leaky encodings
)

# Remove redundant encoded features
optimal = smith.remove_redundancy(threshold=0.9)
```

## 🔧 Advanced Configuration

### Custom Feature Selection

```python
from feature_forge.selectors import ImportanceSelector, RecursiveSelector

# Importance-based selection
importance_selector = ImportanceSelector(
    X_augmented, 
    y_train, 
    task='classification',
    n_features=30
)
selected = importance_selector.select(model_type='lgbm')

# RFE-based selection
rfe_selector = RecursiveSelector(
    X_augmented,
    y_train,
    task='classification',
    n_features=20,
    step=5
)
selected = rfe_selector.select()
```

### Data Leakage Detection

```python
from feature_forge.validators import LeakDetector

detector = LeakDetector(X_train, y_train, threshold=0.98)

# Check all features
leak_report = detector.check_all_features()
print(leak_report[leak_report['is_leaky']])
```

### Feature Stability Analysis

```python
from feature_forge.validators import StabilityChecker

checker = StabilityChecker(X_train, y_train, task='classification')
stability = checker.check_stability(model_type='lgbm')

# Features with low CV (coefficient of variation) are more stable
stable_features = stability[stability['cv_importance'] < 0.3]
```

## 💡 Tips & Best Practices

1. **Start Simple**: Begin with `['polynomial', 'interactions']` before adding more complex strategies

2. **Monitor Leakage**: Always use `validate=True` when generating features, especially with encoding strategies

3. **Feature Budget**: Set `max_features` based on your data size (rule of thumb: sqrt(n_samples))

4. **Iteration**: Generate features → rank → select → train → repeat

5. **Documentation**: Always generate and save the HTML report for reproducibility

6. **Cross-Validation**: Use stratified k-fold CV to validate your feature engineering pipeline

## 📊 Example Output

After running `smith.generate_report()`, you'll get a comprehensive HTML report showing:

- Summary statistics (original vs augmented features)
- Feature generation breakdown by strategy
- Top 20 features by importance with interactive charts
- Dataset statistics and missing value analysis
- Visualization of feature correlations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

**Development Setup:**

```bash
# Clone repository
git clone https://github.com/yourusername/feature-forge.git
cd feature-forge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the Kaggle community and countless hours of feature engineering
- Built with scikit-learn, pandas, and LightGBM
- Special thanks to all contributors

## 📬 Contact

- **GitHub**: [2244shek](https://github.com/2244shek)
- **Email**: abhishekpanigrahi.work@gmail.com
- **Kaggle**: [abhishekdp2244](https://www.kaggle.com/abhishekdp2244)

## ⭐ Star History

If you find FeatureForge useful, please consider giving it a star! It helps others discover the project.

---

**Made with ❤️ for the ML community**
