# FeatureSmith Quick Start Guide

## 🚀 Installation & Testing

### Step 1: Navigate to Project Directory

```bash
cd D:\My Project\feature-forge
```

### Step 2: Install Dependencies

```bash
# Make sure virtual environment is activated
# If not, activate it:
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install the package in development mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black flake8
```

### Step 3: Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=feature_forge --cov-report=html

# Run specific test
python -m pytest tests/test_smith.py::test_forge_polynomial -v
```

### Step 4: Test the Package Manually

Create a file `test_manual.py` in the root directory:

```python
from feature_forge import FeatureSmith
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
X = pd.DataFrame({
    'age': np.random.randint(18, 80, 100),
    'salary': np.random.randint(20000, 150000, 100),
    'experience': np.random.randint(0, 30, 100),
    'department': np.random.choice(['Sales', 'IT', 'HR', 'Finance'], 100),
    'city': np.random.choice(['NYC', 'LA', 'Chicago'], 100)
})
y = pd.Series(np.random.randint(0, 2, 100))

print("=" * 50)
print("Testing FeatureSmith")
print("=" * 50)

# Initialize
print("\n1. Initializing FeatureSmith...")
smith = FeatureSmith(X, y, task='classification')

# Generate features
print("\n2. Generating features...")
X_augmented = smith.forge(
    strategies=['polynomial', 'interactions', 'encoding'],
    max_features=30
)

print(f"   Original features: {len(X.columns)}")
print(f"   Generated features: {len(X_augmented.columns) - len(X.columns)}")
print(f"   Total features: {len(X_augmented.columns)}")

# Rank features
print("\n3. Ranking features...")
ranked = smith.rank_features(model_type='rf')
print("\n   Top 10 features:")
print(ranked.head(10))

# Remove redundancy
print("\n4. Removing redundant features...")
optimal = smith.remove_redundancy(threshold=0.95)
print(f"   Selected {len(optimal)} non-redundant features")

# Generate report
print("\n5. Generating HTML report...")
smith.generate_report('test_report.html')
print("   Report saved to: test_report.html")

# Summary
print("\n6. Summary:")
summary = smith.get_summary()
for key, value in summary.items():
    print(f"   {key}: {value}")

print("\n" + "=" * 50)
print("✓ All tests completed successfully!")
print("=" * 50)
```

Run it:

```bash
python test_manual.py
```

### Step 5: Build Distribution

```bash
# Build the package
python -m build

# This creates:
# - dist/feature_forge-0.1.0.tar.gz
# - dist/feature_forge-0.1.0-py3-none-any.whl
```

### Step 6: Test Installation from Wheel

```bash
# Create a fresh environment
python -m venv test_env
test_env\Scripts\activate  # Windows

# Install from wheel
pip install dist/feature_forge-0.1.0-py3-none-any.whl

# Test import
python -c "from feature_forge import FeatureSmith; print('✓ Import successful!')"

# Deactivate
deactivate
```

## 📊 Creating a Demo Notebook

Create `examples/demo.py`:

```python
"""
FeatureSmith Demo - Titanic Dataset
"""

from feature_forge import FeatureSmith
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load Titanic dataset (or create sample data)
np.random.seed(42)
n_samples = 200

X = pd.DataFrame({
    'Pclass': np.random.choice([1, 2, 3], n_samples),
    'Age': np.random.normal(30, 10, n_samples),
    'SibSp': np.random.poisson(0.5, n_samples),
    'Parch': np.random.poisson(0.3, n_samples),
    'Fare': np.random.exponential(30, n_samples),
    'Sex': np.random.choice(['male', 'female'], n_samples),
    'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples)
})

y = pd.Series(np.random.randint(0, 2, n_samples))

print("🚢 Titanic Survival Prediction - FeatureSmith Demo")
print("=" * 60)

# Original features baseline
print("\n1️⃣  BASELINE (Original Features)")
print("-" * 60)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_scores = cross_val_score(rf, X.select_dtypes(include=[np.number]), y, cv=5, scoring='accuracy')
print(f"   CV Accuracy: {baseline_scores.mean():.4f} (+/- {baseline_scores.std():.4f})")

# Initialize FeatureSmith
print("\n2️⃣  FEATURE ENGINEERING")
print("-" * 60)
smith = FeatureSmith(X, y, task='classification', verbose=True)

# Generate features
X_augmented = smith.forge(
    strategies=['polynomial', 'interactions', 'encoding'],
    max_features=50,
    validate=True
)

# Rank features
print("\n3️⃣  FEATURE RANKING")
print("-" * 60)
ranked = smith.rank_features(model_type='rf', method='importance')
print("\nTop 15 Features:")
print(ranked.head(15).to_string(index=False))

# Select top features
top_features = ranked.head(30)['feature'].tolist()

# Evaluate with engineered features
print("\n4️⃣  EVALUATION WITH ENGINEERED FEATURES")
print("-" * 60)
engineered_scores = cross_val_score(rf, X_augmented[top_features], y, cv=5, scoring='accuracy')
print(f"   CV Accuracy: {engineered_scores.mean():.4f} (+/- {engineered_scores.std():.4f})")

# Improvement
improvement = (engineered_scores.mean() - baseline_scores.mean()) / baseline_scores.mean() * 100
print(f"\n   📈 Improvement: {improvement:+.2f}%")

# Remove redundancy
print("\n5️⃣  REDUNDANCY REMOVAL")
print("-" * 60)
optimal_features = smith.remove_redundancy(X=X_augmented[top_features], threshold=0.95)
print(f"   Reduced from {len(top_features)} to {len(optimal_features)} features")

# Final evaluation
final_scores = cross_val_score(rf, X_augmented[optimal_features], y, cv=5, scoring='accuracy')
print(f"   Final CV Accuracy: {final_scores.mean():.4f} (+/- {final_scores.std():.4f})")

# Generate report
print("\n6️⃣  GENERATING REPORT")
print("-" * 60)
smith.generate_report('titanic_feature_report.html')

print("\n" + "=" * 60)
print("✅ Demo completed! Check 'titanic_feature_report.html' for details")
print("=" * 60)
```

Run it:

```bash
python examples/demo.py
```

## 🐛 Troubleshooting

### Import Error

```bash
# Make sure package is installed
pip install -e .

# Or reinstall
pip uninstall featuresmith
pip install -e .
```

### Missing Dependencies

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn scipy
```

### Test Failures

```bash
# Run specific test with more details
python -m pytest tests/test_smith.py::test_forge_polynomial -vv

# Check if all dependencies are installed
pip list
```

### LightGBM Not Found

LightGBM is optional. The package will fall back to RandomForest if LightGBM is not available:

```bash
pip install lightgbm
```

## 📦 Next Steps

1. **Customize for your needs**: Modify generators or add new ones
2. **Create examples**: Build example notebooks for different use cases
3. **Add tests**: Write more comprehensive tests
4. **Documentation**: Create full documentation with Sphinx
5. **Publish**: Upload to PyPI (see main guide)

## 🎯 Ready for Production

Once you've tested everything:

1. Update version in `pyproject.toml` and `setup.py`
2. Update `README.md` with your information
3. Build: `python -m build`
4. Upload to TestPyPI first to verify
5. Upload to PyPI
6. Share on Kaggle, GitHub, LinkedIn!

---

**Happy Feature Engineering! 🔥**