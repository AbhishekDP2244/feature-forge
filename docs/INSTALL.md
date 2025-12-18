# Installation Guide for FeatureForge

## 📦 Installation Options

### Option 1: Standard Installation (Recommended)

For normal usage with all features:

```bash
# Clone or navigate to the project
cd feature-forge

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### Option 2: Minimal Installation

For minimal footprint (basic functionality only):

```bash
pip install -r requirements-minimal.txt
pip install -e .
```

**Note**: With minimal installation, XGBoost and CatBoost won't be available. The package will automatically fall back to RandomForest.

### Option 3: Development Installation

For contributing or development:

```bash
pip install -r requirements-dev.txt
pip install -e .
```

This includes testing tools, code formatters, and documentation builders.

### Option 4: From PyPI (After Publishing)

Once published to PyPI:

```bash
# Basic installation
pip install feature-forge

# With all optional dependencies
pip install feature-forge[all]

# With development tools
pip install feature-forge[dev]

# With XGBoost support
pip install feature-forge[xgboost]

# With CatBoost support
pip install feature-forge[catboost]
```

## 🔍 Verify Installation

Test that everything is installed correctly:

```bash
# Test import
python -c "from feature_forge import FeatureSmith; print('✓ Import successful!')"

# Check version
python -c "import feature_forge; print(f'FeatureForge version: {feature_forge.__version__}')"

# Run tests
pytest tests/ -v

# Quick functionality test
python -c "
from feature_forge import FeatureSmith
import pandas as pd
import numpy as np

X = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
y = pd.Series([0,1,0])
smith = FeatureSmith(X, y)
print('✓ FeatureSmith initialized successfully!')
"
```

## 🐛 Troubleshooting

### Issue: "No module named 'feature_forge'"

**Solution**: Make sure you've installed the package:
```bash
pip install -e .
```

### Issue: "ImportError: No module named 'lightgbm'"

**Solution**: LightGBM requires specific system libraries. Try:

**Windows:**
```bash
pip install lightgbm
```

**Mac:**
```bash
brew install libomp
pip install lightgbm
```

**Linux:**
```bash
pip install lightgbm
```

If LightGBM fails to install, the package will work fine without it (falls back to RandomForest).

### Issue: "ModuleNotFoundError: No module named 'sklearn'"

**Solution**: Install scikit-learn:
```bash
pip install scikit-learn>=1.0.0
```

### Issue: "ImportError: cannot import name 'PolynomialFeatures'"

**Solution**: Update scikit-learn:
```bash
pip install --upgrade scikit-learn
```

### Issue: Tests fail with "matplotlib" errors

**Solution**: Install GUI backend for matplotlib:

**Windows/Mac:**
```bash
pip install pyqt5
```

**Linux:**
```bash
sudo apt-get install python3-tk
```

Or run tests in headless mode:
```bash
export MPLBACKEND=Agg  # Linux/Mac
set MPLBACKEND=Agg     # Windows
pytest tests/
```

## 📊 Dependency Overview

### Core Dependencies (Required)

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >=1.3.0 | Data manipulation |
| numpy | >=1.21.0 | Numerical operations |
| scikit-learn | >=1.0.0 | ML algorithms |
| scipy | >=1.7.0 | Scientific computing |
| lightgbm | >=3.0.0 | Gradient boosting |
| matplotlib | >=3.4.0 | Visualization |
| seaborn | >=0.11.0 | Statistical plots |

### Optional Dependencies

| Package | Version | Purpose | Install Command |
|---------|---------|---------|----------------|
| xgboost | >=1.5.0 | Alternative GB model | `pip install xgboost` |
| catboost | >=1.0.0 | Alternative GB model | `pip install catboost` |

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | >=7.0.0 | Testing framework |
| pytest-cov | >=4.0.0 | Coverage reporting |
| black | >=23.0.0 | Code formatter |
| flake8 | >=6.0.0 | Linter |
| mypy | >=1.0.0 | Type checker |

## 🔄 Updating Dependencies

### Check for outdated packages
```bash
pip list --outdated
```

### Update specific package
```bash
pip install --upgrade pandas
```

### Update all packages
```bash
pip install --upgrade -r requirements.txt
```

## 🌐 Platform-Specific Notes

### Windows
- Use `venv\Scripts\activate` to activate virtual environment
- Some packages may require Visual C++ Build Tools
- Install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

### macOS
- Use `source venv/bin/activate` to activate virtual environment
- May need to install Xcode Command Line Tools: `xcode-select --install`
- For M1/M2 Macs, ensure you're using ARM-compatible versions

### Linux
- Use `source venv/bin/activate` to activate virtual environment
- May need system packages: `sudo apt-get install python3-dev build-essential`
- For headless servers, set `MPLBACKEND=Agg` environment variable

## 📱 Kaggle Installation

To use in Kaggle notebooks:

```python
# Install from GitHub (after publishing)
!pip install git+https://github.com/yourusername/feature-forge.git

# Or upload wheel file and install
!pip install /kaggle/input/feature-forge/feature_forge-0.1.0-py3-none-any.whl

# Import and use
from feature_forge import FeatureSmith
```

## 🐳 Docker Installation

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy package
COPY . .

# Install package
RUN pip install -e .

CMD ["python"]
```

Build and run:
```bash
docker build -t feature-forge .
docker run -it feature-forge python
```

## ✅ Installation Checklist

After installation, verify:

- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip list`)
- [ ] Package importable (`from feature_forge import FeatureSmith`)
- [ ] Tests pass (`pytest tests/`)
- [ ] No import errors in Python


## 🚀 Next Steps

After successful installation:

1. Read the [QUICKSTART.md](QUICKSTART.md)
2. Check out the [README.md](README.md) for usage examples
3. Browse the `/examples` directory for more demos

---

**Installation successful? Start creating amazing features! 🔥**
