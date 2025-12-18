# Contributing to FeatureForge

Thank you for your interest in contributing! 🎉

## How to Contribute

1. Fork the repository
2. Create a branch (e.g., `git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup
```bash
git clone https://github.com/AbhishekDP2244/feature-forge.git
cd feature-forge
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
```

## Running Tests
```bash
pytest tests/ -v
```

## Code Style

We use Black for formatting:
```bash
black src/
```

## Questions?

Open an issue or reach out at [abhishekpanigrahi.work@gmail.com]
