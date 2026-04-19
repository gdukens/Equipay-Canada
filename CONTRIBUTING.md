# Contributing to EquiPay Canada

Thank you for your interest in contributing to EquiPay Canada! This document provides guidelines and information for contributors.

##  Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/equipay-canada.git
   cd equipay-canada
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

4. **Install pre-commit hooks**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

5. **Run tests to verify setup**
   ```bash
   pytest tests/ -v
   ```

##  Code Style Guidelines

### Python Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting (line length: 120)
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Add type hints for function parameters and return values
- Write docstrings for all public functions and classes (Google style)

### Formatting Commands

```bash
# Format code
black src/ tests/ api/ app/

# Sort imports
isort src/ tests/ api/ app/

# Lint
flake8 src/ tests/ api/ app/
```

### Example Code Style

```python
"""Module docstring explaining the purpose."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .constants import COLS


def calculate_wage_gap(
    df: pd.DataFrame,
    wage_col: str = COLS.HOURLY_EARNINGS,
    gender_col: str = COLS.GENDER,
) -> Dict[str, float]:
    """
    Calculate the gender wage gap.

    Args:
        df: DataFrame containing wage data
        wage_col: Name of the wage column
        gender_col: Name of the gender column

    Returns:
        Dictionary containing gap statistics

    Raises:
        ValueError: If required columns are missing
    """
    if wage_col not in df.columns:
        raise ValueError(f"Column '{wage_col}' not found in DataFrame")
    
    # Implementation...
    return {"gap_pct": 12.5, "gap_dollars": 3.50}
```

##  Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_pipeline.py -v

# Run specific test
pytest tests/test_pipeline.py::TestDataPipeline::test_generate_synthetic_data -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test names that explain what is being tested
- Use pytest fixtures for common setup
- Aim for good coverage of edge cases

```python
import pytest
import pandas as pd

from src.analysis import PayEquityAnalyzer


class TestPayEquityAnalyzer:
    """Tests for PayEquityAnalyzer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'HRLYEARN': [20, 25, 18, 22, 30],
            'GENDER': [1, 1, 2, 2, 1],
        })

    def test_compute_raw_wage_gap(self, sample_data):
        """Test that raw wage gap is calculated correctly."""
        analyzer = PayEquityAnalyzer(sample_data)
        result = analyzer.compute_raw_wage_gap()
        
        assert 'raw_gap' in result
        assert 'mean_gap_pct' in result['raw_gap']
        assert isinstance(result['raw_gap']['mean_gap_pct'], float)
```

##  Pull Request Process

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes**
   - Write clear, concise commit messages
   - Keep commits focused and atomic
   - Update documentation as needed

3. **Test your changes**
   ```bash
   pytest tests/ -v
   black --check src/ tests/
   flake8 src/ tests/
   ```

4. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added for new functionality
- [ ] All tests pass
- [ ] Documentation updated if needed
- [ ] PR description explains the changes
- [ ] Linked to relevant issues (if any)

##  Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the issue
2. **Steps to reproduce**: Minimal steps to reproduce the problem
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**: Python version, OS, relevant package versions
6. **Error messages**: Full stack traces if applicable

##  Project Structure

```
equipay-canada/
├── src/                    # Core source code
│   ├── __init__.py
│   ├── constants.py        # Centralized constants
│   ├── data_pipeline.py    # Data loading/processing
│   ├── models.py           # ML models
│   ├── analysis.py         # Statistical analysis
│   ├── fairness.py         # Fairness evaluation
│   └── ...
├── api/                    # FastAPI application
├── app/                    # Streamlit dashboard
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks
├── data/                   # Data files
├── models/                 # Saved model artifacts
└── reports/                # Generated reports
```

##  Key Modules

| Module | Description |
|--------|-------------|
| `constants.py` | Centralized LFS codes and column mappings |
| `data_pipeline.py` | Data loading, cleaning, feature creation |
| `models.py` | Salary prediction models |
| `analysis.py` | Pay equity statistical analysis |
| `fairness.py` | Algorithmic fairness evaluation |

##  Questions?

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Be respectful and constructive in discussions

Thank you for contributing! 
