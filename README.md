# EquiPay Canada 🇨🇦

**Machine Learning-Powered Compensation Analysis & Pay Equity System**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

EquiPay Canada is a comprehensive machine learning system that analyzes the Canadian labour market to:

- **Predict salaries** based on job characteristics and demographics
- **Detect pay inequities** across gender, age, and other protected groups
- **Identify compensation drivers** that explain wage variations
- **Generate actionable insights** for HR professionals and policymakers

## 📊 Data Sources

This project uses **only two data sources**:

### 1. Statistics Canada Labour Force Survey (LFS) PUMF
- **Catalogue**: 71M0001X
- **Time Period**: 2010-2025 (16 years)
- **Scope**: ~100,000+ individual records per month
- **Variables**: Age, Gender, Education, Occupation, Industry, Province, Hourly Earnings, Employment Type, Union Status

### 2. Canadian Macroeconomic Data
Built-in macroeconomic indicators (see `src/macro_data.py`):
- **CPI**: Consumer Price Index (for real wage calculations)
- **GDP Growth**: Annual GDP growth rate
- **Unemployment**: National unemployment rate
- **Interest Rate**: Bank of Canada policy rate

**No other external data sources are required for this project.**

## 🏗️ Project Structure

```
equipay-canada/
├── README.md                 # Project overview
├── requirements.txt          # Python dependencies
├── config.yaml               # Configuration settings
├── data/
│   ├── raw/                  # Original LFS files
│   └── processed/            # Cleaned datasets
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Data exploration & quality
│   ├── 02_model_training.ipynb        # ML model training
│   ├── 03_pay_equity_analysis.ipynb   # Pay gap analysis
│   ├── 04_fairness_evaluation.ipynb   # Algorithmic fairness
│   ├── 05_econometric_analysis.ipynb  # Oaxaca-Blinder, quantile regression
│   ├── 06_time_series_analysis.ipynb  # Temporal trends 2010-2025
│   └── 07_advanced_statistics.ipynb   # Bootstrap, power analysis
├── src/
│   ├── __init__.py
│   ├── constants.py          # Centralized LFS codes & mappings
│   ├── macro_data.py         # Canadian macro data (2010-2025)
│   ├── data_pipeline.py      # LFS data loading & processing
│   ├── lfs_loader.py         # LFS PUMF file loader
│   ├── feature_engineering.py
│   ├── models.py             # ML models
│   ├── fairness.py           # Bias detection
│   ├── analysis.py           # Statistical analysis
│   ├── statistical_tests.py  # Advanced stats
│   ├── time_series.py        # Time series analysis
│   └── utils.py
├── app/
│   └── dashboard.py          # Streamlit dashboard
├── api/
│   ├── main.py               # FastAPI application
│   └── schemas.py            # Pydantic models
├── reports/
│   └── templates/            # Report templates
├── models/                   # Saved model artifacts
└── tests/                    # Unit tests
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/equipay-canada.git
cd equipay-canada

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Data

```bash
# Download LFS PUMF data from Statistics Canada
python scripts/download_lfs_data.py --all

# Or download specific years
python scripts/download_lfs_data.py --years 2020 2021 2022 2023

# Check available data
python scripts/download_lfs_data.py --summary

# Data source: Statistics Canada, Catalogue 71M0001X
# https://www150.statcan.gc.ca/n1/en/catalogue/71M0001X
```

See [DATA_SOURCES.md](DATA_SOURCES.md) for complete data source documentation.

### Run the Dashboard

```bash
streamlit run app/dashboard.py
```

### Start the API

```bash
uvicorn api.main:app --reload
```

## 📈 Features

### 1. Salary Prediction Model
- Ensemble model (XGBoost + LightGBM + CatBoost)
- R² > 0.70 on test data
- Confidence intervals for predictions

### 2. Pay Equity Analysis
- Raw and adjusted gender wage gap calculations
- Industry and occupation-level analysis
- Time series trend analysis (2010-2025)
- Statistical significance testing
- Oaxaca-Blinder decomposition
- Quantile regression (glass ceiling analysis)

### 3. Fairness-Aware ML
- Bias detection using Fairlearn
- Multiple fairness metrics (demographic parity, equalized odds)
- Bias mitigation techniques

### 4. Interactive Dashboard
- Salary calculator
- Pay gap explorer
- Trend analyzer
- Equity scorecard

### 5. REST API
- Salary prediction endpoint
- Pay equity check endpoint
- Batch prediction support

### 6. Automated Pipeline
- Monthly data refresh
- Model retraining
- Report generation

## 📊 Key Findings

| Metric | Value |
|--------|-------|
| Raw Gender Wage Gap | ~12-15% |
| Adjusted Gender Wage Gap | ~5-8% |
| Model R² Score | 0.72 |
| Top Predictor | Occupation |

## 🛠️ Technologies

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Fairness**: Fairlearn, SHAP
- **Visualization**: Plotly, Seaborn
- **Dashboard**: Streamlit
- **API**: FastAPI
- **Statistical Analysis**: scipy, statsmodels

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Statistics Canada for providing the Labour Force Survey data
- Canadian Pay Equity Act for inspiring this work

## 📧 Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/equipay-canada](https://github.com/yourusername/equipay-canada)
