"""
Utility Functions Module
Helper functions for the EquiPay Canada project
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
import yaml
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def setup_logging(level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    Configure logging for the project
    """
    log_level = getattr(logging, level.upper())
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_config(config_path: str = 'config.yaml') -> Dict:
    """
    Load configuration from YAML file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, config_path: str = 'config.yaml') -> None:
    """
    Save configuration to YAML file
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def format_currency(value: float, currency: str = 'CAD') -> str:
    """
    Format value as currency
    """
    if currency == 'CAD':
        return f"${value:,.2f}"
    return f"{value:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format value as percentage
    """
    return f"{value:.{decimals}f}%"


def calculate_confidence_interval(data: np.ndarray, 
                                   confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for mean
    """
    from scipy import stats
    
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return (mean - h, mean + h)


def bootstrap_confidence_interval(data: np.ndarray,
                                    statistic: callable = np.mean,
                                    n_bootstrap: int = 1000,
                                    confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval
    """
    n = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_stats, alpha * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha) * 100)
    
    return (lower, upper)


def describe_dataset(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive dataset description
    """
    description = {
        'shape': {
            'rows': df.shape[0],
            'columns': df.shape[1],
        },
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
    }
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        description['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    # Categorical columns summary
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        description['categorical_summary'] = {
            col: {
                'unique': df[col].nunique(),
                'top_values': df[col].value_counts().head(5).to_dict(),
            }
            for col in cat_cols
        }
    
    return description


def validate_data(df: pd.DataFrame, 
                   required_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame has required columns
    """
    missing = [col for col in required_columns if col not in df.columns]
    return len(missing) == 0, missing


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names
    """
    df = df.copy()
    df.columns = [
        col.strip().upper().replace(' ', '_').replace('-', '_')
        for col in df.columns
    ]
    return df


def create_output_directories() -> Dict[str, Path]:
    """
    Create standard output directories
    """
    dirs = {
        'data_raw': Path('data/raw'),
        'data_processed': Path('data/processed'),
        'models': Path('models'),
        'reports': Path('reports'),
        'figures': Path('reports/figures'),
    }
    
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def save_results(results: Dict, 
                  filepath: str,
                  format: str = 'json') -> None:
    """
    Save results to file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif format == 'yaml':
        with open(filepath, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_results(filepath: str, format: str = 'json') -> Dict:
    """
    Load results from file
    """
    if format == 'json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif format == 'yaml':
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unknown format: {format}")


def get_label_mapping(config: Dict) -> Dict[str, Dict]:
    """
    Get human-readable labels for coded variables from config
    """
    return config.get('labels', {})


def code_to_label(code: Union[int, str], 
                   variable: str,
                   labels: Dict[str, Dict]) -> str:
    """
    Convert coded value to human-readable label
    """
    if variable not in labels:
        return str(code)
    
    var_labels = labels[variable]
    return var_labels.get(str(code), var_labels.get(code, str(code)))


def generate_report_header(title: str, subtitle: str = "") -> str:
    """
    Generate formatted report header
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    header = f"""
{'=' * 70}
{title.upper().center(70)}
{subtitle.center(70) if subtitle else ''}
{'=' * 70}

Generated: {timestamp}
{'=' * 70}
"""
    return header


def create_data_dictionary(df: pd.DataFrame, 
                            descriptions: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Create data dictionary for DataFrame
    """
    descriptions = descriptions or {}
    
    dict_data = []
    for col in df.columns:
        dict_data.append({
            'column': col,
            'dtype': str(df[col].dtype),
            'non_null': df[col].notna().sum(),
            'null_pct': f"{df[col].isna().mean() * 100:.1f}%",
            'unique_values': df[col].nunique(),
            'description': descriptions.get(col, ''),
        })
    
    return pd.DataFrame(dict_data)


class Timer:
    """
    Context manager for timing code blocks
    """
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start = None
        self.end = None
        self.elapsed = None
    
    def __enter__(self):
        self.start = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.end = datetime.now()
        self.elapsed = (self.end - self.start).total_seconds()
        logger.info(f"{self.name} completed in {self.elapsed:.2f} seconds")


class DataValidator:
    """
    Validate data quality and constraints
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.issues = []
    
    def check_missing(self, threshold: float = 0.05) -> 'DataValidator':
        """Check for columns with missing values above threshold"""
        for col in self.df.columns:
            missing_pct = self.df[col].isna().mean()
            if missing_pct > threshold:
                self.issues.append({
                    'type': 'missing_values',
                    'column': col,
                    'missing_pct': missing_pct,
                    'severity': 'warning' if missing_pct < 0.2 else 'error',
                })
        return self
    
    def check_range(self, column: str, 
                     min_val: Optional[float] = None,
                     max_val: Optional[float] = None) -> 'DataValidator':
        """Check if values are within expected range"""
        if column not in self.df.columns:
            return self
        
        values = self.df[column].dropna()
        
        if min_val is not None and (values < min_val).any():
            count = (values < min_val).sum()
            self.issues.append({
                'type': 'out_of_range',
                'column': column,
                'issue': f'{count} values below {min_val}',
                'severity': 'error',
            })
        
        if max_val is not None and (values > max_val).any():
            count = (values > max_val).sum()
            self.issues.append({
                'type': 'out_of_range',
                'column': column,
                'issue': f'{count} values above {max_val}',
                'severity': 'error',
            })
        
        return self
    
    def check_unique(self, column: str, 
                      expected_values: Optional[List] = None) -> 'DataValidator':
        """Check for unexpected categorical values"""
        if column not in self.df.columns:
            return self
        
        if expected_values:
            actual = set(self.df[column].dropna().unique())
            unexpected = actual - set(expected_values)
            if unexpected:
                self.issues.append({
                    'type': 'unexpected_values',
                    'column': column,
                    'unexpected': list(unexpected),
                    'severity': 'warning',
                })
        
        return self
    
    def get_report(self) -> Dict:
        """Get validation report"""
        return {
            'total_issues': len(self.issues),
            'errors': [i for i in self.issues if i['severity'] == 'error'],
            'warnings': [i for i in self.issues if i['severity'] == 'warning'],
            'passed': len(self.issues) == 0,
        }
    
    def raise_on_error(self) -> 'DataValidator':
        """Raise exception if any errors found"""
        errors = [i for i in self.issues if i['severity'] == 'error']
        if errors:
            raise ValueError(f"Data validation failed: {errors}")
        return self


# =============================================================================
# Classification Change Detection
# =============================================================================

# Known classification revision years
CLASSIFICATION_REVISIONS = {
    'NOC': {
        2011: 'NOC 2011 introduced (from NOC-S 2006)',
        2016: 'NOC 2016 v1.0 (minor updates)',
        2022: 'NOC 2021 major restructuring (TEER categories)',
    },
    'NAICS': {
        2012: 'NAICS 2012 (information sector restructured)',
        2017: 'NAICS 2017 (cannabis production, professional services)',
        2022: 'NAICS 2022 (digital industries expanded)',
    }
}


def detect_classification_breaks(
    df: pd.DataFrame,
    variable: str,
    classification_col: str = 'NOC_10',
    weight_col: str = 'FINALWT',
    test_years: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Detect potential structural breaks caused by classification revisions.
    
    Parameters
    ----------
    df : DataFrame
        LFS data with SURVYEAR, classification column, and analysis variable
    variable : str
        Variable to analyze (e.g., 'HRLYEARN', computed wage gap)
    classification_col : str
        Classification column ('NOC_10', 'NOC_43', 'NAICS_21')
    weight_col : str
        Survey weight column
    test_years : List[int], optional
        Years to test for breaks. Default: NOC/NAICS revision years
        
    Returns
    -------
    Dict with break detection results
    """
    from scipy import stats
    
    if test_years is None:
        # Default to classification revision years
        test_years = [2012, 2017, 2022]
    
    results = {
        'classification': classification_col,
        'variable': variable,
        'test_years': test_years,
        'breaks': [],
        'recommendations': []
    }
    
    for year in test_years:
        if year not in df['SURVYEAR'].values or (year - 1) not in df['SURVYEAR'].values:
            continue
            
        # Compare distributions before/after break year
        before = df[df['SURVYEAR'] == year - 1]
        after = df[df['SURVYEAR'] == year]
        
        break_result = {
            'year': year,
            'tests': {}
        }
        
        # 1. Chi-square test for distribution of classification codes
        before_counts = before.groupby(classification_col)[weight_col].sum()
        after_counts = after.groupby(classification_col)[weight_col].sum()
        
        # Align indices
        all_codes = sorted(set(before_counts.index) | set(after_counts.index))
        before_freq = np.array([before_counts.get(c, 0) for c in all_codes])
        after_freq = np.array([after_counts.get(c, 0) for c in all_codes])
        
        # Normalize to compare proportions
        before_prop = before_freq / before_freq.sum()
        after_prop = after_freq / after_freq.sum()
        
        # Two-sample chi-square test
        expected = (before_freq.sum() + after_freq.sum()) / 2 * (before_prop + after_prop) / 2
        expected = np.maximum(expected, 1)  # Avoid division by zero
        
        chi2_stat = np.sum((before_freq - expected)**2 / expected + 
                          (after_freq - expected)**2 / expected)
        chi2_pval = 1 - stats.chi2.cdf(chi2_stat, len(all_codes) - 1)
        
        break_result['tests']['chi2_distribution'] = {
            'statistic': chi2_stat,
            'p_value': chi2_pval,
            'significant': chi2_pval < 0.05,
            'interpretation': 'Distribution shift detected' if chi2_pval < 0.05 else 'No significant shift'
        }
        
        # 2. Mean comparison by classification code
        mean_changes = []
        for code in all_codes:
            b = before[before[classification_col] == code][variable]
            a = after[after[classification_col] == code][variable]
            
            if len(b) >= 30 and len(a) >= 30:
                t_stat, t_pval = stats.ttest_ind(b, a)
                mean_change = a.mean() - b.mean()
                mean_changes.append({
                    'code': code,
                    'mean_change': mean_change,
                    'pct_change': mean_change / b.mean() * 100 if b.mean() != 0 else np.nan,
                    't_statistic': t_stat,
                    'p_value': t_pval,
                    'significant': t_pval < 0.05
                })
        
        break_result['tests']['mean_changes'] = mean_changes
        
        # Count significant changes
        n_significant = sum(1 for m in mean_changes if m['significant'])
        break_result['n_significant_changes'] = n_significant
        break_result['pct_significant'] = n_significant / len(mean_changes) * 100 if mean_changes else 0
        
        # Flag potential break
        break_result['potential_break'] = (
            chi2_pval < 0.05 or 
            n_significant > len(mean_changes) * 0.3  # >30% of codes show significant change
        )
        
        results['breaks'].append(break_result)
    
    # Generate recommendations
    for br in results['breaks']:
        if br['potential_break']:
            results['recommendations'].append(
                f"⚠️ {br['year']}: Potential classification break detected. "
                f"Consider including structural break dummy or splitting analysis."
            )
    
    return results


def create_structural_break_dummies(
    df: pd.DataFrame,
    break_years: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Create dummy variables for known classification break years.
    
    Parameters
    ----------
    df : DataFrame
        Must contain 'SURVYEAR' column
    break_years : List[int], optional
        Years for break dummies. Default: [2012, 2017, 2022]
        
    Returns
    -------
    DataFrame with added dummy columns
    """
    if break_years is None:
        break_years = [2012, 2017, 2022]
    
    df = df.copy()
    
    for year in break_years:
        df[f'post_{year}'] = (df['SURVYEAR'] >= year).astype(int)
    
    return df
