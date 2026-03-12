"""
EquiPay Canada - Interactive Dashboard
Streamlit application for pay equity analysis visualization

This dashboard reflects the full project analysis including:
- Data exploration and quality metrics
- Gender wage gap analysis (raw and adjusted)
- Oaxaca-Blinder decomposition
- Fairness metrics and bias detection
- Econometric analysis with macro controls
- Time series trends and forecasting
- Geographic/provincial analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import yaml
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_store import EquiPayDataStore
from src.data_pipeline import LFSDataPipeline
from src.feature_engineering import FeatureEngineer
from src.analysis import PayEquityAnalyzer
from src.fairness import FairnessAnalyzer
from src.time_series import WageGapTimeSeriesAnalyzer
from src.macro_data import MACRO_DATA, get_macro_dataframe, ECONOMIC_PERIODS, BASE_YEAR
from src.constants import (
    COLS, GENDER_CODES, EDUCATION_CODES, NOC_10_CODES, PROVINCE_CODES,
    DATA_SCOPE_START, DATA_SCOPE_END,
    EQUITY_DIMENSIONS, get_equity_dimension, EquityDimension
)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Calculate population-weighted mean for survey data.
    
    LFS data requires FINALWT weights for population-level inference.
    """
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    """Calculate weighted median (approximate via interpolation)."""
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    v = values[mask].values
    w = weights[mask].values
    sorted_idx = np.argsort(v)
    v_sorted = v[sorted_idx]
    w_sorted = w[sorted_idx]
    cumsum = np.cumsum(w_sorted)
    mid = cumsum[-1] / 2
    idx = np.searchsorted(cumsum, mid)
    return v_sorted[min(idx, len(v_sorted) - 1)]


def load_cached_aggregates():
    """Load precomputed Parquet artifacts from reports/cache/ for fast dashboard rendering."""
    cache_dir = Path('reports') / 'cache'
    results = {}
    if (cache_dir / 'annual_gap.parquet').exists():
        results['annual_gap'] = pd.read_parquet(cache_dir / 'annual_gap.parquet')
    if (cache_dir / 'provincial_means.parquet').exists():
        results['provincial_means'] = pd.read_parquet(cache_dir / 'provincial_means.parquet')
    return results


# Page configuration
st.set_page_config(
    page_title="EquiPay Canada - Pay Equity Dashboard",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_population_stats():
    """Query population-level statistics from full dataset via DuckDB (not the sample)."""
    try:
        store = EquiPayDataStore(memory_limit_mb=3000)
        row = store.sql("""
            SELECT
                COUNT(*) AS total_records,
                SUM(FINALWT) AS total_weight,
                AVG(FINALWT) AS avg_weight,
                MIN(SURVYEAR) AS min_year,
                MAX(SURVYEAR) AS max_year,
                COUNT(DISTINCT SURVYEAR) AS n_years
            FROM lfs_enriched
            WHERE HRLYEARN IS NOT NULL AND FINALWT > 0
        """)
        return row.iloc[0].to_dict()
    except Exception:
        return None


@st.cache_data
def load_data():
    """Load and cache data using DuckDB data store (memory-efficient)"""
    try:
        # Use DuckDB data store for memory-efficient data access
        store = EquiPayDataStore(memory_limit_mb=3000)
        
        # Load a sample for dashboard (full data is 19M+ rows)
        # For interactive dashboards, we use a representative sample
        # Include FINALWT for proper population-weighted statistics
        df = store.sql("""
            SELECT * FROM lfs_enriched 
            WHERE HRLYEARN IS NOT NULL AND FINALWT > 0
            USING SAMPLE 500000 ROWS
        """)
        
        # Ensure SEX alias exists for backward compatibility
        if 'GENDER' in df.columns and 'SEX' not in df.columns:
            df['SEX'] = df['GENDER']
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


@st.cache_resource
def load_config():
    """Load configuration"""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def get_labels():
    """Get human-readable labels for coded values"""
    return {
        'GENDER': {1: 'Male', 2: 'Female'},
        'SEX': {1: 'Male', 2: 'Female'},  # Legacy alias
        'EDUC': {
            0: 'Less than high school',
            1: 'High school graduate',
            2: 'Some college',
            3: 'College diploma',
            4: 'University certificate',
            5: "Bachelor's degree",
            6: 'Graduate degree',
        },
        'NOC_10': {
            0: 'Management',
            1: 'Business/Finance',
            2: 'Sciences',
            3: 'Health',
            4: 'Education/Law/Social',
            5: 'Art/Culture/Recreation',
            6: 'Sales/Service',
            7: 'Trades/Transport',
            8: 'Resources/Agriculture',
            9: 'Manufacturing',
        },
        'PROV': {
            10: 'Newfoundland and Labrador',
            11: 'Prince Edward Island',
            12: 'Nova Scotia',
            13: 'New Brunswick',
            24: 'Quebec',
            35: 'Ontario',
            46: 'Manitoba',
            47: 'Saskatchewan',
            48: 'Alberta',
            59: 'British Columbia',
        },
        'FTPTMAIN': {
            1: 'Full-time',
            2: 'Part-time',
        },
        'UNION': {
            1: 'Union member',
            2: 'Covered by union',
            3: 'Not unionized',
        },
    }


def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<p class="main-header"><img src="https://flagcdn.com/24x18/ca.png" alt="Canada" style="vertical-align: middle; margin-right: 8px;"/> EquiPay Canada</p>', unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #666;'>Pay Equity Analysis Dashboard</h3>", 
                unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading data...'):
        df = load_data()
        labels = get_labels()
    
    # Sidebar filters
    st.sidebar.header("Filters", anchor=False)
    st.sidebar.caption(":material/filter_alt: Configure analysis parameters")
    
    # === Equity Dimension Selector (NEW) ===
    st.sidebar.subheader(":material/analytics: Primary Equity Dimension")
    equity_options = {
        'Gender': 'gender',
        'Immigration Status': 'immigration',
        'Age (Youth vs Prime)': 'age_young',
        'Age (Older vs Peak)': 'age_older',
        'Union Status': 'union',
        'Employment Type': 'employment_type',
        'Full-time/Part-time': 'fulltime_parttime'
    }
    selected_equity_name = st.sidebar.selectbox(
        "Select equity dimension to analyze",
        options=list(equity_options.keys()),
        index=0,
        help="Choose which protected attribute to focus on for gap analysis"
    )
    selected_equity_key = equity_options[selected_equity_name]
    selected_equity_dim = get_equity_dimension(selected_equity_key)
    
    st.sidebar.markdown(f"*{selected_equity_dim.description}*")
    st.sidebar.markdown(f"**Reference:** {selected_equity_dim.reference_label}")
    st.sidebar.markdown(f"**Comparison:** {selected_equity_dim.comparison_label}")
    st.sidebar.markdown("---")
    
    # Province filter
    if 'PROV' in df.columns:
        provinces = df['PROV'].unique()
        province_names = [labels['PROV'].get(p, str(p)) for p in sorted(provinces)]
        selected_provinces = st.sidebar.multiselect(
            "Province",
            options=province_names,
            default=province_names
        )
        # Map back to codes
        name_to_code = {v: k for k, v in labels['PROV'].items()}
        prov_codes = [name_to_code.get(p, p) for p in selected_provinces]
        if prov_codes:
            df = df[df['PROV'].isin(prov_codes)]
    
    # Education filter
    if 'EDUC' in df.columns:
        st.sidebar.subheader("Education Level")
        educ_levels = sorted(df['EDUC'].unique())
        educ_names = [labels['EDUC'].get(e, str(e)) for e in educ_levels]
        selected_educ = st.sidebar.multiselect(
            "Select education levels",
            options=educ_names,
            default=educ_names
        )
        name_to_code = {v: k for k, v in labels['EDUC'].items()}
        educ_codes = [name_to_code.get(e, e) for e in selected_educ]
        if educ_codes:
            df = df[df['EDUC'].isin(educ_codes)]
    
    # Wage range filter
    if 'HRLYEARN' in df.columns:
        st.sidebar.subheader("Hourly Wage Range")
        min_wage = float(df['HRLYEARN'].min())
        max_wage = float(df['HRLYEARN'].max())
        wage_range = st.sidebar.slider(
            "Select wage range ($/hr)",
            min_value=min_wage,
            max_value=max_wage,
            value=(min_wage, max_wage),
            step=1.0
        )
        df = df[(df['HRLYEARN'] >= wage_range[0]) & (df['HRLYEARN'] <= wage_range[1])]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Filtered Sample Size:** {len(df):,}")
    st.sidebar.markdown(f"**Data Period:** {DATA_SCOPE_START}-{DATA_SCOPE_END}")
    
    # Main content tabs - reflecting the full project
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        ":material/trending_up: Overview", 
        ":material/balance: Wage Gap Analysis", 
        ":material/biotech: Decomposition",
        ":material/timeline: Time Series",
        ":material/target: Fairness",
        ":material/query_stats: Econometrics",
        ":material/map: Geographic",
        ":material/payments: Salary Predictor",
        ":material/checklist: Recommendations"
    ])
    
    # Tab 1: Overview
    with tab1:
        display_overview(df, labels, selected_equity_dim)
    
    # Tab 2: Wage Gap Analysis (generalized from Gender Gap)
    with tab2:
        display_wage_gap_analysis(df, labels, selected_equity_dim)
    
    # Tab 3: Oaxaca-Blinder Decomposition
    with tab3:
        display_decomposition(df, labels, selected_equity_dim)
    
    # Tab 4: Time Series Analysis
    with tab4:
        display_time_series(selected_equity_dim)
    
    # Tab 5: Fairness Metrics
    with tab5:
        display_fairness_metrics(df, labels)
    
    # Tab 6: Econometric Analysis
    with tab6:
        display_econometrics(df, labels)
    
    # Tab 7: Geographic Analysis
    with tab7:
        display_geographic(df, labels)
    
    # Tab 8: Salary Predictor
    with tab8:
        display_model_predictions(df, labels)
    
    # Tab 9: Recommendations
    with tab9:
        display_recommendations(df, labels)


def display_overview(df: pd.DataFrame, labels: dict, equity_dim: EquityDimension = None):
    """Display overview metrics with proper survey weighting"""
    st.header(":material/trending_up: Overview Dashboard")
    
    # Default to gender if no dimension specified
    if equity_dim is None:
        equity_dim = get_equity_dimension('gender')
    
    # Check for weights
    has_weights = 'FINALWT' in df.columns and df['FINALWT'].notna().any()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if has_weights:
            avg_wage = weighted_mean(df['HRLYEARN'], df['FINALWT'])
        else:
            avg_wage = df['HRLYEARN'].mean()
        st.metric(
            label="Average Hourly Wage",
            value=f"${avg_wage:.2f}",
            delta=None,
            help="Population-weighted mean" if has_weights else "Sample mean"
        )
    
    with col2:
        # Use selected equity dimension for gap calculation
        group_col = equity_dim.column
        if group_col in df.columns:
            ref_df = df[df[group_col] == equity_dim.reference_value]
            comp_df = df[df[group_col] == equity_dim.comparison_value]
            if has_weights and len(ref_df) > 0 and len(comp_df) > 0:
                ref_wage = weighted_mean(ref_df['HRLYEARN'], ref_df['FINALWT'])
                comp_wage = weighted_mean(comp_df['HRLYEARN'], comp_df['FINALWT'])
            else:
                ref_wage = ref_df['HRLYEARN'].mean() if len(ref_df) > 0 else 0
                comp_wage = comp_df['HRLYEARN'].mean() if len(comp_df) > 0 else 0
            if ref_wage > 0:
                gap_pct = ((ref_wage - comp_wage) / ref_wage) * 100
                st.metric(
                    label=f"{equity_dim.description} Gap",
                    value=f"{gap_pct:.1f}%",
                    delta=None,
                    delta_color="inverse"
                )
            else:
                st.metric(label=f"{equity_dim.description} Gap", value="N/A")
    
    with col3:
        if has_weights:
            pop_stats = load_population_stats()
            if pop_stats:
                # Average annual population estimate: total weight / (n_years * 12 months)
                n_years = pop_stats['n_years']
                avg_annual_pop = pop_stats['total_weight'] / n_years
                st.metric(
                    label="Avg Annual Labour Force",
                    value=f"{avg_annual_pop:,.0f}",
                    delta=None,
                    help=(
                        f"Average annual working population estimate ({int(pop_stats['min_year'])}–{int(pop_stats['max_year'])}). "
                        f"Derived from {pop_stats['total_records']:,.0f} total survey records × FINALWT survey weights. "
                        f"Each monthly LFS record carries a weight reflecting how many Canadians it represents."
                    )
                )
            else:
                pop_estimate = df['FINALWT'].sum()
                st.metric(
                    label="Population Estimate (sample)",
                    value=f"{pop_estimate:,.0f}",
                    delta=None,
                    help=f"Approximate – based on {len(df):,} sampled rows, not full dataset"
                )
        else:
            st.metric(
                label="Sample Size",
                value=f"{len(df):,}",
                delta=None
            )
    
    with col4:
        if has_weights:
            median_wage = weighted_median(df['HRLYEARN'], df['FINALWT'])
        else:
            median_wage = df['HRLYEARN'].median()
        st.metric(
            label="Median Hourly Wage",
            value=f"${median_wage:.2f}",
            delta=None,
            help="Population-weighted median" if has_weights else "Sample median"
        )
    
    # Wage distribution
    st.subheader("Wage Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df, x='HRLYEARN',
            nbins=50,
            title="Distribution of Hourly Wages",
            labels={'HRLYEARN': 'Hourly Wage ($)', 'count': 'Count'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'GENDER' in df.columns:
            df_plot = df.copy()
            df_plot['Gender'] = df_plot['GENDER'].map(labels['GENDER'])
            
            fig = px.histogram(
                df_plot, x='HRLYEARN',
                color='Gender',
                nbins=50,
                barmode='overlay',
                opacity=0.7,
                title="Wage Distribution by Gender",
                labels={'HRLYEARN': 'Hourly Wage ($)', 'count': 'Count'},
                color_discrete_map={'Male': '#1f77b4', 'Female': '#e377c2'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Wage by province map (simplified bar chart)
    if 'PROV' in df.columns:
        st.subheader("Average Wages by Province")
        
        prov_wages = df.groupby('PROV')['HRLYEARN'].mean().reset_index()
        prov_wages['Province'] = prov_wages['PROV'].map(labels['PROV'])
        prov_wages = prov_wages.sort_values('HRLYEARN', ascending=True)
        
        fig = px.bar(
            prov_wages,
            y='Province',
            x='HRLYEARN',
            orientation='h',
            title="Average Hourly Wage by Province",
            labels={'HRLYEARN': 'Average Hourly Wage ($)', 'Province': ''},
            color='HRLYEARN',
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Data Validation / Truthfulness
    st.subheader(":material/verified: Data Validation")
    with st.expander("View data quality and validation checks", expanded=False):
        checks = []

        # 1. Wage range - check for extreme outliers
        # Real data: p01=$9.50, median=$24, p99=$82, max=$271
        # Minimum wage in Canada is ~$15-17/hr, so <$10 is suspicious
        wage_min, wage_max = df['HRLYEARN'].min(), df['HRLYEARN'].max()
        wage_plausible = 10 < wage_min and wage_max < 300
        checks.append({
            'Check': 'Wage range plausible',
            'Result': 'PASS' if wage_plausible else 'WARN',
            'Detail': f"${wage_min:.2f} – ${wage_max:.2f}/hr (expect $10-300)"
        })

        # 2. Gender balance - check for severe imbalance
        if 'GENDER' in df.columns:
            male_pct = (df['GENDER'] == 1).mean() * 100
            balanced = 40 < male_pct < 60
            checks.append({
                'Check': 'Gender distribution balanced',
                'Result': 'PASS' if balanced else 'WARN',
                'Detail': f"Male {male_pct:.1f}% / Female {100 - male_pct:.1f}% (expect ~50/50)"
            })

        # 3. Province coverage - LFS PUMF covers all 10 provinces
        if 'PROV' in df.columns:
            n_prov = df['PROV'].nunique()
            checks.append({
                'Check': 'All 10 provinces present',
                'Result': 'PASS' if n_prov == 10 else 'WARN',
                'Detail': f"{n_prov} provinces found (LFS covers 10 provinces)"
            })

        # 4. Year coverage - check for missing years
        if 'SURVYEAR' in df.columns:
            years = sorted(df['SURVYEAR'].unique())
            expected = list(range(DATA_SCOPE_START, DATA_SCOPE_END + 1))
            missing = [y for y in expected if y not in years]
            checks.append({
                'Check': f'Year coverage ({DATA_SCOPE_START}–{DATA_SCOPE_END})',
                'Result': 'PASS' if not missing else 'WARN',
                'Detail': f"{len(years)} years present" + (f"; missing {missing}" if missing else "")
            })

        # 5. Weight validity - critical for survey data
        if has_weights:
            neg_wt = (df['FINALWT'] <= 0).sum()
            checks.append({
                'Check': 'Survey weights positive',
                'Result': 'PASS' if neg_wt == 0 else 'FAIL',
                'Detail': f"{neg_wt} invalid weights" if neg_wt else "All weights > 0"
            })

        # 6. Missing data - check demographic variables (wage can be legitimately missing)
        # Note: HRLYEARN is missing for ~50% (unemployed), so we check other key vars
        demo_cols = [c for c in ['GENDER', 'PROV', 'EDUC'] if c in df.columns]
        if demo_cols:
            missing_pct = df[demo_cols].isnull().mean() * 100
            worst = missing_pct.max()
            checks.append({
                'Check': 'Demographic data completeness',
                'Result': 'PASS' if worst < 1 else 'WARN',
                'Detail': f"Max missing: {worst:.1f}% ({missing_pct.idxmax()})" if worst > 0 else "No missing demographics"
            })

        # 7. Median wage vs StatsCan benchmark
        # Real data: median=$24/hr, but varies 2010-2025 with inflation
        # Recent years (2023-2025): weighted avg $31-38/hr
        median_w = df['HRLYEARN'].median()
        median_ok = 21 < median_w < 35
        checks.append({
            'Check': 'Median wage realistic',
            'Result': 'PASS' if median_ok else 'WARN',
            'Detail': f"Median ${median_w:.2f}/hr (expect $21-35/hr for 2010-2025)"
        })

        # 8. Gender gap vs StatsCan benchmark
        # Real data (2020-2025): 12.08-13.30% weighted gap
        # Allow 8-18% range to catch major deviations
        if 'GENDER' in df.columns:
            if has_weights:
                m_w = weighted_mean(df[df['GENDER'] == 1]['HRLYEARN'], df[df['GENDER'] == 1]['FINALWT'])
                f_w = weighted_mean(df[df['GENDER'] == 2]['HRLYEARN'], df[df['GENDER'] == 2]['FINALWT'])
            else:
                m_w = df[df['GENDER'] == 1]['HRLYEARN'].mean()
                f_w = df[df['GENDER'] == 2]['HRLYEARN'].mean()
            raw_gap = ((m_w - f_w) / m_w) * 100 if m_w > 0 else 0
            gap_ok = 8 < raw_gap < 18
            checks.append({
                'Check': 'Gender wage gap realistic',
                'Result': 'PASS' if gap_ok else 'WARN',
                'Detail': f"Gap {raw_gap:.1f}% (StatsCan 2020-25: ~12-13%)"
            })

        check_df = pd.DataFrame(checks)
        pass_count = (check_df['Result'] == 'PASS').sum()
        st.markdown(f"**{pass_count}/{len(checks)}** checks passed")

        def color_result(val):
            if val == 'PASS':
                return 'background-color: #c6efce; color: #006100'
            elif val == 'WARN':
                return 'background-color: #ffeb9c; color: #9c5700'
            return 'background-color: #ffc7ce; color: #9c0006'

        st.dataframe(
            check_df.style.map(color_result, subset=['Result']),
            use_container_width=True, hide_index=True
        )


def display_wage_gap_analysis(df: pd.DataFrame, labels: dict, equity_dim: EquityDimension = None):
    """Display generalized wage gap analysis for any equity dimension"""
    
    # Default to gender if no dimension specified
    if equity_dim is None:
        equity_dim = get_equity_dimension('gender')
    
    st.header(f":material/balance: {equity_dim.description} Analysis")
    
    group_col = equity_dim.column
    
    if group_col not in df.columns:
        st.warning(f"{group_col} data not available")
        return
    
    # Check for weights
    has_weights = 'FINALWT' in df.columns and df['FINALWT'].notna().any()
    
    # Calculate wage gap using appropriate weighting
    ref_df = df[df[group_col] == equity_dim.reference_value]
    comp_df = df[df[group_col] == equity_dim.comparison_value]
    
    if len(ref_df) == 0 or len(comp_df) == 0:
        st.warning(f"Insufficient data for {equity_dim.description} analysis")
        return
    
    if has_weights:
        ref_mean = weighted_mean(ref_df['HRLYEARN'], ref_df['FINALWT'])
        comp_mean = weighted_mean(comp_df['HRLYEARN'], comp_df['FINALWT'])
    else:
        ref_mean = ref_df['HRLYEARN'].mean()
        comp_mean = comp_df['HRLYEARN'].mean()
    
    gap = ref_mean - comp_mean
    gap_pct = (gap / ref_mean) * 100 if ref_mean > 0 else 0
    ratio = comp_mean / ref_mean if ref_mean > 0 else 0
    
    # Display key findings
    weight_note = " (population-weighted)" if has_weights else ""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #1f77b4;">${comp_mean:.2f}</h2>
            <p>{equity_dim.comparison_label} Average Wage{weight_note}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #1f77b4;">${ref_mean:.2f}</h2>
            <p>{equity_dim.reference_label} Average Wage{weight_note}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        color = '#d62728' if gap_pct > 0 else '#28a745'
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: {color};">{gap_pct:.1f}%</h2>
            <p>{equity_dim.description}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Insight box
    if gap > 0:
        st.markdown(f"""
        <div class="insight-box">
            <strong>Key Insight:</strong> {equity_dim.comparison_label} workers earn <strong>${ratio:.2f}</strong> 
            for every $1.00 {equity_dim.reference_label} workers earn. 
            This represents a wage gap of <strong>${gap:.2f}/hour</strong> or approximately 
            <strong>${gap * 2000:.0f}</strong> less per year (assuming 2000 work hours).
            <br><br><em>Legal basis: {equity_dim.legal_basis}</em>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-box">
            <strong>Key Insight:</strong> {equity_dim.comparison_label} workers actually earn 
            <strong>${abs(gap):.2f}/hour more</strong> than {equity_dim.reference_label} workers in this sample.
            <br><br><em>Legal basis: {equity_dim.legal_basis}</em>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot comparison
        df_plot = df[df[group_col].isin([equity_dim.reference_value, equity_dim.comparison_value])].copy()
        group_labels = {
            equity_dim.reference_value: equity_dim.reference_label,
            equity_dim.comparison_value: equity_dim.comparison_label
        }
        df_plot['Group'] = df_plot[group_col].map(group_labels)
        
        fig = px.box(
            df_plot,
            x='Group',
            y='HRLYEARN',
            color='Group',
            title=f"Wage Distribution: {equity_dim.description}",
            labels={'HRLYEARN': 'Hourly Wage ($)'},
            color_discrete_map={
                equity_dim.reference_label: '#1f77b4', 
                equity_dim.comparison_label: '#e377c2'
            }
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bar comparison
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name=equity_dim.reference_label,
            x=['Average Wage'],
            y=[ref_mean],
            marker_color='#1f77b4',
            text=[f'${ref_mean:.2f}'],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name=equity_dim.comparison_label,
            x=['Average Wage'],
            y=[comp_mean],
            marker_color='#e377c2',
            text=[f'${comp_mean:.2f}'],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f'Average Hourly Wage: {equity_dim.reference_label} vs {equity_dim.comparison_label}',
            barmode='group',
            yaxis_title='Hourly Wage ($)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Gap by occupation
    if 'NOC_10' in df.columns:
        st.subheader(f"{equity_dim.description} by Occupation")
        
        gap_by_occ = []
        for occ in df['NOC_10'].unique():
            occ_df = df[df['NOC_10'] == occ]
            ref_occ = occ_df[occ_df[group_col] == equity_dim.reference_value]['HRLYEARN'].mean()
            comp_occ = occ_df[occ_df[group_col] == equity_dim.comparison_value]['HRLYEARN'].mean()
            if pd.notna(ref_occ) and pd.notna(comp_occ) and ref_occ > 0:
                gap_by_occ.append({
                    'Occupation': labels['NOC_10'].get(occ, str(occ)),
                    equity_dim.reference_label: ref_occ,
                    equity_dim.comparison_label: comp_occ,
                    'Gap %': ((ref_occ - comp_occ) / ref_occ) * 100
                })
        
        if gap_by_occ:
            gap_df = pd.DataFrame(gap_by_occ).sort_values('Gap %', ascending=False)
            
            fig = px.bar(
                gap_df,
                y='Occupation',
                x='Gap %',
                orientation='h',
                title=f'{equity_dim.description} by Occupation Category',
                labels={'Gap %': 'Wage Gap (%)'},
                color='Gap %',
                color_continuous_scale=['#28a745', '#ffc107', '#dc3545'],
                range_color=[min(0, gap_df['Gap %'].min()), max(gap_df['Gap %'].max(), 0)]
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def display_gender_analysis(df: pd.DataFrame, labels: dict):
    """Display gender wage gap analysis with proper survey weighting"""
    st.header(":material/balance: Gender Wage Gap Analysis")
    
    if 'GENDER' not in df.columns:
        st.warning("Gender data not available")
        return
    
    # Check for weights
    has_weights = 'FINALWT' in df.columns and df['FINALWT'].notna().any()
    
    # Calculate wage gap using appropriate weighting
    male_df = df[df['GENDER'] == 1]
    female_df = df[df['GENDER'] == 2]
    
    if has_weights:
        male_mean = weighted_mean(male_df['HRLYEARN'], male_df['FINALWT'])
        female_mean = weighted_mean(female_df['HRLYEARN'], female_df['FINALWT'])
    else:
        male_mean = male_df['HRLYEARN'].mean()
        female_mean = female_df['HRLYEARN'].mean()
    
    gap = male_mean - female_mean
    gap_pct = (gap / male_mean) * 100
    ratio = female_mean / male_mean
    
    # Display key findings
    weight_note = " (population-weighted)" if has_weights else ""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #1f77b4;">${female_mean:.2f}</h2>
            <p>Women's Average Hourly Wage{weight_note}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #1f77b4;">${male_mean:.2f}</h2>
            <p>Men's Average Hourly Wage{weight_note}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #d62728;">{gap_pct:.1f}%</h2>
            <p>Gender Wage Gap</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Insight box
    st.markdown(f"""
    <div class="insight-box">
        <strong>Key Insight:</strong> Women earn <strong>${ratio:.2f}</strong> for every $1.00 men earn. 
        This represents a wage gap of <strong>${gap:.2f}/hour</strong> or approximately 
        <strong>${gap * 2000:.0f}</strong> less per year (assuming 2000 work hours).
    </div>
    """, unsafe_allow_html=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot comparison
        df_plot = df.copy()
        df_plot['Gender'] = df_plot['GENDER'].map(labels['GENDER'])
        
        fig = px.box(
            df_plot,
            x='Gender',
            y='HRLYEARN',
            color='Gender',
            title="Wage Distribution by Gender",
            labels={'HRLYEARN': 'Hourly Wage ($)'},
            color_discrete_map={'Male': '#1f77b4', 'Female': '#e377c2'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Wage gap visualization
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Male',
            x=['Average Wage'],
            y=[male_mean],
            marker_color='#1f77b4',
            text=[f'${male_mean:.2f}'],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Female',
            x=['Average Wage'],
            y=[female_mean],
            marker_color='#e377c2',
            text=[f'${female_mean:.2f}'],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Average Hourly Wage Comparison',
            barmode='group',
            yaxis_title='Hourly Wage ($)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Gap by occupation
    if 'NOC_10' in df.columns:
        st.subheader("Gender Wage Gap by Occupation")
        
        gap_by_occ = []
        for occ in df['NOC_10'].unique():
            occ_df = df[df['NOC_10'] == occ]
            m = occ_df[occ_df['GENDER'] == 1]['HRLYEARN'].mean()
            f = occ_df[occ_df['GENDER'] == 2]['HRLYEARN'].mean()
            if pd.notna(m) and pd.notna(f) and m > 0:
                gap_by_occ.append({
                    'Occupation': labels['NOC_10'].get(occ, str(occ)),
                    'Male': m,
                    'Female': f,
                    'Gap %': ((m - f) / m) * 100
                })
        
        gap_df = pd.DataFrame(gap_by_occ).sort_values('Gap %', ascending=False)
        
        fig = px.bar(
            gap_df,
            y='Occupation',
            x='Gap %',
            orientation='h',
            title='Gender Wage Gap by Occupation Category',
            labels={'Gap %': 'Wage Gap (%)'},
            color='Gap %',
            color_continuous_scale=['#28a745', '#ffc107', '#dc3545'],
            range_color=[0, max(gap_df['Gap %'])]
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def display_detailed_breakdowns(df: pd.DataFrame, labels: dict):
    """Display detailed breakdowns"""
    st.header(":material/bar_chart: Detailed Wage Analysis")
    
    # Analysis selector
    analysis_type = st.selectbox(
        "Select Analysis Dimension",
        ["Education Level", "Occupation", "Full-time vs Part-time", "Union Status"]
    )
    
    if analysis_type == "Education Level" and 'EDUC' in df.columns:
        display_breakdown_chart(df, 'EDUC', labels, 'Education Level')
    
    elif analysis_type == "Occupation" and 'NOC_10' in df.columns:
        display_breakdown_chart(df, 'NOC_10', labels, 'Occupation')
    
    elif analysis_type == "Full-time vs Part-time" and 'FTPTMAIN' in df.columns:
        display_breakdown_chart(df, 'FTPTMAIN', labels, 'Employment Type')
    
    elif analysis_type == "Union Status" and 'UNION' in df.columns:
        display_breakdown_chart(df, 'UNION', labels, 'Union Status')
    
    # Intersectional analysis
    st.subheader("Intersectional Analysis")
    
    if 'GENDER' in df.columns and 'EDUC' in df.columns:
        df_plot = df.copy()
        df_plot['Gender'] = df_plot['GENDER'].map(labels['GENDER'])
        df_plot['Education'] = df_plot['EDUC'].map(labels['EDUC'])
        
        pivot = df_plot.groupby(['Education', 'Gender'])['HRLYEARN'].mean().unstack()
        
        fig = go.Figure()
        
        for gender in ['Male', 'Female']:
            if gender in pivot.columns:
                fig.add_trace(go.Scatter(
                    x=pivot.index,
                    y=pivot[gender],
                    mode='lines+markers',
                    name=gender,
                    marker=dict(size=10)
                ))
        
        fig.update_layout(
            title='Wage by Education Level and Gender',
            xaxis_title='Education Level',
            yaxis_title='Average Hourly Wage ($)',
            legend_title='Gender',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)


def display_breakdown_chart(df: pd.DataFrame, column: str, labels: dict, title: str):
    """Display wage breakdown by a category"""
    df_plot = df.copy()
    df_plot['Category'] = df_plot[column].map(labels.get(column, {}))
    
    if 'GENDER' in df.columns:
        df_plot['Gender'] = df_plot['GENDER'].map(labels['GENDER'])
        
        summary = df_plot.groupby(['Category', 'Gender']).agg({
            'HRLYEARN': ['mean', 'count']
        }).reset_index()
        summary.columns = ['Category', 'Gender', 'Mean Wage', 'Count']
        
        fig = px.bar(
            summary,
            x='Category',
            y='Mean Wage',
            color='Gender',
            barmode='group',
            title=f'Average Wage by {title} and Gender',
            labels={'Mean Wage': 'Average Hourly Wage ($)'},
            color_discrete_map={'Male': '#1f77b4', 'Female': '#e377c2'}
        )
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        summary = df_plot.groupby('Category')['HRLYEARN'].mean().reset_index()
        
        fig = px.bar(
            summary,
            x='Category',
            y='HRLYEARN',
            title=f'Average Wage by {title}',
            labels={'HRLYEARN': 'Average Hourly Wage ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)


def display_model_predictions(df: pd.DataFrame, labels: dict):
    """Display salary predictions using the trained ML ensemble model."""
    st.header(":material/payments: Salary Prediction Tool")

    st.markdown("""
    <div class="insight-box">
        <strong>ML Ensemble Model:</strong> Predictions are generated by a trained ensemble of 
        XGBoost, LightGBM, CatBoost, Random Forest, Gradient Boosting, and Ridge models 
        fitted on 19.5 M Labour Force Survey records.
    </div>
    """, unsafe_allow_html=True)

    # Load trained model and feature engineer
    from src.models import SalaryPredictor as _SP

    model_path = Path("models/salary_predictor.joblib")
    fe_path = Path("models/feature_engineer.joblib")

    if not model_path.exists() or not fe_path.exists():
        st.warning("Trained model not found. Run `python run_pipeline.py` first to train the model.")
        return

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", list(GENDER_CODES.values()))
        education = st.selectbox("Education Level", list(EDUCATION_CODES.values()))
        occupation = st.selectbox("Occupation Category", list(NOC_10_CODES.values()))
        province = st.selectbox("Province", list(PROVINCE_CODES.values()))

    with col2:
        age = st.slider("Age", 15, 70, 35)
        employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time"])
        union_status = st.selectbox("Union Status", [
            "Union member",
            "Covered by collective agreement",
            "Non-unionized"
        ])
        hours_per_week = st.slider("Usual Weekly Hours", 1, 60, 37)

    if st.button("Predict Salary", type="primary"):
        try:
            # Reverse-map human labels → numeric codes
            gender_code = {v: k for k, v in GENDER_CODES.items()}[gender]
            educ_code = {v: k for k, v in EDUCATION_CODES.items()}[education]
            noc_code = {v: k for k, v in NOC_10_CODES.items()}[occupation]
            prov_code = {v: k for k, v in PROVINCE_CODES.items()}[province]
            ft_code = 1 if employment_type == "Full-time" else 2
            union_code = {
                "Union member": 1,
                "Covered by collective agreement": 2,
                "Non-unionized": 3
            }[union_status]

            # Build a single-row DataFrame matching the feature engineer's expectations
            row = {
                'AGE_APPROX': float(age),
                'EXPERIENCE_PROXY': max(float(age) - 18, 0),
                'TENURE': 5.0,  # default mid-career
                'UHRSMAIN': float(hours_per_week),
                'GENDER': gender_code,
                'EDUC': educ_code,
                'NOC_10': noc_code,
                'NAICS_21': 0,
                'PROV': prov_code,
                'FTPTMAIN': ft_code,
                'PERMTEMP': 1,
                'UNION': union_code,
                'ESTSIZE': 3,
                'MARSTAT': 6,
            }
            input_df = pd.DataFrame([row])

            # Load and predict
            fe = FeatureEngineer()
            fe.load(str(fe_path))
            X = fe.transform(input_df)

            predictor = _SP()
            predictor.load(str(model_path))
            preds, lower, upper = predictor.predict_with_confidence(X, confidence=0.95)
            predicted_wage = float(preds[0])
            ci_low = float(lower[0])
            ci_high = float(upper[0])

            # --- Results ---
            st.success(f"### Predicted Hourly Wage: ${predicted_wage:.2f}")
            st.info(f"95 % Confidence Interval: ${ci_low:.2f} – ${ci_high:.2f}")

            # Compare to overall averages
            overall_mean = df['HRLYEARN'].mean()
            diff = predicted_wage - overall_mean
            diff_pct = (diff / overall_mean) * 100

            if diff > 0:
                st.markdown(f"This is **${diff:.2f}/hr ({diff_pct:+.1f}%)** above the overall average of ${overall_mean:.2f}/hr.")
            else:
                st.markdown(f"This is **${abs(diff):.2f}/hr ({diff_pct:.1f}%)** below the overall average of ${overall_mean:.2f}/hr.")

            # Gender counterfactual
            other_code = 2 if gender_code == 1 else 1
            row_cf = row.copy()
            row_cf['GENDER'] = other_code
            cf_df = pd.DataFrame([row_cf])
            X_cf = fe.transform(cf_df)
            cf_pred = float(predictor.predict(X_cf)[0])
            gap_dollar = predicted_wage - cf_pred
            gap_pct = (gap_dollar / max(predicted_wage, cf_pred)) * 100

            other_label = GENDER_CODES[other_code]
            st.markdown(f"""
            **Gender counterfactual:** An identical {other_label} worker would earn 
            **${cf_pred:.2f}/hr** — a difference of **${abs(gap_dollar):.2f}/hr ({abs(gap_pct):.1f}%)**.
            """)

            # Feature importance
            with st.expander("Model feature importance"):
                try:
                    imp_df = predictor.get_feature_importance()
                    fig = px.bar(
                        imp_df.head(15),
                        x='importance', y='feature',
                        orientation='h',
                        title='Top 15 Feature Importances',
                        labels={'importance': 'Importance', 'feature': 'Feature'}
                    )
                    fig.update_layout(height=400, yaxis={'autorange': 'reversed'})
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Feature importance data not available for this model configuration.")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")


def display_recommendations(df: pd.DataFrame, labels: dict):
    """Display data-driven recommendations based on actual analysis results"""
    st.header(":material/checklist: Data-Driven Recommendations")

    gender_col = 'GENDER' if 'GENDER' in df.columns else ('SEX' if 'SEX' in df.columns else None)
    wage_col = 'HRLYEARN'
    has_gender = gender_col is not None

    # ── Compute findings from the data ──────────────────────────────
    findings = []

    # 1. Overall gender wage gap
    if has_gender:
        male_wage = weighted_mean(df[df[gender_col] == 1][wage_col], df[df[gender_col] == 1]['FINALWT']) if 'FINALWT' in df.columns else df[df[gender_col] == 1][wage_col].mean()
        female_wage = weighted_mean(df[df[gender_col] == 2][wage_col], df[df[gender_col] == 2]['FINALWT']) if 'FINALWT' in df.columns else df[df[gender_col] == 2][wage_col].mean()
        gap_pct = ((male_wage - female_wage) / male_wage) * 100 if male_wage > 0 else 0

        if gap_pct > 15:
            sev = 'high'
        elif gap_pct > 10:
            sev = 'medium'
        else:
            sev = 'low'
        findings.append({
            'severity': sev,
            'area': 'Overall Wage Gap',
            'finding': f"Raw gender wage gap is {gap_pct:.1f}% (Male ${male_wage:.2f}/hr vs Female ${female_wage:.2f}/hr)",
            'recommendation': 'Conduct a formal pay equity audit under the Pay Equity Act and develop a remediation timeline.' if sev == 'high' else ('Review compensation practices for systemic bias.' if sev == 'medium' else 'Continue monitoring; gap is within acceptable range.')
        })

    # 2. Worst occupation gap
    if has_gender and 'NOC_10' in df.columns:
        occ_gaps = []
        for occ in df['NOC_10'].dropna().unique():
            occ_df = df[df['NOC_10'] == occ]
            m = occ_df[occ_df[gender_col] == 1][wage_col].mean()
            f = occ_df[occ_df[gender_col] == 2][wage_col].mean()
            n = len(occ_df)
            if pd.notna(m) and pd.notna(f) and m > 0 and n > 100:
                occ_gaps.append({'code': occ, 'gap': ((m - f) / m) * 100, 'n': n})
        if occ_gaps:
            occ_gaps.sort(key=lambda x: x['gap'], reverse=True)
            worst = occ_gaps[0]
            occ_label = labels.get('NOC_10', {}).get(worst['code'], str(worst['code']))
            findings.append({
                'severity': 'high' if worst['gap'] > 15 else 'medium',
                'area': 'Occupation',
                'finding': f"Largest occupational gap in **{occ_label}**: {worst['gap']:.1f}% (n={worst['n']:,})",
                'recommendation': f"Prioritise pay equity review for {occ_label} roles — this sector shows the widest disparity."
            })

    # 3. Worst province gap
    if has_gender and 'PROV' in df.columns:
        prov_gaps = []
        for prov in df['PROV'].dropna().unique():
            p_df = df[df['PROV'] == prov]
            m = p_df[p_df[gender_col] == 1][wage_col].mean()
            f = p_df[p_df[gender_col] == 2][wage_col].mean()
            if pd.notna(m) and pd.notna(f) and m > 0 and len(p_df) > 200:
                prov_gaps.append({'code': prov, 'gap': ((m - f) / m) * 100})
        if prov_gaps:
            prov_gaps.sort(key=lambda x: x['gap'], reverse=True)
            worst_p = prov_gaps[0]
            best_p = prov_gaps[-1]
            worst_label = labels.get('PROV', {}).get(worst_p['code'], str(worst_p['code']))
            best_label = labels.get('PROV', {}).get(best_p['code'], str(best_p['code']))
            findings.append({
                'severity': 'high' if worst_p['gap'] > 15 else 'medium',
                'area': 'Province',
                'finding': f"Provincial gap ranges from **{best_label}** ({best_p['gap']:.1f}%) to **{worst_label}** ({worst_p['gap']:.1f}%)",
                'recommendation': f"Investigate labour-market conditions in {worst_label} that drive the widest provincial gap."
            })

    # 4. Education effect
    if has_gender and 'EDUC' in df.columns:
        educ_gaps = []
        for ed in sorted(df['EDUC'].dropna().unique()):
            e_df = df[df['EDUC'] == ed]
            m = e_df[e_df[gender_col] == 1][wage_col].mean()
            f = e_df[e_df[gender_col] == 2][wage_col].mean()
            if pd.notna(m) and pd.notna(f) and m > 0 and len(e_df) > 100:
                educ_gaps.append({'code': ed, 'gap': ((m - f) / m) * 100})
        if educ_gaps:
            widest = max(educ_gaps, key=lambda x: x['gap'])
            ed_label = labels.get('EDUC', {}).get(widest['code'], str(widest['code']))
            findings.append({
                'severity': 'medium' if widest['gap'] > 10 else 'low',
                'area': 'Education',
                'finding': f"Widest gap at education level **{ed_label}**: {widest['gap']:.1f}%",
                'recommendation': f"Examine whether credentials are valued equally regardless of gender in {ed_label} roles."
            })

    # 5. Employment type differential
    if has_gender and 'FTPTMAIN' in df.columns:
        ft_m = df[(df[gender_col] == 1) & (df['FTPTMAIN'] == 1)][wage_col].mean()
        ft_f = df[(df[gender_col] == 2) & (df['FTPTMAIN'] == 1)][wage_col].mean()
        pt_f_pct = ((df[gender_col] == 2) & (df['FTPTMAIN'] == 2)).sum() / max((df[gender_col] == 2).sum(), 1) * 100
        pt_m_pct = ((df[gender_col] == 1) & (df['FTPTMAIN'] == 2)).sum() / max((df[gender_col] == 1).sum(), 1) * 100
        if pd.notna(ft_m) and pd.notna(ft_f) and ft_m > 0:
            ft_gap = ((ft_m - ft_f) / ft_m) * 100
            findings.append({
                'severity': 'medium' if pt_f_pct > pt_m_pct * 1.5 else 'low',
                'area': 'Employment Type',
                'finding': f"Full-time gender gap: {ft_gap:.1f}%. Women in part-time: {pt_f_pct:.0f}% vs Men: {pt_m_pct:.0f}%",
                'recommendation': 'Ensure part-time workers receive proportional benefits and that part-time status does not penalise career advancement.' if pt_f_pct > pt_m_pct * 1.5 else 'Part-time distribution is relatively balanced.'
            })

    # 6. Union effect
    if has_gender and 'UNION' in df.columns:
        union_df = df[df['UNION'].isin([1, 2])]
        non_union_df = df[df['UNION'] == 3]
        if len(union_df) > 100 and len(non_union_df) > 100:
            u_gap_m = union_df[union_df[gender_col] == 1][wage_col].mean()
            u_gap_f = union_df[union_df[gender_col] == 2][wage_col].mean()
            n_gap_m = non_union_df[non_union_df[gender_col] == 1][wage_col].mean()
            n_gap_f = non_union_df[non_union_df[gender_col] == 2][wage_col].mean()
            u_gap = ((u_gap_m - u_gap_f) / u_gap_m * 100) if u_gap_m > 0 else 0
            n_gap = ((n_gap_m - n_gap_f) / n_gap_m * 100) if n_gap_m > 0 else 0
            findings.append({
                'severity': 'low' if u_gap < n_gap else 'medium',
                'area': 'Unionisation',
                'finding': f"Unionised gap: {u_gap:.1f}% vs Non-unionised gap: {n_gap:.1f}%",
                'recommendation': 'Collective bargaining appears to narrow the gap — consider supporting pay-transparency provisions.' if u_gap < n_gap else 'Examine collective agreement structures for potential bias.'
            })

    # ── Display findings ────────────────────────────────────────────
    st.markdown("""
    <div class="insight-box">
        <strong>All recommendations below are computed from the current filtered dataset.</strong>
        Adjust sidebar filters to see how findings change by province, education level, or time period.
    </div>
    """, unsafe_allow_html=True)

    for f in findings:
        icon = {
            'high': ':material/error:',
            'medium': ':material/warning:',
            'low': ':material/check_circle:'
        }[f['severity']]
        color = {'high': '#ffc7ce', 'medium': '#ffeb9c', 'low': '#c6efce'}[f['severity']]
        st.markdown(f"""
<div style="background:{color}; padding:12px 16px; border-radius:8px; margin-bottom:10px;">
{icon} <strong>[{f['area']}]</strong> {f['finding']}<br>
<em>Recommendation:</em> {f['recommendation']}
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # Compliance note
    st.markdown("""
    <div class="warning-box">
        <h4>Canadian Pay Equity Legislation</h4>
        <p>The <strong>Pay Equity Act</strong> requires federally regulated employers to establish a 
        pay equity plan that identifies and corrects gender-based pay discrimination. 
        Provincial legislation may also apply to your organization.</p>
        <p>Key requirements:</p>
        <ul>
            <li>Identify job classes and determine their value</li>
            <li>Compare compensation for female and male job classes of equal value</li>
            <li>Develop a plan to increase compensation where gaps exist</li>
            <li>Post the pay equity plan within three years</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# NEW ANALYSIS TABS - Reflecting the full project
# ============================================================================

@st.cache_data
def load_time_series_data():
    """Load time series wage data using DuckDB (memory-efficient)"""
    try:
        store = EquiPayDataStore(memory_limit_mb=3000)
        # Aggregate yearly wage data by gender directly in DuckDB
        df = store.sql("""
            SELECT 
                SURVYEAR as year,
                GENDER as SEX,
                AVG(HRLYEARN) as avg_wage,
                MEDIAN(HRLYEARN) as median_wage,
                COUNT(*) as n
            FROM lfs_enriched
            WHERE HRLYEARN IS NOT NULL
            GROUP BY SURVYEAR, GENDER
            ORDER BY SURVYEAR, GENDER
        """)
        return df
    except Exception as e:
        # Fallback to CSV if available
        ts_path = Path("data/processed/yearly_wages_by_gender.csv")
        if ts_path.exists():
            return pd.read_csv(ts_path)
        return None


def display_decomposition(df: pd.DataFrame, labels: dict, equity_dim: EquityDimension = None):
    """Display comprehensive wage gap decomposition analysis for any equity dimension"""
    
    # Default to gender if no dimension specified
    if equity_dim is None:
        equity_dim = get_equity_dimension('gender')
    
    st.header(f"Wage Gap Decomposition: {equity_dim.description}")
    
    st.markdown(f"""
    <div class="insight-box">
        <strong>What is Wage Decomposition?</strong> This analysis separates the pay gap into two components:
        <ul>
            <li><strong>Explained Gap</strong>: Differences due to measurable characteristics (education, occupation, location, employment type)</li>
            <li><strong>Unexplained Gap</strong>: Residual difference after accounting for all observables</li>
        </ul>
        <p style="margin-top: 10px;"><strong>Comparing:</strong> {equity_dim.reference_label} vs {equity_dim.comparison_label}</p>
    </div>
    """, unsafe_allow_html=True)
    
    group_col = equity_dim.column
    
    # Support both SEX and GENDER columns
    if group_col not in df.columns:
        if group_col == 'GENDER' and 'SEX' in df.columns:
            group_col = 'SEX'
        else:
            st.warning(f"{equity_dim.column} data not available")
            return
    
    try:
        # Prepare data - ensure numeric types
        df_clean = df.copy()
        numeric_cols = ['HRLYEARN', 'EDUC', 'AGE_6', 'AGE_12', 'NOC_10', 'PROV', 'FTPTMAIN', 'UNION', 'PERMTEMP']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype('float64')
        
        # Filter to two groups being compared
        df_analysis = df_clean[df_clean[group_col].isin([equity_dim.reference_value, equity_dim.comparison_value])].copy()
        
        # Calculate group means
        ref_df = df_analysis[df_analysis[group_col] == equity_dim.reference_value]
        comp_df = df_analysis[df_analysis[group_col] == equity_dim.comparison_value]
        
        ref_mean = ref_df['HRLYEARN'].mean()
        comp_mean = comp_df['HRLYEARN'].mean()
        gap_dollars = ref_mean - comp_mean
        gap_pct = (gap_dollars / ref_mean) * 100 if ref_mean > 0 else 0
        
        # === TOP METRICS ===
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"{equity_dim.reference_label} Avg Wage", f"${ref_mean:.2f}/hr")
        with col2:
            st.metric(f"{equity_dim.comparison_label} Avg Wage", f"${comp_mean:.2f}/hr")
        with col3:
            st.metric("Wage Gap", f"${gap_dollars:.2f}/hr", delta=f"{gap_pct:.1f}%", delta_color="inverse")
        with col4:
            ratio = comp_mean / ref_mean if ref_mean > 0 else 0
            st.metric("Wage Ratio", f"${ratio:.2f}", help=f"{equity_dim.comparison_label} earn ${ratio:.2f} for every $1.00 {equity_dim.reference_label} earn")
        
        # === CONTROL VARIABLES ===
        control_vars = [c for c in ['EDUC', 'AGE_6', 'NOC_10', 'PROV', 'FTPTMAIN', 'UNION'] if c in df_analysis.columns]
        
        if not control_vars:
            st.warning("No control variables available for decomposition")
            return
        
        st.markdown(f"**Control Variables Used:** {', '.join([labels.get(c, {}).get('label', c) for c in control_vars])}")
        
        # === RUN DECOMPOSITION ===
        df_reg = df_analysis[['HRLYEARN', group_col] + control_vars].dropna()
        
        # Ensure all columns are numeric float64 (pandas nullable types cause issues with statsmodels)
        for col in control_vars:
            df_reg[col] = pd.to_numeric(df_reg[col], errors='coerce').astype('float64')
        
        df_reg['IS_COMPARISON'] = (df_reg[group_col] == equity_dim.comparison_value).astype('float64')
        df_reg['LOG_WAGE'] = np.log(df_reg['HRLYEARN'].clip(lower=1)).astype('float64')
        
        import statsmodels.api as sm
        
        # Model 1: Unadjusted (no controls)
        X_unadj = sm.add_constant(df_reg['IS_COMPARISON'])
        y = df_reg['LOG_WAGE']
        model_unadj = sm.OLS(y, X_unadj).fit()
        unadj_coef = model_unadj.params['IS_COMPARISON']
        unadj_gap_pct = (np.exp(unadj_coef) - 1) * 100
        
        # Model 2: Adjusted (with controls)
        X_adj = sm.add_constant(df_reg[['IS_COMPARISON'] + control_vars])
        model_adj = sm.OLS(y, X_adj).fit()
        adj_coef = model_adj.params['IS_COMPARISON']
        adj_gap_pct = (np.exp(adj_coef) - 1) * 100
        
        explained_pct = unadj_gap_pct - adj_gap_pct
        unexplained_pct = adj_gap_pct
        explained_pct_of_total = (explained_pct / abs(unadj_gap_pct) * 100) if unadj_gap_pct != 0 else 0
        unexplained_pct_of_total = (unexplained_pct / abs(unadj_gap_pct) * 100) if unadj_gap_pct != 0 else 0
        
        # === KEY RESULTS ===
        st.subheader("Decomposition Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Gap", f"{unadj_gap_pct:.2f}%", 
                     help="Raw wage gap before controlling for characteristics")
        with col2:
            st.metric("Explained", f"{explained_pct:.2f}%", 
                     delta=f"{explained_pct_of_total:.0f}% of total",
                     help="Gap explained by differences in education, occupation, etc.")
        with col3:
            st.metric("Unexplained", f"{unexplained_pct:.2f}%",
                     delta=f"{unexplained_pct_of_total:.0f}% of total",
                     delta_color="inverse",
                     help="Gap remaining after controlling for observables")
        
        # === VISUALIZATIONS ===
        tab1, tab2, tab3 = st.tabs(["Decomposition", "Detailed Breakdown", "Characteristics Comparison"])
        
        with tab1:
            # Waterfall/stacked bar showing decomposition
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Explained',
                x=['Wage Gap Decomposition'],
                y=[explained_pct],
                marker_color='#2ca02c',
                text=[f'{explained_pct:.1f}%<br>({explained_pct_of_total:.0f}% of total)'],
                textposition='inside',
                hovertemplate='<b>Explained by Characteristics</b><br>%{y:.2f}%<extra></extra>'
            ))
            
            fig.add_trace(go.Bar(
                name='Unexplained',
                x=['Wage Gap Decomposition'],
                y=[unexplained_pct],
                marker_color='#d62728',
                text=[f'{unexplained_pct:.1f}%<br>({unexplained_pct_of_total:.0f}% of total)'],
                textposition='inside',
                hovertemplate='<b>Unexplained Component</b><br>%{y:.2f}%<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"Total Gap: {unadj_gap_pct:.2f}% ({equity_dim.reference_label} earn {abs(unadj_gap_pct):.1f}% more)",
                barmode='stack',
                yaxis_title="Wage Gap (%)",
                height=400,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Dynamic interpretation
            pct_explained = abs(explained_pct_of_total)
            pct_unexplained = abs(unexplained_pct_of_total)
            
            if pct_unexplained > 70:
                insight_color = "warning-box"
                explanation = f"**The unexplained portion ({unexplained_pct:.1f}%) represents {pct_unexplained:.0f}% of the total gap.** This large unexplained component may indicate systemic barriers, unobserved skill differences, or measurement limitations in the data."
            elif pct_unexplained > 40:
                insight_color = "insight-box"
                explanation = f"**The gap is split relatively evenly:** {pct_explained:.0f}% is explained by measurable characteristics and {pct_unexplained:.0f}% remains unexplained. Both observed differences and potential systemic factors contribute to the wage differential."
            else:
                insight_color = "insight-box"
                explanation = f"**Most of the gap ({pct_explained:.0f}%) is explained by differences** in education, occupation, experience, and location. The unexplained portion ({pct_unexplained:.0f}%) is relatively small, suggesting measured characteristics account for most of the wage difference."
            
            st.markdown(f"""
            <div class="{insight_color}">
                <strong>Finding:</strong> {explanation}
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.subheader("Contribution by Control Variable")
            
            # Show how much each control variable matters
            # Calculate difference in means for each control
            contributions = []
            for var in control_vars:
                ref_val = ref_df[var].mean()
                comp_val = comp_df[var].mean()
                diff = comp_val - ref_val
                diff_pct = (diff / ref_val * 100) if ref_val != 0 else 0
                
                var_label = labels.get(var, {}).get('label', var)
                contributions.append({
                    'Variable': var_label,
                    'Reference Mean': ref_val,
                    'Comparison Mean': comp_val,
                    'Difference': diff,
                    'Difference %': diff_pct
                })
            
            contrib_df = pd.DataFrame(contributions)
            
            # Bar chart of differences
            fig = px.bar(
                contrib_df.sort_values('Difference %', key=abs, ascending=True),
                y='Variable',
                x='Difference %',
                orientation='h',
                title="Average Characteristic Differences (% deviation from reference group)",
                labels={'Difference %': 'Percent Difference'},
                color='Difference %',
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(
                contrib_df.style.format({
                    'Reference Mean': '{:.2f}',
                    'Comparison Mean': '{:.2f}',
                    'Difference': '{:.2f}',
                    'Difference %': '{:.1f}%'
                }),
                use_container_width=True
            )
        
        with tab3:
            st.subheader("How Groups Differ in Observable Characteristics")
            
            st.markdown("""
            This section reveals where the two groups differ most in their characteristics.
            Understanding these differences helps explain why wages vary between groups.
            """)
            
            # === ANALYZE ALL CHARACTERISTICS ===
            comparison_vars = [v for v in ['EDUC', 'NOC_10', 'PROV', 'FTPTMAIN', 'UNION'] if v in df_analysis.columns]
            
            # Calculate differences for summary table
            differences_summary = []
            for var in comparison_vars:
                var_label = labels.get(var, {}).get('label', var)
                
                # Calculate distributions
                ref_dist = ref_df[var].value_counts(normalize=True) * 100
                comp_dist = comp_df[var].value_counts(normalize=True) * 100
                
                # Calculate total variation distance (how different the distributions are)
                all_categories = set(ref_dist.index) | set(comp_dist.index)
                tvd = sum(abs(ref_dist.get(cat, 0) - comp_dist.get(cat, 0)) for cat in all_categories) / 2
                
                # Find category with largest difference
                diff_by_cat = {cat: abs(ref_dist.get(cat, 0) - comp_dist.get(cat, 0)) for cat in all_categories}
                max_diff_cat = max(diff_by_cat, key=diff_by_cat.get)
                max_diff = diff_by_cat[max_diff_cat]
                
                # Get category label
                cat_label = labels.get(var, {}).get('values', {}).get(max_diff_cat, f'Category {max_diff_cat}')
                
                differences_summary.append({
                    'Characteristic': var_label,
                    'Total Difference': tvd,
                    'Biggest Gap Category': cat_label,
                    'Gap Size': max_diff
                })
            
            summary_df = pd.DataFrame(differences_summary).sort_values('Total Difference', ascending=False)
            
            # Display summary
            st.markdown("### Difference Summary")
            st.markdown(f"""
            The **Total Difference** score (0-100%) measures how different the two groups are on each characteristic.
            Higher scores indicate larger distributional differences.
            """)
            
            # Create horizontal bar chart for differences
            fig_summary = go.Figure()
            fig_summary.add_trace(go.Bar(
                x=summary_df['Total Difference'],
                y=summary_df['Characteristic'],
                orientation='h',
                marker_color=summary_df['Total Difference'],
                marker_colorscale='Reds',
                text=[f"{val:.1f}%" for val in summary_df['Total Difference']],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Total Difference: %{x:.1f}%<extra></extra>'
            ))
            fig_summary.update_layout(
                title="Where Groups Differ Most",
                xaxis_title="Total Variation Distance (%)",
                height=300,
                showlegend=False,
                xaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_summary, use_container_width=True)
            
            # Display detailed table
            st.dataframe(
                summary_df.style.format({
                    'Total Difference': '{:.1f}%',
                    'Gap Size': '{:.1f} pp'
                }).background_gradient(subset=['Total Difference'], cmap='Reds', vmin=0, vmax=50),
                use_container_width=True,
                hide_index=True
            )
            
            # === DETAILED DISTRIBUTION COMPARISONS ===
            st.markdown("### Detailed Distribution Comparisons")
            
            # Focus on top 3 variables with most difference
            top_vars = summary_df.head(3)['Characteristic'].tolist()
            top_var_cols = [var for var in comparison_vars if labels.get(var, {}).get('label', var) in top_vars]
            
            for var in top_var_cols:
                var_label = labels.get(var, {}).get('label', var)
                
                # Calculate distributions
                ref_dist = ref_df[var].value_counts(normalize=True).sort_index() * 100
                comp_dist = comp_df[var].value_counts(normalize=True).sort_index() * 100
                
                # Merge and prepare for plotting
                dist_df = pd.DataFrame({
                    'ref': ref_dist,
                    'comp': comp_dist
                }).fillna(0)
                dist_df['diff'] = dist_df['comp'] - dist_df['ref']
                dist_df = dist_df.reset_index()
                dist_df.columns = ['Category', 'ref_pct', 'comp_pct', 'diff_pct']
                
                # Map category codes to labels
                if var in labels and 'values' in labels[var]:
                    dist_df['Category'] = dist_df['Category'].map(
                        lambda x: labels[var]['values'].get(x, f'Category {x}')
                    )
                
                # Sort by absolute difference for better visualization
                dist_df = dist_df.sort_values('diff_pct', key=abs, ascending=False)
                
                st.markdown(f"#### {var_label}")
                
                # Create diverging bar chart showing differences
                fig = go.Figure()
                
                colors = ['#d62728' if x < 0 else '#2ca02c' for x in dist_df['diff_pct']]
                
                fig.add_trace(go.Bar(
                    x=dist_df['diff_pct'],
                    y=dist_df['Category'],
                    orientation='h',
                    marker_color=colors,
                    text=[f"{val:+.1f} pp" for val in dist_df['diff_pct']],
                    textposition='auto',
                    customdata=dist_df[['ref_pct', 'comp_pct']].values,
                    hovertemplate=(
                        '<b>%{y}</b><br>' +
                        f'{equity_dim.reference_label}: %{{customdata[0]:.1f}}%<br>' +
                        f'{equity_dim.comparison_label}: %{{customdata[1]:.1f}}%<br>' +
                        'Difference: %{x:.1f} pp<extra></extra>'
                    )
                ))
                
                fig.update_layout(
                    title=f"Percentage Point Difference in {var_label} Distribution",
                    xaxis_title=f"← More {equity_dim.reference_label}    |    More {equity_dim.comparison_label} →",
                    height=max(300, len(dist_df) * 30),
                    showlegend=False,
                    xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add insight
                max_diff_row = dist_df.iloc[0]
                direction = equity_dim.comparison_label if max_diff_row['diff_pct'] > 0 else equity_dim.reference_label
                st.markdown(f"""
                <div class="insight-box">
                    <strong>Key Insight:</strong> The largest difference is in <strong>{max_diff_row['Category']}</strong>, 
                    where {direction} are {abs(max_diff_row['diff_pct']):.1f} percentage points more represented 
                    ({max_diff_row['ref_pct']:.1f}% vs {max_diff_row['comp_pct']:.1f}%).
                </div>
                """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error performing decomposition: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def display_time_series(equity_dim: EquityDimension = None):
    """Display time series analysis (from Notebook 06)"""
    
    # Default to gender if no dimension specified
    if equity_dim is None:
        equity_dim = get_equity_dimension('gender')
    
    st.header(f":material/timeline: Time Series Analysis: {equity_dim.description}")
    
    st.markdown(f"""
    <div class="insight-box">
        <strong>Data Period:</strong> {DATA_SCOPE_START} - {DATA_SCOPE_END}<br>
        <strong>Source:</strong> Statistics Canada Labour Force Survey (Table 14-10-0064-01)<br>
        <strong>Dimension:</strong> {equity_dim.reference_label} vs {equity_dim.comparison_label}
    </div>
    """, unsafe_allow_html=True)
    
    # For non-gender dimensions, compute time series on the fly
    if equity_dim.column != 'GENDER':
        try:
            cached = load_cached_aggregates()
            if 'annual_gap' in cached and equity_dim.column == 'GENDER':
                ts_data = cached['annual_gap'].rename(columns={'mean_male':'male_wage','mean_female':'female_wage'})
            else:
                store = EquiPayDataStore()
                ts_data = store.wage_gap(
                    group_column=equity_dim.column,
                    reference_value=equity_dim.reference_value,
                    comparison_value=equity_dim.comparison_value,
                    by=['SURVYEAR'],
                    reference_label=equity_dim.reference_label,
                    comparison_label=equity_dim.comparison_label
                )
                ts_data = ts_data.rename(columns={
                'SURVYEAR': 'year',
                'reference_mean': 'reference_wage',
                'comparison_mean': 'comparison_wage'
            })
            ts_data['wage_gap'] = ts_data['gap_pct']
        except Exception as e:
            st.warning(f"Could not compute time series: {str(e)}")
            return
    else:
        # Load pre-computed time series data for gender
        ts_raw = load_time_series_data()
    
        if ts_raw is None or len(ts_raw) == 0:
            st.warning("Time series data not available. Run the data pipeline first.")
            return

        # The query returns long format (year, SEX, avg_wage …).
        # Pivot to wide format with male_wage / female_wage columns.
        gender_col_ts = 'SEX' if 'SEX' in ts_raw.columns else 'GENDER'
        if gender_col_ts in ts_raw.columns and 'avg_wage' in ts_raw.columns:
            male_ts = ts_raw[ts_raw[gender_col_ts] == 1][['year', 'avg_wage']].rename(columns={'avg_wage': 'male_wage'})
            female_ts = ts_raw[ts_raw[gender_col_ts] == 2][['year', 'avg_wage']].rename(columns={'avg_wage': 'female_wage'})
            ts_data = male_ts.merge(female_ts, on='year', how='outer').sort_values('year').reset_index(drop=True)
        else:
            ts_data = ts_raw
    
    # Ensure wage_gap column exists
    if 'wage_gap' not in ts_data.columns and 'male_wage' in ts_data.columns:
        ts_data['wage_gap'] = (ts_data['male_wage'] - ts_data['female_wage']) / ts_data['male_wage'] * 100
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    first_gap = ts_data['wage_gap'].iloc[0]
    last_gap = ts_data['wage_gap'].iloc[-1]
    change = last_gap - first_gap
    
    with col1:
        st.metric(f"Gap in {int(ts_data['year'].iloc[0])}", f"{first_gap:.1f}%")
    with col2:
        st.metric(f"Gap in {int(ts_data['year'].iloc[-1])}", f"{last_gap:.1f}%")
    with col3:
        st.metric("Change", f"{change:+.1f}pp", delta_color="inverse" if change > 0 else "normal")
    with col4:
        years = len(ts_data)
        annual_change = change / years if years > 1 else 0
        st.metric("Annual Trend", f"{annual_change:+.2f}pp/yr")
    
    # Main trend chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Wage gap line
    fig.add_trace(
        go.Scatter(x=ts_data['year'], y=ts_data['wage_gap'], 
                   mode='lines+markers', name='Wage Gap (%)',
                   line=dict(color='purple', width=3)),
        secondary_y=False
    )
    
    # Add trend line
    z = np.polyfit(ts_data['year'], ts_data['wage_gap'], 1)
    trend_line = np.poly1d(z)(ts_data['year'])
    fig.add_trace(
        go.Scatter(x=ts_data['year'], y=trend_line,
                   mode='lines', name=f'Trend (slope={z[0]:.3f})',
                   line=dict(color='red', dash='dash')),
        secondary_y=False
    )
    
    # Male/Female wages on secondary axis (or reference/comparison for other dimensions)
    ref_col = 'male_wage' if 'male_wage' in ts_data.columns else 'reference_wage'
    comp_col = 'female_wage' if 'female_wage' in ts_data.columns else 'comparison_wage'
    
    if ref_col in ts_data.columns:
        fig.add_trace(
            go.Scatter(x=ts_data['year'], y=ts_data[ref_col],
                       mode='lines', name=f'{equity_dim.reference_label} Wage ($)',
                       line=dict(color='blue', width=1), opacity=0.6),
            secondary_y=True
        )
        fig.add_trace(
            go.Scatter(x=ts_data['year'], y=ts_data[comp_col],
                       mode='lines', name=f'{equity_dim.comparison_label} Wage ($)',
                       line=dict(color='pink', width=1), opacity=0.6),
            secondary_y=True
        )
    
    fig.update_layout(
        title='Gender Wage Gap Evolution in Canada',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Wage Gap (%)", secondary_y=False)
    fig.update_yaxes(title_text="Hourly Wage ($)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Economic periods overlay
    st.subheader("Economic Context")
    
    period_data = []
    for period, (start, end) in ECONOMIC_PERIODS.items():
        period_ts = ts_data[(ts_data['year'] >= start) & (ts_data['year'] <= end)]
        if len(period_ts) > 0:
            period_data.append({
                'Period': period.replace('_', ' ').title(),
                'Years': f"{start}-{end}",
                'Avg Gap': period_ts['wage_gap'].mean(),
                'Start Gap': period_ts['wage_gap'].iloc[0] if len(period_ts) > 0 else None,
                'End Gap': period_ts['wage_gap'].iloc[-1] if len(period_ts) > 0 else None
            })
    
    if period_data:
        period_df = pd.DataFrame(period_data)
        st.dataframe(period_df.round(2), use_container_width=True, hide_index=True)
    
    # Interpretation
    if z[0] < 0:
        st.success(f":material/trending_down: **Trend:** The wage gap is narrowing at approximately {abs(z[0]):.2f} percentage points per year.")
    else:
        st.warning(f":material/trending_up: **Trend:** The wage gap is widening at approximately {z[0]:.2f} percentage points per year.")


def display_fairness_metrics(df: pd.DataFrame, labels: dict):
    """Display fairness metrics analysis (from Notebook 04)"""
    st.header(":material/target: Fairness & Bias Metrics")
    
    st.markdown("""
    <div class="insight-box">
        <strong>Fairness Metrics:</strong> These metrics quantify potential discrimination 
        across protected attributes, aligned with algorithmic fairness standards and 
        Canadian human rights legislation.
    </div>
    """, unsafe_allow_html=True)
    
    if 'GENDER' not in df.columns and 'SEX' not in df.columns:
        st.warning("Gender data not available for fairness analysis")
        return
    
    try:
        # Use actual column names from the dataframe
        gender_col = 'SEX' if 'SEX' in df.columns else COLS.GENDER
        wage_col = 'HRLYEARN' if 'HRLYEARN' in df.columns else COLS.WAGE
        
        analyzer = FairnessAnalyzer(protected_features=[gender_col])
        
        # Calculate basic fairness metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Disparate Impact Ratio")
            
            # Calculate ratio of female to male average wages
            male_mean = df[df[gender_col] == 1][wage_col].mean()
            female_mean = df[df[gender_col] == 2][wage_col].mean()
            di_ratio = female_mean / male_mean if male_mean > 0 else 0
            
            # 4/5ths rule threshold
            threshold = 0.8
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=di_ratio,
                delta={'reference': threshold, 'relative': False},
                gauge={
                    'axis': {'range': [0, 1.2]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.8], 'color': "#ffcccc"},
                        {'range': [0.8, 1.0], 'color': "#ffffcc"},
                        {'range': [1.0, 1.2], 'color': "#ccffcc"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.8
                    }
                },
                title={'text': "Female/Male Wage Ratio"}
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            if di_ratio < 0.8:
                st.error(f":material/gpp_bad: **Fails 4/5ths Rule**: Ratio of {di_ratio:.2f} is below 0.80 threshold")
            elif di_ratio < 0.9:
                st.warning(f":material/warning: **Marginal**: Ratio of {di_ratio:.2f} passes 4/5ths rule but gap exists")
            else:
                st.success(f":material/verified: **Passes**: Ratio of {di_ratio:.2f} indicates near-parity")
        
        with col2:
            st.subheader("Equal Opportunity Analysis")
            
            # Calculate opportunity metrics by wage quartile
            df['wage_quartile'] = pd.qcut(df[wage_col], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            
            opportunity_data = []
            for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                q_df = df[df['wage_quartile'] == q]
                male_pct = (q_df[gender_col] == 1).sum() / (df[gender_col] == 1).sum() * 100
                female_pct = (q_df[gender_col] == 2).sum() / (df[gender_col] == 2).sum() * 100
                opportunity_data.append({
                    'Quartile': q,
                    'Male %': male_pct,
                    'Female %': female_pct
                })
            
            opp_df = pd.DataFrame(opportunity_data)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Male', x=opp_df['Quartile'], y=opp_df['Male %'], marker_color='#1f77b4'))
            fig.add_trace(go.Bar(name='Female', x=opp_df['Quartile'], y=opp_df['Female %'], marker_color='#e377c2'))
            fig.update_layout(
                barmode='group',
                title='Distribution Across Wage Quartiles',
                yaxis_title='% of Gender Group',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Fairness by attribute
        st.subheader("Wage Gap by Protected Attributes")
        
        protected_attrs = ['EDUC', 'NOC_10', 'PROV', 'AGE_6']
        available_attrs = [a for a in protected_attrs if a in df.columns]
        
        if available_attrs:
            selected_attr = st.selectbox("Select attribute", available_attrs)
            
            # Calculate gap by attribute
            gap_data = []
            for val in df[selected_attr].unique():
                subset = df[df[selected_attr] == val]
                if len(subset) > 50:  # Minimum sample size
                    m = subset[subset[gender_col] == 1][wage_col].mean()
                    f = subset[subset[gender_col] == 2][wage_col].mean()
                    if pd.notna(m) and pd.notna(f) and m > 0:
                        gap_data.append({
                            'Category': labels.get(selected_attr, {}).get(val, str(val)),
                            'Gap %': ((m - f) / m) * 100,
                            'Male Wage': m,
                            'Female Wage': f
                        })
            
            if gap_data:
                gap_df = pd.DataFrame(gap_data).sort_values('Gap %', ascending=False)
                
                fig = px.bar(
                    gap_df, y='Category', x='Gap %',
                    orientation='h',
                    color='Gap %',
                    color_continuous_scale=['green', 'yellow', 'red'],
                    title=f'Gender Wage Gap by {selected_attr}'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error calculating fairness metrics: {str(e)}")


def display_econometrics(df: pd.DataFrame, labels: dict):
    """Display econometric analysis (from Notebook 05)"""
    st.header(":material/query_stats: Econometric Analysis")
    
    st.markdown("""
    <div class="insight-box">
        <strong>Methodology:</strong> Regression analysis controlling for human capital variables 
        (education, experience) and macroeconomic factors (unemployment, GDP growth, inflation) 
        to isolate the gender effect on wages.
    </div>
    """, unsafe_allow_html=True)
    
    # Macro data summary
    st.subheader("Macroeconomic Context")
    
    try:
        macro_df = get_macro_dataframe()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Display recent macro data
            display_cols = ['year', 'unemployment', 'gdp_growth', 'inflation']
            available_cols = [c for c in display_cols if c in macro_df.columns]
            st.dataframe(macro_df[available_cols].tail(10).round(2), use_container_width=True, hide_index=True)
        
        with col2:
            # Macro trends chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            if 'unemployment' in macro_df.columns:
                fig.add_trace(
                    go.Scatter(x=macro_df['year'], y=macro_df['unemployment'],
                               mode='lines', name='Unemployment (%)',
                               line=dict(color='orange')),
                    secondary_y=False
                )
            
            if 'gdp_growth' in macro_df.columns:
                fig.add_trace(
                    go.Scatter(x=macro_df['year'], y=macro_df['gdp_growth'],
                               mode='lines', name='GDP Growth (%)',
                               line=dict(color='green')),
                    secondary_y=True
                )
            
            fig.update_layout(title='Canadian Economic Indicators', height=350)
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Macro data not available: {str(e)}")
    
    # Regression summary
    st.subheader("Regression Analysis Summary")
    
    if 'GENDER' not in df.columns and 'SEX' not in df.columns:
        st.warning("Cannot perform regression without gender data")
        return
    
    # Simple regression display
    try:
        from scipy import stats as scipy_stats
        
        gender_col = 'SEX' if 'SEX' in df.columns else COLS.GENDER
        wage_col = 'HRLYEARN' if 'HRLYEARN' in df.columns else COLS.WAGE
        
        # Create female dummy
        df_reg = df.copy()
        df_reg['IS_FEMALE'] = (df_reg[gender_col] == 2).astype(int)
        
        # Model 1: Raw gender effect
        female_coef_raw = df_reg[df_reg['IS_FEMALE']==1][wage_col].mean() - df_reg[df_reg['IS_FEMALE']==0][wage_col].mean()
        
        # Model 2: With controls (simplified)
        control_vars = [c for c in ['EDUC', 'AGE_6', 'FTPTMAIN'] if c in df_reg.columns]
        
        results_data = [
            {'Model': 'Model 1: No Controls', 'Female Coefficient': female_coef_raw, 
             'Controls': 'None', 'Interpretation': f'Women earn ${female_coef_raw:.2f}/hr less than men'},
        ]
        
        if control_vars:
            # Simplified adjusted effect
            analyzer = PayEquityAnalyzer(df, wage_col=wage_col, gender_col=gender_col)
            adjusted = analyzer.compute_adjusted_wage_gap(control_vars)
            adj_gap = adjusted['adjusted_model']['gap_pct']
            adj_coef = adjusted['adjusted_model']['female_coefficient'] 
            
            results_data.append({
                'Model': 'Model 2: With Controls',
                'Female Coefficient': adj_coef,
                'Controls': ', '.join(control_vars),
                'Interpretation': f'Adjusted gap: ${adj_coef:.2f}/hr ({adj_gap:.1f}%)'
            })
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Visual comparison
        fig = go.Figure()
        for i, row in results_df.iterrows():
            fig.add_trace(go.Bar(
                name=row['Model'],
                x=[row['Model']],
                y=[abs(row['Female Coefficient'])],
                text=[f"${abs(row['Female Coefficient']):.2f}"],
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Gender Wage Penalty (Absolute Value)',
            yaxis_title='Wage Difference ($/hr)',
            showlegend=False,
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in regression analysis: {str(e)}")


def _get_canada_provinces_geojson():
    """
    Fetch or return a proper GeoJSON for Canadian provinces with actual boundaries.
    
    Uses a CDN-hosted Natural Earth dataset for accurate province shapes.
    Falls back to a simplified embedded version if network unavailable.
    """
    import json
    from urllib.request import urlopen
    from urllib.error import URLError
    
    # Try to fetch proper GeoJSON from a reliable CDN
    GEOJSON_URL = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson"
    
    try:
        with urlopen(GEOJSON_URL, timeout=5) as response:
            geojson_data = json.load(response)
            
        # Map the feature names to Statistics Canada PROV codes
        name_to_code = {
            "Newfoundland and Labrador": 10,
            "Prince Edward Island": 11,
            "Nova Scotia": 12,
            "New Brunswick": 13,
            "Quebec": 24,
            "Ontario": 35,
            "Manitoba": 46,
            "Saskatchewan": 47,
            "Alberta": 48,
            "British Columbia": 59,
        }
        
        # Update feature IDs to match PROV codes
        for feature in geojson_data['features']:
            name = feature['properties'].get('name') or feature['properties'].get('NAME')
            if name in name_to_code:
                feature['id'] = name_to_code[name]
                feature['properties']['prov_code'] = name_to_code[name]
        
        return geojson_data
        
    except (URLError, Exception):
        # Fallback: return a simplified but more accurate embedded GeoJSON
        # This is a very simplified version with rough province shapes
        return {
            "type": "FeatureCollection",
            "features": [
                # Atlantic provinces - smaller eastern boxes
                {"type": "Feature", "id": 10, "properties": {"name": "Newfoundland and Labrador"},
                 "geometry": {"type": "Polygon", "coordinates": [[[-60, 51], [-52, 51], [-52, 55], [-60, 55], [-60, 51]]]}},
                {"type": "Feature", "id": 11, "properties": {"name": "Prince Edward Island"},
                 "geometry": {"type": "Polygon", "coordinates": [[[-64, 45.5], [-62, 45.5], [-62, 47], [-64, 47], [-64, 45.5]]]}},
                {"type": "Feature", "id": 12, "properties": {"name": "Nova Scotia"},
                 "geometry": {"type": "Polygon", "coordinates": [[[-66, 43], [-60, 43], [-60, 47], [-66, 47], [-66, 43]]]}},
                {"type": "Feature", "id": 13, "properties": {"name": "New Brunswick"},
                 "geometry": {"type": "Polygon", "coordinates": [[[-69, 45], [-64, 45], [-64, 48], [-69, 48], [-69, 45]]]}},
                # Quebec - large eastern region
                {"type": "Feature", "id": 24, "properties": {"name": "Quebec"},
                 "geometry": {"type": "Polygon", "coordinates": [[[-79, 45], [-57, 45], [-57, 62], [-79, 62], [-79, 45]]]}},
                # Ontario - central large region
                {"type": "Feature", "id": 35, "properties": {"name": "Ontario"},
                 "geometry": {"type": "Polygon", "coordinates": [[[-95, 42], [-74, 42], [-74, 57], [-95, 57], [-95, 42]]]}},
                # Prairie provinces
                {"type": "Feature", "id": 46, "properties": {"name": "Manitoba"},
                 "geometry": {"type": "Polygon", "coordinates": [[[-102, 49], [-89, 49], [-89, 60], [-102, 60], [-102, 49]]]}},
                {"type": "Feature", "id": 47, "properties": {"name": "Saskatchewan"},
                 "geometry": {"type": "Polygon", "coordinates": [[[-110, 49], [-101, 49], [-101, 60], [-110, 60], [-110, 49]]]}},
                {"type": "Feature", "id": 48, "properties": {"name": "Alberta"},
                 "geometry": {"type": "Polygon", "coordinates": [[[-120, 49], [-110, 49], [-110, 60], [-120, 60], [-120, 49]]]}},
                # BC - large western region
                {"type": "Feature", "id": 59, "properties": {"name": "British Columbia"},
                 "geometry": {"type": "Polygon", "coordinates": [[[-139, 48], [-114, 48], [-114, 60], [-139, 60], [-139, 48]]]}},
            ]
        }


def display_geographic(df: pd.DataFrame, labels: dict):
    """Display geographic/provincial analysis with choropleth map."""
    st.header(":material/map: Geographic Analysis")

    if 'PROV' not in df.columns:
        st.warning("Provincial data not available")
        return

    if 'GENDER' not in df.columns and 'SEX' not in df.columns:
        st.warning("Gender data not available")
        return

    gender_col = 'GENDER' if 'GENDER' in df.columns else 'SEX'
    wage_col = 'HRLYEARN' if 'HRLYEARN' in df.columns else COLS.WAGE
    has_weights = 'FINALWT' in df.columns and df['FINALWT'].notna().any()

    # ── Compute provincial statistics ───────────────────────────────
    prov_data = []
    for prov in df['PROV'].unique():
        p = df[df['PROV'] == prov]
        males = p[p[gender_col] == 1]
        females = p[p[gender_col] == 2]

        if has_weights:
            m_wage = weighted_mean(males[wage_col], males['FINALWT'])
            f_wage = weighted_mean(females[wage_col], females['FINALWT'])
            avg_wage = weighted_mean(p[wage_col], p['FINALWT'])
        else:
            m_wage = males[wage_col].mean()
            f_wage = females[wage_col].mean()
            avg_wage = p[wage_col].mean()

        if pd.notna(m_wage) and pd.notna(f_wage) and m_wage > 0:
            prov_data.append({
                'Province Code': int(prov),
                'Province': labels['PROV'].get(prov, str(prov)),
                'Male Wage': m_wage,
                'Female Wage': f_wage,
                'Avg Wage': avg_wage,
                'Wage Gap %': ((m_wage - f_wage) / m_wage) * 100,
                'Sample Size': len(p),
            })

    prov_df = pd.DataFrame(prov_data).sort_values('Wage Gap %', ascending=False)
    national_gap = ((df[df[gender_col] == 1][wage_col].mean()
                     - df[df[gender_col] == 2][wage_col].mean())
                    / df[df[gender_col] == 1][wage_col].mean() * 100)

    # ── KPI row ─────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        w = prov_df.iloc[0]
        st.metric("Highest Gap", w['Province'], f"{w['Wage Gap %']:.1f}%")
    with col2:
        b = prov_df.iloc[-1]
        st.metric("Lowest Gap", b['Province'], f"{b['Wage Gap %']:.1f}%")
    with col3:
        st.metric("National Average Gap", f"{national_gap:.1f}%")

    # ── Map selector ────────────────────────────────────────────────
    map_metric = st.radio(
        "Map metric",
        ["Wage Gap %", "Avg Wage", "Male Wage", "Female Wage"],
        horizontal=True,
    )

    # ── Choropleth map ──────────────────────────────────────────────
    geojson = _get_canada_provinces_geojson()

    # Enhanced color scales for better aesthetics
    if map_metric == "Wage Gap %":
        color_scale = "RdYlGn_r"  # Red for high gap, green for low gap (reversed)
    elif map_metric == "Avg Wage":
        color_scale = "Viridis"
    else:
        color_scale = "Blues"

    fig = px.choropleth_map(
        prov_df,
        geojson=geojson,
        locations="Province Code",
        color=map_metric,
        hover_name="Province",
        hover_data={
            "Male Wage": ":.2f",
            "Female Wage": ":.2f",
            "Wage Gap %": ":.1f",
            "Sample Size": ":,",
            "Province Code": False,
        },
        color_continuous_scale=color_scale,
        map_style="carto-positron",
        center={"lat": 60, "lon": -95},
        zoom=2.8,
        opacity=0.8,
        title=f"{map_metric} by Province",
    )
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(
            title=dict(text=map_metric, font=dict(size=13)),
            thickness=20,
            len=0.7,
        ),
        font=dict(size=12),
    )
    # Enhance the map appearance
    fig.update_traces(
        marker_line_width=1.5,
        marker_line_color='white',
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Horizontal bar chart ────────────────────────────────────────
    st.subheader("Provincial Wage Gap Comparison")
    fig2 = px.bar(
        prov_df.sort_values('Wage Gap %'),
        y='Province', x='Wage Gap %',
        orientation='h',
        color='Wage Gap %',
        color_continuous_scale=['green', 'yellow', 'red'],
        text=prov_df.sort_values('Wage Gap %')['Wage Gap %'].apply(lambda v: f"{v:.1f}%"),
        title='Gender Wage Gap by Province',
    )
    fig2.add_vline(x=national_gap, line_dash="dash", line_color="black",
                   annotation_text=f"National Avg: {national_gap:.1f}%")
    fig2.update_traces(textposition='outside')
    fig2.update_layout(height=480, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Male vs Female scatter ──────────────────────────────────────
    st.subheader("Male vs Female Average Wage by Province")
    fig3 = px.scatter(
        prov_df,
        x='Male Wage', y='Female Wage',
        size='Sample Size',
        color='Wage Gap %',
        color_continuous_scale=['green', 'yellow', 'red'],
        hover_name='Province',
        text='Province',
        title='Provincial Wage Comparison (size = sample)',
    )
    # Add parity line
    max_w = max(prov_df['Male Wage'].max(), prov_df['Female Wage'].max()) * 1.05
    fig3.add_shape(type="line", x0=0, y0=0, x1=max_w, y1=max_w,
                   line=dict(color="grey", dash="dot"))
    fig3.update_traces(textposition='top center')
    fig3.update_layout(
        height=480,
        xaxis_title="Male Average Hourly Wage ($)",
        yaxis_title="Female Average Hourly Wage ($)",
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Regional aggregation ────────────────────────────────────────
    st.subheader("Regional Patterns")
    region_map = {
        10: 'Atlantic', 11: 'Atlantic', 12: 'Atlantic', 13: 'Atlantic',
        24: 'Quebec', 35: 'Ontario',
        46: 'Prairies', 47: 'Prairies', 48: 'Prairies',
        59: 'British Columbia',
    }
    prov_df['Region'] = prov_df['Province Code'].map(region_map)

    region_stats = prov_df.groupby('Region').agg(
        gap_mean=('Wage Gap %', 'mean'),
        male_mean=('Male Wage', 'mean'),
        female_mean=('Female Wage', 'mean'),
        n=('Sample Size', 'sum'),
    ).reset_index()

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(name='Male', x=region_stats['Region'], y=region_stats['male_mean'],
                          marker_color='#1f77b4', text=region_stats['male_mean'].apply(lambda v: f"${v:.2f}"),
                          textposition='auto'))
    fig4.add_trace(go.Bar(name='Female', x=region_stats['Region'], y=region_stats['female_mean'],
                          marker_color='#e377c2', text=region_stats['female_mean'].apply(lambda v: f"${v:.2f}"),
                          textposition='auto'))
    fig4.update_layout(barmode='group', title='Average Wage by Region and Gender',
                       yaxis_title='Average Hourly Wage ($)', height=420)
    st.plotly_chart(fig4, use_container_width=True)

    # ── Detailed table ──────────────────────────────────────────────
    st.subheader("Provincial Details")
    display_tbl = prov_df[['Province', 'Male Wage', 'Female Wage', 'Wage Gap %', 'Sample Size']].copy()
    display_tbl['Male Wage'] = display_tbl['Male Wage'].apply(lambda x: f"${x:.2f}")
    display_tbl['Female Wage'] = display_tbl['Female Wage'].apply(lambda x: f"${x:.2f}")
    display_tbl['Wage Gap %'] = display_tbl['Wage Gap %'].apply(lambda x: f"{x:.1f}%")
    st.dataframe(display_tbl, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
