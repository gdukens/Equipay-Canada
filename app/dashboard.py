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


# Page configuration
st.set_page_config(
    page_title="EquiPay Canada - Pay Equity Dashboard",
    page_icon="📊",
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
def load_data():
    """Load and cache data using DuckDB data store (memory-efficient)"""
    try:
        # Use DuckDB data store for memory-efficient data access
        store = EquiPayDataStore(memory_limit='3GB')
        
        # Load a sample for dashboard (full data is 19M+ rows)
        # For interactive dashboards, we use a representative sample
        # Include FINALWT for proper population-weighted statistics
        df = store.query("""
            SELECT * FROM lfs 
            WHERE HRLYEARN IS NOT NULL AND FINALWT > 0
            USING SAMPLE 500000 ROWS
        """)
        
        # Data already has GENDER column from the view
        # No aliasing needed - use GENDER consistently
        
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
    st.markdown('<p class="main-header">🇨🇦 EquiPay Canada</p>', unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #666;'>Pay Equity Analysis Dashboard</h3>", 
                unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading data...'):
        df = load_data()
        labels = get_labels()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # === Equity Dimension Selector (NEW) ===
    st.sidebar.subheader("📊 Primary Equity Dimension")
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Overview", 
        "⚖️ Wage Gap Analysis", 
        "🔬 Decomposition",
        "📊 Time Series",
        "🎯 Fairness",
        "📉 Econometrics",
        "🗺️ Geographic",
        "📋 Recommendations"
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
    
    # Tab 8: Recommendations
    with tab8:
        display_recommendations(df, labels)


def display_overview(df: pd.DataFrame, labels: dict, equity_dim: EquityDimension = None):
    """Display overview metrics with proper survey weighting"""
    st.header("📈 Overview Dashboard")
    
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
            pop_estimate = df['FINALWT'].sum()
            st.metric(
                label="Population Estimate",
                value=f"{pop_estimate:,.0f}",
                delta=None,
                help=f"Based on {len(df):,} sample observations"
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


def display_wage_gap_analysis(df: pd.DataFrame, labels: dict, equity_dim: EquityDimension = None):
    """Display generalized wage gap analysis for any equity dimension"""
    
    # Default to gender if no dimension specified
    if equity_dim is None:
        equity_dim = get_equity_dimension('gender')
    
    st.header(f"⚖️ {equity_dim.description} Analysis")
    
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
    st.header("⚖️ Gender Wage Gap Analysis")
    
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
    st.header("📊 Detailed Wage Analysis")
    
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
    """Display model predictions interface"""
    st.header("🎯 Salary Prediction Tool")
    
    st.markdown("""
    Use this tool to predict expected hourly wages based on worker characteristics.
    This can help identify potential pay inequities for specific demographic profiles.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Input fields
        gender = st.selectbox("Gender", ["Male", "Female"])
        
        education = st.selectbox("Education Level", [
            "Less than high school",
            "High school graduate",
            "Some college",
            "College diploma",
            "University certificate",
            "Bachelor's degree",
            "Graduate degree"
        ])
        
        occupation = st.selectbox("Occupation Category", [
            "Management",
            "Business/Finance",
            "Sciences",
            "Health",
            "Education/Law/Social",
            "Art/Culture/Recreation",
            "Sales/Service",
            "Trades/Transport",
            "Resources/Agriculture",
            "Manufacturing"
        ])
    
    with col2:
        age = st.slider("Age", 15, 65, 35)
        
        employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time"])
        
        union_status = st.selectbox("Union Status", [
            "Union member",
            "Covered by union",
            "Not unionized"
        ])
    
    if st.button("Predict Salary", type="primary"):
        # Simple prediction based on averages (placeholder for actual model)
        # In production, this would load the trained model
        
        # Filter data for similar profile
        filters = []
        
        if 'GENDER' in df.columns:
            sex_code = 1 if gender == "Male" else 2
            filters.append(df['GENDER'] == sex_code)
        
        if 'FTPTMAIN' in df.columns:
            ft_code = 1 if employment_type == "Full-time" else 2
            filters.append(df['FTPTMAIN'] == ft_code)
        
        if filters:
            mask = filters[0]
            for f in filters[1:]:
                mask = mask & f
            similar_df = df[mask]
        else:
            similar_df = df
        
        if len(similar_df) > 0:
            predicted_wage = similar_df['HRLYEARN'].mean()
            wage_std = similar_df['HRLYEARN'].std()
            lower = max(predicted_wage - 1.96 * wage_std / np.sqrt(len(similar_df)), 
                       similar_df['HRLYEARN'].quantile(0.1))
            upper = predicted_wage + 1.96 * wage_std / np.sqrt(len(similar_df))
            
            st.success(f"### Predicted Hourly Wage: ${predicted_wage:.2f}")
            st.info(f"95% Confidence Interval: ${lower:.2f} - ${upper:.2f}")
            
            # Compare to overall
            overall_mean = df['HRLYEARN'].mean()
            diff = predicted_wage - overall_mean
            diff_pct = (diff / overall_mean) * 100
            
            if diff > 0:
                st.markdown(f"This is **${diff:.2f}/hr ({diff_pct:+.1f}%)** above the overall average.")
            else:
                st.markdown(f"This is **${abs(diff):.2f}/hr ({diff_pct:.1f}%)** below the overall average.")
        else:
            st.warning("Not enough data points for this profile.")


def display_recommendations(df: pd.DataFrame, labels: dict):
    """Display recommendations based on analysis"""
    st.header("📋 Recommendations")
    
    # Calculate key metrics for recommendations
    if 'GENDER' in df.columns:
        male_wage = df[df['GENDER'] == 1]['HRLYEARN'].mean()
        female_wage = df[df['GENDER'] == 2]['HRLYEARN'].mean()
        gap_pct = ((male_wage - female_wage) / male_wage) * 100
    else:
        gap_pct = 0
    
    st.markdown("""
    <div class="insight-box">
        <h3>🎯 Key Findings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    findings = []
    
    if gap_pct > 15:
        findings.append({
            'severity': 'high',
            'finding': f'Significant gender wage gap detected ({gap_pct:.1f}%)',
            'recommendation': 'Conduct comprehensive pay equity audit and develop remediation plan'
        })
    elif gap_pct > 10:
        findings.append({
            'severity': 'medium',
            'finding': f'Moderate gender wage gap detected ({gap_pct:.1f}%)',
            'recommendation': 'Review compensation practices and identify root causes'
        })
    else:
        findings.append({
            'severity': 'low',
            'finding': f'Gender wage gap is relatively small ({gap_pct:.1f}%)',
            'recommendation': 'Continue monitoring and maintain equitable practices'
        })
    
    # Display findings
    for finding in findings:
        if finding['severity'] == 'high':
            emoji = "🔴"
        elif finding['severity'] == 'medium':
            emoji = "🟡"
        else:
            emoji = "🟢"
        
        st.markdown(f"""
        {emoji} **{finding['finding']}**
        
        *Recommendation:* {finding['recommendation']}
        """)
    
    st.markdown("---")
    
    st.subheader("📌 Recommended Actions")
    
    actions = [
        "**1. Pay Equity Audit**: Conduct regular audits to identify and address pay disparities across all demographic groups.",
        "**2. Transparent Pay Bands**: Implement clear salary ranges for each role to ensure consistency in compensation decisions.",
        "**3. Bias Training**: Provide training for hiring managers and HR on unconscious bias in compensation decisions.",
        "**4. Regular Reviews**: Schedule annual compensation reviews to ensure pay equity is maintained over time.",
        "**5. Data Collection**: Improve demographic data collection to enable more granular analysis of pay equity.",
        "**6. Policy Updates**: Review and update compensation policies to align with pay equity legislation.",
    ]
    
    for action in actions:
        st.markdown(action)
    
    st.markdown("---")
    
    # Compliance note
    st.markdown("""
    <div class="warning-box">
        <h4>⚖️ Canadian Pay Equity Legislation</h4>
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
        store = EquiPayDataStore(memory_limit='3GB')
        # Aggregate yearly wage data by gender directly in DuckDB
        df = store.query("""
            SELECT 
                year,
                GENDER as SEX,
                AVG(HRLYEARN) as avg_wage,
                MEDIAN(HRLYEARN) as median_wage,
                COUNT(*) as n
            FROM lfs
            WHERE HRLYEARN IS NOT NULL
            GROUP BY year, GENDER
            ORDER BY year, GENDER
        """)
        return df
    except Exception as e:
        # Fallback to CSV if available
        ts_path = Path("data/processed/yearly_wages_by_gender.csv")
        if ts_path.exists():
            return pd.read_csv(ts_path)
        return None


def display_decomposition(df: pd.DataFrame, labels: dict, equity_dim: EquityDimension = None):
    """Display Oaxaca-Blinder decomposition analysis (from Notebook 03)"""
    
    # Default to gender if no dimension specified
    if equity_dim is None:
        equity_dim = get_equity_dimension('gender')
    
    st.header(f"🔬 Oaxaca-Blinder Decomposition: {equity_dim.description}")
    
    st.markdown(f"""
    <div class="insight-box">
        <strong>Methodology:</strong> The Oaxaca-Blinder decomposition separates the wage gap into:
        <ul>
            <li><strong>Explained</strong>: Differences due to observable characteristics (education, experience, occupation)</li>
            <li><strong>Unexplained</strong>: Residual gap that may indicate discrimination or unobserved factors</li>
        </ul>
        <p><em>Analyzing: {equity_dim.reference_label} vs {equity_dim.comparison_label}</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    group_col = equity_dim.column
    
    # Support both SEX and GENDER columns for backward compatibility
    if group_col not in df.columns:
        if group_col == 'GENDER' and 'SEX' in df.columns:
            group_col = 'SEX'
        else:
            st.warning(f"{equity_dim.column} data not available for decomposition")
            return
    
    # Perform simplified decomposition
    try:
        # For gender dimension, use the existing PayEquityAnalyzer
        if equity_dim.column in ['GENDER', 'SEX']:
            analyzer = PayEquityAnalyzer(df, wage_col='HRLYEARN', gender_col=group_col)
            raw_gap = analyzer.compute_raw_gap()
            gap_pct = raw_gap['raw_gap']['mean_gap_pct']
        else:
            # For other dimensions, calculate manually
            ref_df = df[df[group_col] == equity_dim.reference_value]
            comp_df = df[df[group_col] == equity_dim.comparison_value]
            ref_mean = ref_df['HRLYEARN'].mean()
            comp_mean = comp_df['HRLYEARN'].mean()
            gap_pct = ((ref_mean - comp_mean) / ref_mean) * 100 if ref_mean > 0 else 0
        
        # Estimate explained vs unexplained (simplified)
        # In full analysis, this uses regression-based decomposition
        control_vars = [c for c in ['EDUC', 'AGE_6', 'NOC_10', 'PROV', 'FTPTMAIN'] if c in df.columns]
        
        if control_vars:
            adjusted = analyzer.compute_adjusted_gap(control_vars)
            adjusted_gap = adjusted['adjusted_model']['gap_pct']
            
            explained_pct = gap_pct - adjusted_gap
            unexplained_pct = adjusted_gap
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Wage Gap", f"{gap_pct:.1f}%")
            with col2:
                st.metric("Explained by Characteristics", f"{explained_pct:.1f}%", 
                         delta=f"{(explained_pct/gap_pct*100):.0f}% of total" if gap_pct > 0 else None)
            with col3:
                st.metric("Unexplained (Potential Bias)", f"{unexplained_pct:.1f}%",
                         delta=f"{(unexplained_pct/gap_pct*100):.0f}% of total" if gap_pct > 0 else None,
                         delta_color="inverse")
            
            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['Explained (Characteristics)', 'Unexplained (Potential Bias)'],
                values=[max(0, explained_pct), max(0, unexplained_pct)],
                hole=0.4,
                marker_colors=['#2ca02c', '#d62728']
            )])
            fig.update_layout(
                title=f"Wage Gap Decomposition (Total: {gap_pct:.1f}%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            if unexplained_pct > explained_pct:
                st.markdown("""
                <div class="warning-box">
                    <strong>⚠️ Finding:</strong> The majority of the wage gap ({:.1f}%) cannot be explained 
                    by observable characteristics. This suggests potential systemic bias or unmeasured factors.
                </div>
                """.format(unexplained_pct), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="insight-box">
                    <strong>Finding:</strong> Most of the wage gap ({:.1f}%) is explained by differences 
                    in education, occupation, and experience. However, the unexplained portion ({:.1f}%) 
                    still warrants attention.
                </div>
                """.format(explained_pct, unexplained_pct), unsafe_allow_html=True)
        else:
            st.warning("Insufficient control variables for decomposition analysis")
            
    except Exception as e:
        st.error(f"Error performing decomposition: {str(e)}")


def display_time_series(equity_dim: EquityDimension = None):
    """Display time series analysis (from Notebook 06)"""
    
    # Default to gender if no dimension specified
    if equity_dim is None:
        equity_dim = get_equity_dimension('gender')
    
    st.header(f"📊 Time Series Analysis: {equity_dim.description}")
    
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
        ts_data = load_time_series_data()
    
        if ts_data is None or len(ts_data) == 0:
            st.warning("Time series data not available. Run the data pipeline first.")
            return
    
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
        st.success(f"📉 **Trend:** The wage gap is narrowing at approximately {abs(z[0]):.2f} percentage points per year.")
    else:
        st.warning(f"📈 **Trend:** The wage gap is widening at approximately {z[0]:.2f} percentage points per year.")


def display_fairness_metrics(df: pd.DataFrame, labels: dict):
    """Display fairness metrics analysis (from Notebook 04)"""
    st.header("🎯 Fairness & Bias Metrics")
    
    st.markdown("""
    <div class="insight-box">
        <strong>Fairness Metrics:</strong> These metrics quantify potential discrimination 
        across protected attributes, aligned with algorithmic fairness standards and 
        Canadian human rights legislation.
    </div>
    """, unsafe_allow_html=True)
    
    if 'SEX' not in df.columns:
        st.warning("Gender data not available for fairness analysis")
        return
    
    try:
        # Use actual column names from the dataframe
        gender_col = 'SEX' if 'SEX' in df.columns else COLS.GENDER
        wage_col = 'HRLYEARN' if 'HRLYEARN' in df.columns else COLS.WAGE
        
        analyzer = FairnessAnalyzer(
            data=df,
            target_col=wage_col,
            protected_col=gender_col
        )
        
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
                st.error(f"⚠️ **Fails 4/5ths Rule**: Ratio of {di_ratio:.2f} is below 0.80 threshold")
            elif di_ratio < 0.9:
                st.warning(f"⚡ **Marginal**: Ratio of {di_ratio:.2f} passes 4/5ths rule but gap exists")
            else:
                st.success(f"✅ **Passes**: Ratio of {di_ratio:.2f} indicates near-parity")
        
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
    st.header("📉 Econometric Analysis")
    
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
    
    if 'SEX' not in df.columns:
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
            adjusted = analyzer.compute_adjusted_gap(control_vars)
            adj_gap = adjusted['adjusted_model']['gap_pct']
            adj_coef = adjusted['adjusted_model']['adjusted_gap'] 
            
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


def display_geographic(df: pd.DataFrame, labels: dict):
    """Display geographic/provincial analysis (from Notebook 08)"""
    st.header("🗺️ Geographic Analysis")
    
    if 'PROV' not in df.columns:
        st.warning("Provincial data not available")
        return
    
    if 'SEX' not in df.columns:
        st.warning("Gender data not available")
        return
    
    gender_col = 'SEX'
    wage_col = 'HRLYEARN' if 'HRLYEARN' in df.columns else COLS.WAGE
    
    # Calculate gap by province
    prov_data = []
    for prov in df['PROV'].unique():
        prov_df = df[df['PROV'] == prov]
        
        male_wage = prov_df[prov_df[gender_col] == 1][wage_col].mean()
        female_wage = prov_df[prov_df[gender_col] == 2][wage_col].mean()
        
        if pd.notna(male_wage) and pd.notna(female_wage) and male_wage > 0:
            prov_data.append({
                'Province Code': prov,
                'Province': labels['PROV'].get(prov, str(prov)),
                'Male Wage': male_wage,
                'Female Wage': female_wage,
                'Wage Gap %': ((male_wage - female_wage) / male_wage) * 100,
                'Sample Size': len(prov_df)
            })
    
    prov_df = pd.DataFrame(prov_data).sort_values('Wage Gap %', ascending=False)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        worst_prov = prov_df.iloc[0]
        st.metric(
            "Highest Gap Province",
            worst_prov['Province'],
            f"{worst_prov['Wage Gap %']:.1f}%"
        )
    
    with col2:
        best_prov = prov_df.iloc[-1]
        st.metric(
            "Lowest Gap Province",
            best_prov['Province'],
            f"{best_prov['Wage Gap %']:.1f}%"
        )
    
    with col3:
        national_gap = ((df[df[gender_col]==1][wage_col].mean() - 
                        df[df[gender_col]==2][wage_col].mean()) / 
                       df[df[gender_col]==1][wage_col].mean() * 100)
        st.metric("National Average Gap", f"{national_gap:.1f}%")
    
    # Provincial comparison chart
    fig = px.bar(
        prov_df.sort_values('Wage Gap %'),
        y='Province',
        x='Wage Gap %',
        orientation='h',
        color='Wage Gap %',
        color_continuous_scale=['green', 'yellow', 'red'],
        title='Gender Wage Gap by Province'
    )
    fig.add_vline(x=national_gap, line_dash="dash", line_color="black",
                  annotation_text=f"National Avg: {national_gap:.1f}%")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.subheader("Provincial Details")
    display_df = prov_df[['Province', 'Male Wage', 'Female Wage', 'Wage Gap %', 'Sample Size']].copy()
    display_df['Male Wage'] = display_df['Male Wage'].apply(lambda x: f"${x:.2f}")
    display_df['Female Wage'] = display_df['Female Wage'].apply(lambda x: f"${x:.2f}")
    display_df['Wage Gap %'] = display_df['Wage Gap %'].apply(lambda x: f"{x:.1f}%")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Map visualization (simplified bar chart by region)
    st.subheader("Regional Patterns")
    
    # Group by region
    region_map = {
        10: 'Atlantic', 11: 'Atlantic', 12: 'Atlantic', 13: 'Atlantic',
        24: 'Quebec', 35: 'Ontario',
        46: 'Prairies', 47: 'Prairies', 48: 'Prairies',
        59: 'British Columbia'
    }
    
    prov_df['Region'] = prov_df['Province Code'].map(region_map)
    region_stats = prov_df.groupby('Region').agg({
        'Wage Gap %': 'mean',
        'Sample Size': 'sum'
    }).reset_index()
    
    fig = px.bar(
        region_stats.sort_values('Wage Gap %'),
        x='Region',
        y='Wage Gap %',
        color='Wage Gap %',
        color_continuous_scale=['green', 'yellow', 'red'],
        title='Average Wage Gap by Region'
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
