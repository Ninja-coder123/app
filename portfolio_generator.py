"""
Portfolio Generator - Dual-Mode Streamlit Application
View 1: Interactive Tickertape-style MF Screener with custom PR-FQS scoring
View 2: Automated Portfolio Generator with psychometric assessment
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import altair as alt

# Import from backend modules
from backend.portfolio_engine import generate_portfolio, RISK_ASSESSMENT_QUESTIONS, RISK_TIERS

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# Path to offline database
DATA_PATH = Path(__file__).parent.parent / "data" / "screener_database.parquet"

# Custom PR-FQS Weight Categories
PR_FQS_METRICS = {
    "sharpe": "Sharpe Ratio",
    "sortino": "Sortino Ratio",
    "three_year_return": "3-Year Rolling Return",
    "volatility": "Volatility",
    "consistency": "Consistency Score",
    "beta_fit": "Beta Fit Score",
}

# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class PortfolioAllocation:
    """Individual portfolio fund allocation."""

    scheme_code: str
    fund_name: str
    category: str
    sip_amount: float
    pr_fqs_score: float
    expected_return: float


@dataclass
class GeneratedPortfolio:
    """Complete generated portfolio."""

    client_profile: Dict
    recommended_portfolio: List[PortfolioAllocation]
    total_sip: float
    risk_tier: str
    allocation_breakdown: Dict[str, float]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================



def calculate_risk_score(assessment_answers: Dict[int, int]) -> int:
    """Calculate total risk score from psychometric assessment."""
    return sum(assessment_answers.values())


def get_risk_tier(risk_score: int) -> str:
    """Map risk score to SEBI-compliant risk tier."""
    for tier_name, tier_data in RISK_TIERS.items():
        min_score, max_score = tier_data["range"]
        if min_score <= risk_score <= max_score:
            return tier_name
    return "Moderate"


# ============================================================================
# DATA LOADING & CACHING
# ============================================================================


@st.cache_data
def load_screener_database() -> pd.DataFrame:
    """
    Load screener database from Parquet file and cache it.
    This runs only once per session, dramatically improving responsiveness.
    """
    try:
        df = pd.read_parquet(DATA_PATH)
        
        # Ensure required columns exist
        required_cols = ['fund_code', 'scheme_name', 'category', 'pr_fqs_score', 'sharpe', 'sortino', 'rolling_return_3y', 'volatility', 'consistency_score', 'beta_fit']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            st.warning(f"Note: Some columns not found in database: {missing}. Using available data.")
            # Fill missing columns with defaults
            for col in missing:
                if col not in df.columns:
                    df[col] = 0.0
        
        st.session_state.database_loaded = True
        return df
    except FileNotFoundError:
        st.error(f"❌ Screener database not found at {DATA_PATH}. Please run `python scripts/database_builder.py` first.")
        return pd.DataFrame()


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize custom weights so they sum to 1.0 (100%).
    This ensures the custom PR-FQS scoring is always properly weighted.
    """
    total = sum(weights.values())
    if total == 0:
        # Default equal weights if all are zero
        return {key: 1.0 / len(weights) for key in weights}
    return {key: value / total for key, value in weights.items()}


def calculate_custom_pr_fqs(row: pd.Series, weights: Dict[str, float]) -> float:
    """
    Calculate custom PR-FQS score as weighted sum of metrics.
    Formula: Custom_Score = sum(normalized_weight_i * normalized_metric_i)
    """
    custom_score = 0.0
    
    # Normalize weights
    normalized_weights = normalize_weights(weights)
    
    # Extract metrics from row and normalize to 0-100 scale
    metrics = {
        "sharpe": min(float(row.get("sharpe", 0)) * 10, 100),  # Scale Sharpe to 0-100
        "sortino": min(float(row.get("sortino", 0)) * 10, 100),  # Scale Sortino to 0-100
        "three_year_return": min(float(row.get("rolling_return_3y", 0)) * 2, 100),  # Normalize 3Y return
        "volatility": max(100 - float(row.get("volatility", 0)) * 2, 0),  # Inverse (lower vol = better)
        "consistency": float(row.get("consistency_score", 0)),
        "beta_fit": float(row.get("beta_fit", 0)),
    }
    
    # Calculate weighted score
    for key, weight in normalized_weights.items():
        if key in metrics:
            custom_score += weight * metrics[key]
    
    return min(custom_score, 100.0)  # Cap at 100


# ============================================================================
# VIEW 1: CUSTOM MF SCREENER
# ============================================================================


def render_mf_screener():
    """Render the Tickertape-style MF Screener view."""
    
    st.subheader("🔍 Custom Mutual Fund Screener")
    st.markdown("Filter and rank funds using custom PR-FQS weights")
    
    # Load database
    df_master = load_screener_database()
    
    if df_master.empty:
        st.error("Database not loaded. Please check the data file.")
        return
    
    # ====== SIDEBAR CONTROLS ======
    st.sidebar.markdown("### 📊 Custom PR-FQS Weights")
    st.sidebar.info("Adjust weights below (they will auto-normalize to 100%)")
    
    # Collect custom weights from user
    weights = {}
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        weights["sharpe"] = st.slider("Sharpe Ratio (%)", 0, 100, 20, key="w_sharpe")
        weights["three_year_return"] = st.slider("3Y Return (%)", 0, 100, 30, key="w_3y")
        weights["consistency"] = st.slider("Consistency (%)", 0, 100, 20, key="w_cons")
    
    with col2:
        weights["sortino"] = st.slider("Sortino Ratio (%)", 0, 100, 20, key="w_sortino")
        weights["volatility"] = st.slider("Volatility (%)", 0, 100, 20, key="w_vol")
        weights["beta_fit"] = st.slider("Beta Fit (%)", 0, 100, 10, key="w_beta")
    
    # Display normalized weights
    normalized = normalize_weights(weights)
    st.sidebar.markdown("### ✅ Normalized Weights (Auto-Adjusted)")
    for metric_key, metric_name in PR_FQS_METRICS.items():
        pct = normalized.get(metric_key, 0) * 100
        st.sidebar.metric(metric_name, f"{pct:.1f}%")
    
    # ====== FILTER CONTROLS ======
    st.sidebar.markdown("### 🔎 Filter Controls")
    
    # Multi-select for SEBI categories
    available_categories = sorted(df_master['category'].unique().tolist())
    selected_categories = st.sidebar.multiselect(
        "SEBI Categories",
        available_categories,
        default=available_categories[:5],
        key="category_select"
    )
    
    # Text search for scheme name
    search_text = st.sidebar.text_input(
        "Search Scheme Name",
        placeholder="e.g., Mirae, HDFC, SBI",
        key="scheme_search"
    )
    
    # ====== DATA FILTERING & CALCULATION ======
    # Apply category filter
    df_filtered = df_master[df_master['category'].isin(selected_categories)].copy()
    
    # Apply text search filter
    if search_text.strip():
        df_filtered = df_filtered[
            df_filtered['scheme_name'].str.contains(search_text, case=False, na=False)
        ]
    
    if df_filtered.empty:
        st.warning("No funds match the selected filters.")
        return
    
    # Calculate custom PR-FQS score for each fund
    df_filtered['custom_pr_fqs'] = df_filtered.apply(
        lambda row: calculate_custom_pr_fqs(row, weights),
        axis=1
    )
    
    # Sort by custom PR-FQS descending and add rank
    df_filtered = df_filtered.sort_values('custom_pr_fqs', ascending=False).reset_index(drop=True)
    df_filtered['rank'] = range(1, len(df_filtered) + 1)
    
    # ====== DISPLAY DATA GRID ======
    st.markdown("### 📈 Fund Rankings")
    
    # Create display dataframe with selected columns
    display_cols = ['rank', 'scheme_name', 'category', 'custom_pr_fqs', 'sharpe', 'sortino', 'rolling_return_3y', 'volatility']
    df_display = df_filtered[display_cols].copy()
    
    # Rename columns for display
    df_display.columns = ['Rank', 'Scheme Name', 'Category', 'Custom PR-FQS', 'Sharpe', 'Sortino', '3Y Return %', 'Volatility %']
    
    # Format numeric columns
    numeric_cols = ['Custom PR-FQS', 'Sharpe', 'Sortino', '3Y Return %', 'Volatility %']
    for col in numeric_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}")
    
    # Display as full-width sortable dataframe
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        height=600
    )
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Funds", len(df_filtered))
    with col2:
        st.metric("Categories", df_filtered['category'].nunique())
    with col3:
        top_score = df_filtered['custom_pr_fqs'].max()
        st.metric("Top Custom Score", f"{top_score:.2f}")
    with col4:
        avg_score = df_filtered['custom_pr_fqs'].mean()
        st.metric("Average Score", f"{avg_score:.2f}")


# ============================================================================
# VIEW 2: PORTFOLIO GENERATOR
# ============================================================================


def render_psychometric_assessment():
    """Render the psychometric risk assessment section."""
    st.subheader("📋 Psychometric Risk Assessment")
    st.markdown("Answer 7 questions to determine your risk profile")
    
    answers = {}
    
    for question_data in RISK_ASSESSMENT_QUESTIONS:
        q_id = question_data["id"]
        question_text = question_data["question"]
        options = question_data["options"]
        
        st.markdown(f"**Question {q_id}: {question_text}**")
        selected = st.radio(
            "Select your answer:",
            options,
            format_func=lambda x: x[0],
            key=f"q_{q_id}",
            label_visibility="collapsed"
        )
        answers[q_id] = selected[1]
        st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Generate Portfolio", key="generate_btn", use_container_width=True):
            st.session_state.assessment_answers = answers
            st.session_state.assessment_complete = True
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset Assessment", key="reset_btn", use_container_width=True):
            st.session_state.assessment_complete = False
            st.session_state.generated_portfolio = None
            st.session_state.assessment_answers = {}
            st.rerun()


def render_portfolio_results():
    """Render the portfolio generation and results section."""
    st.subheader("💰 Portfolio Generation & Results")
    
    # Input for monthly SIP
    col1, col2, col3 = st.columns(3)
    with col1:
        monthly_sip = st.number_input(
            "Target Monthly SIP Amount (₹)",
            min_value=5000,
            max_value=10000000,
            step=5000,
            value=50000,
            key="sip_input"
        )
    
    with col2:
        risk_score = calculate_risk_score(st.session_state.assessment_answers)
        st.metric("Your Risk Score", f"{risk_score} / 70")
    
    with col3:
        risk_tier = get_risk_tier(risk_score)
        st.metric("Your Risk Tier", risk_tier)
    
    st.divider()
    
    # Generate portfolio button
    if st.button("📊 Generate Optimized Portfolio", use_container_width=True, key="portfolio_gen"):
        with st.spinner("Generating your portfolio..."):
            try:
                # Call synchronous backend function
                portfolio_obj = generate_portfolio(st.session_state.assessment_answers, monthly_sip)
                
                # Convert to our dataclass for consistency
                portfolio = GeneratedPortfolio(
                    client_profile=portfolio_obj.client_profile,
                    recommended_portfolio=[
                        PortfolioAllocation(
                            scheme_code=alloc.scheme_code,
                            fund_name=alloc.fund_name,
                            category=alloc.category,
                            sip_amount=alloc.sip_amount,
                            pr_fqs_score=alloc.pr_fqs_score,
                            expected_return=alloc.expected_return
                        )
                        for alloc in portfolio_obj.recommended_portfolio
                    ],
                    total_sip=portfolio_obj.total_sip,
                    risk_tier=portfolio_obj.risk_tier,
                    allocation_breakdown=portfolio_obj.allocation_breakdown
                )
                
                st.session_state.generated_portfolio = portfolio
                st.success("✅ Portfolio generated successfully!")
                st.rerun()
                
            except FileNotFoundError as e:
                st.error(f"❌ Database not found: {e}")
                st.info("Please run `python scripts/database_builder.py` to build the database.")
            except Exception as e:
                st.error(f"❌ Error generating portfolio: {str(e)}")
    
    # Display results if portfolio has been generated
    if st.session_state.generated_portfolio:
        portfolio = st.session_state.generated_portfolio
        
        st.markdown("### 📊 Portfolio Allocation Breakdown")
        
        col_chart, col_metrics = st.columns([2, 1])
        
        with col_chart:
            # Create donut chart
            allocation_data = pd.DataFrame([
                {"Asset Class": "Equity", "Percentage": portfolio.allocation_breakdown["equity"]},
                {"Asset Class": "Debt", "Percentage": portfolio.allocation_breakdown["debt"]},
            ])
            
            chart = (
                alt.Chart(allocation_data)
                .mark_arc(innerRadius=60)
                .encode(
                    theta="Percentage",
                    color=alt.Color(
                        "Asset Class:N",
                        scale=alt.Scale(
                            domain=["Equity", "Debt"],
                            range=["#1f77b4", "#ff7f0e"]
                        ),
                    ),
                    tooltip=["Asset Class", "Percentage"],
                )
                .properties(width=400, height=400, title="Asset Allocation")
            )
            st.altair_chart(chart, use_container_width=True)
        
        with col_metrics:
            st.metric("Total Monthly SIP", f"₹{portfolio.total_sip:,.2f}")
            equity_amt = portfolio.total_sip * portfolio.allocation_breakdown["equity"] / 100
            debt_amt = portfolio.total_sip * portfolio.allocation_breakdown["debt"] / 100
            st.metric("Equity Allocation", f"₹{equity_amt:,.2f}")
            st.metric("Debt Allocation", f"₹{debt_amt:,.2f}")
        
        st.divider()
        
        # Fund recommendations table
        st.markdown("### 🎯 Fund Recommendations")
        
        fund_data = []
        for alloc in portfolio.recommended_portfolio:
            fund_data.append({
                "Fund Name": alloc.fund_name,
                "Category": alloc.category,
                "Monthly SIP (₹)": f"{alloc.sip_amount:,.2f}",
                "Custom PR-FQS": f"{alloc.pr_fqs_score:.2f}",
                "Expected Annual Return": f"{alloc.expected_return * 100:.2f}%",
            })
        
        df_funds = pd.DataFrame(fund_data)
        st.dataframe(df_funds, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # JSON payload
        st.markdown("### 📋 Portfolio Execution JSON")
        
        portfolio_json = {
            "client_profile": portfolio.client_profile,
            "recommended_portfolio": [
                {
                    "scheme_code": alloc.scheme_code,
                    "fund_name": alloc.fund_name,
                    "category": alloc.category,
                    "sip_amount": round(alloc.sip_amount, 2),
                    "custom_pr_fqs": round(alloc.pr_fqs_score, 2),
                    "expected_return_annual": round(alloc.expected_return, 4),
                }
                for alloc in portfolio.recommended_portfolio
            ],
            "total_sip_amount": round(portfolio.total_sip, 2),
            "risk_tier": portfolio.risk_tier,
            "allocation_breakdown": portfolio.allocation_breakdown,
        }
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.json(portfolio_json)
        
        with col2:
            json_str = json.dumps(portfolio_json, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )


def render_portfolio_generator():
    """Render the portfolio generator view."""
    
    if not st.session_state.assessment_complete:
        render_psychometric_assessment()
    else:
        render_portfolio_results()


# ============================================================================
# MAIN APPLICATION
# ============================================================================


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "assessment_complete" not in st.session_state:
        st.session_state.assessment_complete = False
    if "generated_portfolio" not in st.session_state:
        st.session_state.generated_portfolio = None
    if "assessment_answers" not in st.session_state:
        st.session_state.assessment_answers = {}
    if "database_loaded" not in st.session_state:
        st.session_state.database_loaded = False


def main():
    """Main Streamlit application with dual-mode interface."""
    
    # Page configuration
    st.set_page_config(
        page_title="Portfolio Generator",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Header
    st.title("🏦 Wealth Management Hub")
    st.markdown(
        """
        **Dual-Mode Mutual Fund Platform**
        
        Explore funds with our custom screener or get a personalized portfolio based on your risk profile.
        """
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About This App")
        st.info(
            """
            **Two Views:**
            1. **Custom MF Screener** - Filter & rank funds with custom PR-FQS weights
            2. **Portfolio Generator** - Get a personalized portfolio from psychometric assessment
            
            **Features:**
            - ⚡ Instant data loading (cached in RAM)
            - 📊 Real-time custom scoring
            - 📋 SEBI-compliant allocations
            - 📥 Download portfolio JSON
            """
        )
    
    # Tab interface
    tab1, tab2 = st.tabs(["🔍 Custom MF Screener", "🏦 Automated Portfolio Generator"])
    
    with tab1:
        render_mf_screener()
    
    with tab2:
        render_portfolio_generator()


if __name__ == "__main__":
    main()
