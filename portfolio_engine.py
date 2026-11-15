"""
Portfolio Engine - Deterministic Backend for Portfolio Generation
Handles all financial math and offline fund selection from Parquet database.
NO STREAMLIT CALLS in this module.
NO NETWORK CALLS - Pure offline execution using data/screener_database.parquet
"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

RISK_FREE_RATE = 0.06  # 6% annual risk-free rate
SCREENER_DATABASE_PATH = Path(__file__).parent.parent / "data" / "screener_database.parquet"
RISK_ASSESSMENT_QUESTIONS = [
    {
        "id": 1,
        "question": "What is your investment time horizon?",
        "options": [
            ("Less than 2 years", 1),
            ("2-5 years", 3),
            ("5-10 years", 6),
            ("10-20 years", 8),
            ("20+ years", 10),
        ],
    },
    {
        "id": 2,
        "question": "How stable is your income?",
        "options": [
            ("Very unstable/volatile", 1),
            ("Somewhat unstable", 3),
            ("Moderate stability", 6),
            ("Quite stable", 8),
            ("Very stable/guaranteed", 10),
        ],
    },
    {
        "id": 3,
        "question": "How do you react to market volatility?",
        "options": [
            ("Panic and sell", 1),
            ("Worried, may sell", 3),
            ("Neutral, hold", 6),
            ("Comfortable", 8),
            ("Eager to invest more", 10),
        ],
    },
    {
        "id": 4,
        "question": "What is your primary investment goal?",
        "options": [
            ("Capital preservation", 1),
            ("Income generation", 4),
            ("Balanced growth", 6),
            ("Long-term wealth", 8),
            ("Aggressive growth", 10),
        ],
    },
    {
        "id": 5,
        "question": "Can you tolerate a portfolio decline of >20% annually?",
        "options": [
            ("No, unacceptable", 1),
            ("Uncomfortable", 3),
            ("Moderately okay", 6),
            ("Acceptable", 8),
            ("Highly acceptable", 10),
        ],
    },
    {
        "id": 6,
        "question": "What is your monthly investable surplus?",
        "options": [
            ("₹10,000 or less", 2),
            ("₹10,000 - ₹50,000", 4),
            ("₹50,000 - ₹1,00,000", 6),
            ("₹1,00,000 - ₹5,00,000", 8),
            ("₹5,00,000+", 10),
        ],
    },
    {
        "id": 7,
        "question": "How much prior experience do you have with equity markets?",
        "options": [
            ("None", 1),
            ("Minimal", 3),
            ("Moderate", 6),
            ("Considerable", 8),
            ("Expert", 10),
        ],
    },
]
# Risk tier definitions (SEBI-compliant allocation matrix)
RISK_TIERS = {
    "Ultra-Conservative": {
        "range": (7, 14),
        "allocation": {"equity": 15, "debt": 85},
        "categories": [
            "Arbitrage Fund",
            "Liquid Fund",
        ],
    },
    "Conservative": {
        "range": (15, 28),
        "allocation": {"equity": 30, "debt": 70},
        "categories": [
            "Balanced Advantage Fund",
            "Multi Asset Fund",
            "Short Duration Fund",
        ],
    },
    "Moderate": {
        "range": (29, 42),
        "allocation": {"equity": 50, "debt": 50},
        "categories": [
            "Large Cap Fund",
            "Multi Cap Fund",
            "Medium Duration Fund",
        ],
    },
    "Moderately Aggressive": {
        "range": (43, 56),
        "allocation": {"equity": 70, "debt": 30},
        "categories": [
            "Mid Cap Fund",
            "Flexi Cap Fund",
            "Short Duration Fund",
            "Small Cap Fund",
        ],
    },
    "Aggressive": {
        "range": (57, 70),
        "allocation": {"equity": 85, "debt": 15},
        "categories": [
            "Small Cap Fund",
            "Flexi Cap Fund",
            "Liquid Fund",
            "Large Cap Fund",
            "Mid Cap Fund",
        ],
    },
}

# Keyword patterns for fund category matching
CATEGORY_PATTERNS = {
    "Arbitrage Fund": re.compile(r"arbitrage", re.IGNORECASE),
    "Liquid Fund": re.compile(r"liquid", re.IGNORECASE),
    "Ultra Short Duration Fund": re.compile(r"ultra\s+short|ultra short", re.IGNORECASE),
    "Short Duration Fund": re.compile(r"short\s+duration|short term", re.IGNORECASE),
    "Medium Duration Fund": re.compile(r"medium\s+duration", re.IGNORECASE),
    "Large Cap Fund": re.compile(r"large\s+cap|largecap", re.IGNORECASE),
    "Mid Cap Fund": re.compile(r"mid\s+cap|midcap", re.IGNORECASE),
    "Small Cap Fund": re.compile(r"small\s+cap|smallcap", re.IGNORECASE),
    "Flexi Cap Fund": re.compile(r"flexi\s+cap|flexicap", re.IGNORECASE),
    "Multi Cap Fund": re.compile(r"multi\s+cap|multicap", re.IGNORECASE),
    "Multi Asset Fund": re.compile(r"multi\s+asset", re.IGNORECASE),
    "Balanced Advantage Fund": re.compile(r"balanced\s+advantage", re.IGNORECASE),
}




# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class FundPerformance:
    """Fund performance metrics."""

    scheme_code: str
    scheme_name: str
    category: str
    sharpe_ratio: float
    sortino_ratio: float
    pr_fqs_score: float


@dataclass
class PortfolioAllocation:
    """Individual portfolio fund allocation."""

    scheme_code: str
    fund_name: str
    category: str
    sip_amount: float
    pr_fqs_score: float
    expected_return: float
    confidence_flag: str = "Standard"  # 'Standard' or 'LowSampleSize'


@dataclass
class GeneratedPortfolio:
    """Complete generated portfolio."""

    client_profile: Dict
    recommended_portfolio: List[PortfolioAllocation]
    total_sip: float
    risk_tier: str
    allocation_breakdown: Dict[str, float]


# ============================================================================
# RISK SCORING & TIER MAPPING
# ============================================================================


def calculate_risk_score(assessment_answers: Dict[int, int]) -> int:
    """
    Calculate total risk score from psychometric assessment.

    Args:
        assessment_answers: Dict mapping question ID to selected option score

    Returns:
        Total risk score (7-70)
    """
    return sum(assessment_answers.values())


def get_risk_tier(risk_score: int) -> str:
    """
    Map risk score to SEBI-compliant risk tier.

    Args:
        risk_score: Total risk score (7-70)

    Returns:
        Risk tier name
    """
    for tier_name, tier_data in RISK_TIERS.items():
        min_score, max_score = tier_data["range"]
        if min_score <= risk_score <= max_score:
            return tier_name
    return "Moderate"  # Default fallback


# ============================================================================
# DATABASE LOADING (OFFLINE)
# ============================================================================


def load_screener_database(parquet_path: str = None) -> pd.DataFrame:
    """
    Load the offline screener database from Parquet file.
    
    Args:
        parquet_path: Path to Parquet file. If None, uses default path.
    
    Returns:
        DataFrame with all funds and their metrics.
    
    Raises:
        FileNotFoundError: If Parquet file has not been built yet.
        ValueError: If DataFrame is empty or malformed.
    """
    if parquet_path is None:
        parquet_path = SCREENER_DATABASE_PATH
    
    parquet_path = Path(parquet_path)
    
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Screener database not found at {parquet_path}. "
            f"Please run 'python scripts/database_builder.py' to build it first."
        )
    
    try:
        df_master = pd.read_parquet(parquet_path)
        
        if df_master.empty:
            raise ValueError("Screener database is empty. Please rebuild it.")
        
        # Verify required columns
        required_columns = {'fund_code', 'scheme_name', 'category', 'pr_fqs_score', 'sharpe', 'sortino'}
        missing_columns = required_columns - set(df_master.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"[DATABASE] Loaded screener database: {len(df_master)} funds across {df_master['category'].nunique()} categories")
        return df_master
    
    except Exception as e:
        raise RuntimeError(f"Failed to load screener database: {e}")


# ============================================================================
# FUND SELECTION (OFFLINE)
# ============================================================================


def fetch_best_funds_for_category(
    category: str, df_master: pd.DataFrame
) -> Optional[Tuple[str, str, float, float, float]]:
    """
    Select the highest-ranking fund for a category from the offline database.
    Uses pre-computed PR-FQS scores.
    
    Args:
        category: Target SEBI category name
        df_master: Loaded screener database DataFrame
    
    Returns:
        Tuple of (scheme_code, scheme_name, pr_fqs_score, sortino, sharpe)
        or None if no funds found for category
    """
    # Filter by category
    category_funds = df_master[df_master['category'] == category]
    
    if category_funds.empty:
        print(f"[FUND-SELECT] No funds found in database for category: {category}")
        return None
    
    # Select top-ranked fund by PR-FQS score
    best_fund = category_funds.nlargest(1, 'pr_fqs_score').iloc[0]
    
    scheme_code = str(best_fund['fund_code'])
    scheme_name = best_fund['scheme_name']
    pr_fqs_score = float(best_fund['pr_fqs_score'])
    sortino = float(best_fund.get('sortino', 0))
    sharpe = float(best_fund.get('sharpe', 0))
    
    print(f"[FUND-SELECT] Category '{category}': Selected fund {scheme_code} ({scheme_name}) with PR-FQS score {pr_fqs_score:.2f}")
    
    return (scheme_code, scheme_name, pr_fqs_score, sortino, sharpe)


# ============================================================================
# PORTFOLIO BUILDER
# ============================================================================


def generate_portfolio(
    assessment_answers: Dict[int, int], monthly_sip: float
) -> GeneratedPortfolio:
    """
    Build complete portfolio based on risk score and monthly SIP amount.
    Uses offline Parquet database for fund selection.
    Pure offline execution - no network calls.

    Args:
        assessment_answers: Dict mapping question ID to risk score
        monthly_sip: Monthly SIP amount in rupees

    Returns:
        GeneratedPortfolio object
    
    Raises:
        FileNotFoundError: If screener database not found
    """
    # Load the offline database (once)
    df_master = load_screener_database()
    
    # Calculate risk tier and allocation
    risk_score = calculate_risk_score(assessment_answers)
    risk_tier = get_risk_tier(risk_score)
    tier_data = RISK_TIERS[risk_tier]
    allocation = tier_data["allocation"]
    categories = tier_data["categories"]

    print(f"\n[PORTFOLIO] Generating portfolio for risk tier: {risk_tier} (score: {risk_score})")
    print(f"[PORTFOLIO] Monthly SIP: ₹{monthly_sip:,.0f}")
    print(f"[PORTFOLIO] Allocation: {allocation['equity']}% Equity, {allocation['debt']}% Debt")

    # Calculate allocations
    equity_allocation = (allocation["equity"] / 100) * monthly_sip
    debt_allocation = (allocation["debt"] / 100) * monthly_sip

    recommended_portfolio = []

    # Count equity vs debt categories for proper allocation
    equity_categories = [
        c
        for c in categories
        if any(kw in c for kw in ["Small Cap", "Mid Cap", "Flexi Cap", "Large Cap", "Multi Cap"])
    ]
    debt_categories = [c for c in categories if c not in equity_categories]

    print(f"[PORTFOLIO] Equity categories: {len(equity_categories)}, Debt categories: {len(debt_categories)}")

    # Fetch best fund for each category from offline database
    for category in categories:
        fund_data = fetch_best_funds_for_category(category, df_master)

        if fund_data:
            scheme_code, scheme_name, pr_fqs_score, sortino, sharpe = fund_data
            
            # Determine if category is equity or debt
            is_equity = category in equity_categories

            # Allocate proportionally
            if is_equity:
                allocation_amount = (
                    equity_allocation / len(equity_categories)
                    if equity_categories
                    else 0
                )
            else:
                allocation_amount = (
                    debt_allocation / len(debt_categories) if debt_categories else 0
                )

            if allocation_amount > 0:
                # Conservative expected return: PR-FQS score represents percentile rank
                # Assume 80% conversion to actual annual return estimate
                expected_return = pr_fqs_score * 0.8 / 100  # Normalize to 0-0.8 (0-80% annual)
                
                portfolio_allocation = PortfolioAllocation(
                    scheme_code=scheme_code,
                    fund_name=scheme_name,
                    category=category,
                    sip_amount=allocation_amount,
                    pr_fqs_score=pr_fqs_score,
                    expected_return=expected_return,
                    confidence_flag="Standard"
                )
                recommended_portfolio.append(portfolio_allocation)
                
                print(f"[PORTFOLIO] Allocated ₹{allocation_amount:,.0f} to {category}")
        else:
            print(f"[PORTFOLIO] WARNING: No fund found for category {category}")

    # Ensure total matches monthly SIP (adjust rounding errors)
    total_allocated = sum(alloc.sip_amount for alloc in recommended_portfolio)
    if total_allocated > 0 and abs(total_allocated - monthly_sip) > 1:
        adjustment_factor = monthly_sip / total_allocated
        for alloc in recommended_portfolio:
            alloc.sip_amount *= adjustment_factor
        print(f"[PORTFOLIO] Adjusted allocations for rounding precision")

    print(f"[PORTFOLIO] Portfolio generation complete: {len(recommended_portfolio)} funds selected\n")

    return GeneratedPortfolio(
        client_profile={
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "monthly_sip": monthly_sip,
            "assessment_date": datetime.now().isoformat(),
        },
        recommended_portfolio=recommended_portfolio,
        total_sip=monthly_sip,
        risk_tier=risk_tier,
        allocation_breakdown=allocation,
    )
