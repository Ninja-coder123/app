"""
Standalone Data Engineering Script: Database Builder

Builds an offline master database of Indian mutual funds by:
1. Scraping active schemes from mfapi.in
2. Filtering by Direct+Growth criteria
3. Running PR-FQS quantitative scoring
4. Exporting to compressed Parquet

Runs completely independent of frontend. No Streamlit dependencies.

Author: Wealth Management SaaS Backend
License: Internal Use Only
"""

import asyncio
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import aiohttp
import pandas as pd
import requests

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.pr_fqs_engine import calculate_category_pr_fqs, PRFQSResult


# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

MFAPI_BASE_URL = "https://api.mfapi.in"
MASTER_INDEX_URL = f"{MFAPI_BASE_URL}/mf"
NAV_DATA_URL_TEMPLATE = f"{MFAPI_BASE_URL}/mf" + "/{scheme_code}"

# Rate limiting
ASYNC_SEMAPHORE_LIMIT = 5
REQUEST_DELAY = 0.05  # seconds between requests

# Directories
DATA_DIR = Path(__file__).parent.parent / "data"
PARQUET_OUTPUT_FILE = DATA_DIR / "screener_database.parquet"

# Date thresholds
SEBI_DIRECT_PLAN_INCEPTION = datetime(2013, 1, 1)
DEAD_FUND_DAYS_THRESHOLD = 20

# SEBI Categories to extract
SEBI_CATEGORIES = [
    "Large Cap Fund",
    "Mid Cap Fund",
    "Small Cap Fund",
    "Flexi Cap Fund",
    "Multi Cap Fund",
    "Liquid Fund",
    "Short Duration Fund",
    "Medium Duration Fund",
    "Arbitrage Fund",
    "Balanced Advantage Fund",
]


# ============================================================================
# DIRECTORY CHECK & INITIALIZATION
# ============================================================================

def ensure_data_directory():
    """Ensure the data/ directory exists locally before saving."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INIT] Data directory ensured: {DATA_DIR}")


# ============================================================================
# MASTER SCHEME DISCOVERY
# ============================================================================

def fetch_master_index() -> Optional[Dict[str, Dict]]:
    """
    Fetch the master index list from mfapi.in.
    
    Returns:
        Dictionary mapping scheme_code -> {name, category, etc.}
        or None if fetch fails.
    """
    try:
        print(f"[MASTER] Fetching master index from {MASTER_INDEX_URL}...")
        response = requests.get(MASTER_INDEX_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Parse into dict keyed by scheme_code
        schemes_dict = {item['schemeCode']: item for item in data}
        print(f"[MASTER] Total schemes in index: {len(schemes_dict)}")
        return schemes_dict
    except Exception as e:
        print(f"[ERROR] Failed to fetch master index: {e}")
        return None


def apply_regex_filter(schemes_dict: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Apply strict regex filter: Keep ONLY schemes with BOTH "Direct" AND "Growth".
    Reject all "Regular", "IDCW", "Dividend", "Bonus", "Institutional" plans.
    
    Args:
        schemes_dict: Dictionary of all schemes from master index.
    
    Returns:
        Filtered dictionary of valid Direct+Growth schemes.
    """
    filtered = {}
    rejected_reasons = {
        "no_direct": 0,
        "no_growth": 0,
        "has_regular": 0,
        "has_idcw": 0,
        "has_dividend": 0,
        "has_bonus": 0,
        "has_institutional": 0,
    }
    
    for scheme_code, scheme_info in schemes_dict.items():
        name = scheme_info.get('schemeName', '')
        
        # Must have BOTH "Direct" AND "Growth"
        has_direct = re.search(r'direct', name, re.IGNORECASE) is not None
        has_growth = re.search(r'growth', name, re.IGNORECASE) is not None
        
        if not has_direct:
            rejected_reasons["no_direct"] += 1
            continue
        if not has_growth:
            rejected_reasons["no_growth"] += 1
            continue
        
        # Reject blacklisted keywords
        if re.search(r'regular', name, re.IGNORECASE):
            rejected_reasons["has_regular"] += 1
            continue
        if re.search(r'idcw', name, re.IGNORECASE):
            rejected_reasons["has_idcw"] += 1
            continue
        if re.search(r'dividend', name, re.IGNORECASE):
            rejected_reasons["has_dividend"] += 1
            continue
        if re.search(r'bonus', name, re.IGNORECASE):
            rejected_reasons["has_bonus"] += 1
            continue
        if re.search(r'institutional', name, re.IGNORECASE):
            rejected_reasons["has_institutional"] += 1
            continue
        
        # Scheme passes all filters
        filtered[scheme_code] = scheme_info
    
    print(f"[FILTER] Filtered to {len(filtered)} Direct+Growth schemes")
    print(f"[FILTER] Rejection breakdown:")
    for reason, count in rejected_reasons.items():
        if count > 0:
            print(f"         - {reason}: {count}")
    
    return filtered


def clean_sebi_category(raw_category: str) -> str:
    """Cleans up the long SEBI API strings into standard tags."""
    if not isinstance(raw_category, str):
        return "Unclassified"
        
    raw = raw_category.lower()
    
    # Official SEBI Equity Categories
    if "large cap" in raw: return "Large Cap Fund"
    if "mid cap" in raw: return "Mid Cap Fund"
    if "small cap" in raw: return "Small Cap Fund"
    if "flexi cap" in raw: return "Flexi Cap Fund"
    if "multi cap" in raw: return "Multi Cap Fund"
    
    # Official SEBI Debt Categories
    if "liquid" in raw: return "Liquid Fund"
    if "ultra short" in raw: return "Ultra Short Duration Fund"
    if "short duration" in raw: return "Short Duration Fund"
    if "medium duration" in raw: return "Medium Duration Fund"
    
    # Official SEBI Hybrid Categories
    if "arbitrage" in raw: return "Arbitrage Fund"
    if "balanced advantage" in raw or "dynamic asset" in raw: return "Balanced Advantage Fund"
    if "multi asset" in raw: return "Multi Asset Fund"
    
    # Return the raw string directly if it's a new or different category (opens up all 40+ SEBI categories)
    return raw_category.strip()

# ============================================================================
# NAV DATA INGESTION & RATE LIMITING
# ============================================================================

async def fetch_fund_nav_async(
    session: aiohttp.ClientSession,
    scheme_code: str,
    semaphore: asyncio.Semaphore,
    delay: float = REQUEST_DELAY
) -> Tuple[str, Optional[pd.DataFrame], str, str]:
    """
    Asynchronously fetch NAV data and official metadata for a single fund with rate limiting.
    """
    async with semaphore:
        try:
            url = f"{MFAPI_BASE_URL}/mf/{scheme_code}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    return scheme_code, None, "", ""
                
                data = await response.json()
                
                # Extract exact official category and name from meta block
                meta = data.get("meta", {})
                raw_category = meta.get("scheme_category", "Unclassified")
                official_category = clean_sebi_category(raw_category)
                scheme_name = meta.get("scheme_name", f"Fund_{scheme_code}")
                
                nav_history = data.get('data', [])
                
                if not nav_history:
                    return scheme_code, None, official_category, scheme_name
                
                # Parse NAV data
                records = []
                for entry in nav_history:
                    try:
                        date = datetime.strptime(entry['date'], '%d-%m-%Y')
                        nav = float(entry['nav'])
                        records.append({'date': date, 'nav': nav})
                    except (ValueError, TypeError, KeyError):
                        continue
                
                if not records:
                    return scheme_code, None, official_category, scheme_name
                
                df = pd.DataFrame(records)
                df = df.sort_values('date').reset_index(drop=True)
                
                await asyncio.sleep(delay)
                return scheme_code, df, official_category, scheme_name
        
        except Exception as e:
            print(f"[WARN] Failed to fetch NAV for scheme {scheme_code}: {e}")
            return scheme_code, None, "", ""


async def fetch_all_nav_data_async(
    scheme_codes: Set[str],
    semaphore_limit: int = ASYNC_SEMAPHORE_LIMIT
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Set[str]], Dict[str, str]]:
    """
    Fetch NAV data for all schemes concurrently and route by official metadata.
    """
    print(f"[NAV] Starting async fetch for {len(scheme_codes)} schemes...")
    semaphore = asyncio.Semaphore(semaphore_limit)
    funds_dict = {}
    category_map = {}
    scheme_name_map = {}
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_fund_nav_async(session, code, semaphore)
            for code in scheme_codes
        ]
        results = await asyncio.gather(*tasks)
        
        for scheme_code, nav_df, category, name in results:
            if nav_df is not None and category:
                funds_dict[scheme_code] = nav_df
                scheme_name_map[scheme_code] = name
                
                if category not in category_map:
                    category_map[category] = set()
                category_map[category].add(scheme_code)
    
    print(f"[NAV] Successfully fetched {len(funds_dict)} funds")
    return funds_dict, category_map, scheme_name_map


def filter_nav_history(nav_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter NAV history to post-2013 era (SEBI Direct Plan inception: 2013-01-01).
    
    Args:
        nav_df: DataFrame with 'date' and 'nav' columns.
    
    Returns:
        Filtered DataFrame (2013-01-01 onwards).
    """
    return nav_df[nav_df['date'] >= SEBI_DIRECT_PLAN_INCEPTION].reset_index(drop=True)


# ============================================================================
# 20-DAY DEAD FUND GATE (GARBAGE COLLECTOR)
# ============================================================================

def apply_dead_fund_gate(funds_dict: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Filter out inactive/merged funds: if latest NAV date is > 20 days old, DROP.
    
    Args:
        funds_dict: Dictionary mapping scheme_code -> nav_df.
    
    Returns:
        Tuple of (filtered_dict, dropped_codes_list).
    """
    print(f"[DEAD_FUND] Applying 20-day gate to {len(funds_dict)} funds...")
    cutoff_date = datetime.now() - timedelta(days=DEAD_FUND_DAYS_THRESHOLD)
    
    filtered = {}
    dropped = []
    
    for scheme_code, nav_df in funds_dict.items():
        if nav_df.empty:
            dropped.append(scheme_code)
            continue
        
        latest_nav_date = nav_df['date'].max()
        days_old = (datetime.now() - latest_nav_date).days
        
        if days_old > DEAD_FUND_DAYS_THRESHOLD:
            dropped.append(scheme_code)
        else:
            filtered[scheme_code] = nav_df
    
    print(f"[DEAD_FUND] Dropped {len(dropped)} inactive funds (latest NAV > 20 days old)")
    print(f"[DEAD_FUND] Active funds remaining: {len(filtered)}")
    
    return filtered, dropped


# ============================================================================
# PR-FQS INTEGRATION
# ============================================================================

def process_category_with_prfqs(
    category: str,
    scheme_codes: Set[str],
    funds_dict: Dict[str, pd.DataFrame],
    scheme_name_map: Dict[str, str]
) -> Optional[pd.DataFrame]:
    """
    Group funds by category and run PR-FQS engine.
    
    Args:
        category: SEBI category name.
        scheme_codes: Set of scheme codes in this category.
        funds_dict: Dictionary mapping scheme_code -> nav_df.
        scheme_name_map: Dictionary mapping scheme_code -> scheme_name.
    
    Returns:
        PR-FQS ranked DataFrame with all metrics, or None if calculation failed.
    """
    print(f"\n[PR-FQS] Processing category: {category}")
    
    # Filter to only schemes we have NAV data for
    category_funds = {
        code: funds_dict[code]
        for code in scheme_codes
        if code in funds_dict
    }
    
    if not category_funds:
        print(f"[PR-FQS] No active funds in category: {category}")
        return None
    
    print(f"[PR-FQS] Category funds count: {len(category_funds)}")
    
    # Call PR-FQS engine
    result: PRFQSResult = calculate_category_pr_fqs(category, category_funds, verbose=False)
    
    if not result.is_valid or result.ranked_funds is None:
        print(f"[PR-FQS] Calculation failed for {category}: {result.error_message}")
        return None
    
    # Add category and scheme_name to results
    ranked_df = result.ranked_funds.copy()
    ranked_df['category'] = category
    ranked_df['scheme_name'] = ranked_df['fund_code'].map(scheme_name_map)
    
    # Reorder columns for clarity
    cols = ['fund_code', 'scheme_name', 'category'] + [
        c for c in ranked_df.columns
        if c not in ['fund_code', 'scheme_name', 'category']
    ]
    ranked_df = ranked_df[[c for c in cols if c in ranked_df.columns]]
    
    print(f"[PR-FQS] {category}: {len(ranked_df)} funds ranked")
    
    return ranked_df


# ============================================================================
# ATOMIC PARQUET EXPORT
# ============================================================================

def flatten_and_export_to_parquet(
    all_results: Dict[str, pd.DataFrame],
    funds_dict: Dict[str, pd.DataFrame],
    scheme_name_map: Dict[str, str]
) -> bool:
    """
    Flatten all category results into consolidated master DataFrame and export to Parquet.
    
    Args:
        all_results: Dictionary mapping category -> ranked_df.
        funds_dict: Dictionary mapping scheme_code -> nav_df.
        scheme_name_map: Dictionary mapping scheme_code -> scheme_name.
    
    Returns:
        True if export successful, False otherwise.
    """
    print(f"\n[EXPORT] Consolidating results from {len(all_results)} categories...")
    
    # Concatenate all category results
    dfs_to_concat = [df for df in all_results.values() if df is not None]
    
    if not dfs_to_concat:
        print("[EXPORT] No valid results to export")
        return False
    
    master_df = pd.concat(dfs_to_concat, ignore_index=True)
    
    # Add latest NAV and last update date from funds_dict
    master_df['latest_nav'] = master_df['fund_code'].apply(
        lambda code: funds_dict[code]['nav'].iloc[-1]
        if code in funds_dict and not funds_dict[code].empty
        else None
    )
    master_df['last_updated_date'] = master_df['fund_code'].apply(
        lambda code: funds_dict[code]['date'].iloc[-1]
        if code in funds_dict and not funds_dict[code].empty
        else None
    )
    
    # Define expected columns
    expected_columns = [
        'fund_code', 'scheme_name', 'category',
        'latest_nav', 'last_updated_date',
        'sharpe', 'sortino', 'volatility',
        'rolling_return', 'relative_return', 'consistency', 'beta',
        'sharpe_norm', 'sortino_norm', 'volatility_norm',
        'rolling_return_norm', 'relative_return_norm', 'consistency_norm', 'beta_fit_norm',
        'pr_fqs_score'
    ]
    
    # Keep only columns that exist
    available_cols = [c for c in expected_columns if c in master_df.columns]
    master_df = master_df[available_cols]
    
    print(f"[EXPORT] Master DataFrame shape: {master_df.shape}")
    print(f"[EXPORT] Columns: {list(master_df.columns)}")
    
    # Save to Parquet with snappy compression
    try:
        PARQUET_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        master_df.to_parquet(
            PARQUET_OUTPUT_FILE,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        print(f"[EXPORT] Successfully exported to: {PARQUET_OUTPUT_FILE}")
        print(f"[EXPORT] File size: {PARQUET_OUTPUT_FILE.stat().st_size / (1024*1024):.2f} MB")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to export Parquet: {e}")
        return False


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

async def main():
    """Main orchestration function for database builder."""
    start_time = datetime.now()
    print(f"\n{'='*80}")
    print(f"DATABASE BUILDER - STARTED")
    print(f"Timestamp: {start_time}")
    print(f"{'='*80}\n")
    
    # Step 1: Ensure data directory exists
    ensure_data_directory()
    
    # Step 2: Fetch master index
    master_schemes = fetch_master_index()
    if not master_schemes:
        print("[ERROR] Failed to fetch master index. Exiting.")
        return False
    
    # Step 3: Apply regex filter (Direct + Growth)
    filtered_schemes = apply_regex_filter(master_schemes)
    if not filtered_schemes:
        print("[ERROR] No schemes passed regex filter. Exiting.")
        return False

    # Step 4 & 5: Fetch NAV data and extract official SEBI categories via metadata
    all_scheme_codes = set(filtered_schemes.keys())
    funds_dict, category_map, scheme_name_map = await fetch_all_nav_data_async(all_scheme_codes, ASYNC_SEMAPHORE_LIMIT)
    
    # Update the global category list dynamically based on what was actually fetched
    global SEBI_CATEGORIES
    SEBI_CATEGORIES = list(category_map.keys())
    print(f"[CATEGORIES] Extracted {len(SEBI_CATEGORIES)} official categories from metadata.")
   
    if not funds_dict:
        print("[ERROR] Failed to fetch any NAV data. Exiting.")
        return False
    
    # Step 6: Filter NAV history to post-2013 era
    print(f"\n[FILTER] Filtering NAV history to post-2013 era...")
    funds_dict = {
        code: filter_nav_history(nav_df)
        for code, nav_df in funds_dict.items()
        if not nav_df.empty
    }
    print(f"[FILTER] Funds with valid post-2013 data: {len(funds_dict)}")
    
    # Step 7: Apply 20-day dead fund gate
    funds_dict, dropped_funds = apply_dead_fund_gate(funds_dict)
    
    if not funds_dict:
        print("[ERROR] All funds filtered out by dead fund gate. Exiting.")
        return False
    
    # Step 8: Run PR-FQS for each category
    print(f"\n{'='*80}")
    print(f"PR-FQS CALCULATION PHASE")
    print(f"{'='*80}")
    
    all_results = {}
    total_funds_processed = 0
    
    for category in SEBI_CATEGORIES:
        scheme_codes = category_map.get(category, set())
        
        # Filter to only active funds
        active_codes = scheme_codes & set(funds_dict.keys())
        
        if not active_codes:
            print(f"\n[PR-FQS] Skipping {category}: no active funds")
            continue
        
        ranked_df = process_category_with_prfqs(
            category,
            active_codes,
            funds_dict,
            scheme_name_map
        )
        
        if ranked_df is not None:
            all_results[category] = ranked_df
            total_funds_processed += len(ranked_df)
    
    print(f"\n[PR-FQS] Total funds ranked across all categories: {total_funds_processed}")
    
    # Step 9: Flatten and export to Parquet
    print(f"\n{'='*80}")
    print(f"EXPORT PHASE")
    print(f"{'='*80}\n")
    
    success = flatten_and_export_to_parquet(all_results, funds_dict, scheme_name_map)
    
    # Final summary
    elapsed = datetime.now() - start_time
    print(f"\n{'='*80}")
    print(f"DATABASE BUILDER - COMPLETED")
    print(f"Elapsed time: {elapsed.total_seconds():.2f} seconds")
    print(f"Dropped dead funds: {len(dropped_funds)}")
    print(f"Final fund count per category:")
    for category in SEBI_CATEGORIES:
        if category in all_results:
            print(f"  - {category}: {len(all_results[category])}")
    print(f"{'='*80}\n")
    
    return success


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        print("\n[ERROR] Database builder interrupted by user.")
        exit_code = 1
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    
    exit(exit_code)
