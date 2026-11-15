"""
Peer-Relative Fund Quality Score (PR-FQS) Engine.

A rigorous, zero-cost quantitative fund evaluation framework for SEBI categories.
Uses synthetic peer-relative benchmarking (leave-one-out) and institutional-grade
performance scoring to rank funds without external paid indices.

Author: Wealth Management SaaS Backend
License: Internal Use Only
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import requests
from dataclasses import dataclass


@dataclass
class PRFQSResult:
    """Container for PR-FQS calculation results with explicit diagnostic status."""
    category: str
    eligible_fund_count: int
    is_valid: bool
    status: str = "VALID"  # VALID, INSUFFICIENT_PEERS, INSUFFICIENT_HISTORY
    error_message: Optional[str] = None
    ranked_funds: Optional[pd.DataFrame] = None


class PRFQSEngine:
    """
    Peer-Relative Fund Quality Score Engine.
    
    Implements a zero-cost, vectorized fund scoring pipeline using mfapi.in
    for Indian mutual fund data. Enforces strict data gates and uses leave-one-out
    benchmarking to eliminate self-inclusion bias.
    """
    
    # PR-FQS weighting scheme (must sum to 1.0)
    WEIGHTS = {
        'sharpe': 0.20,
        'sortino': 0.15,
        'rolling_return': 0.15,
        'relative_return': 0.15,
        'consistency': 0.15,
        'beta_fit': 0.10,
        'volatility': 0.10,
    }
    
    # Risk-free rate (India: ~6% annual, daily equivalent)
    RISK_FREE_RATE_ANNUAL = 0.06
    RISK_FREE_RATE_DAILY = RISK_FREE_RATE_ANNUAL / 252
    
    # Minimum sample size gate
    MIN_ELIGIBLE_FUNDS = 11
    
    # Data window
    DATA_YEARS = 3
    DATA_DAYS = 3 * 252  # Approximate trading days in 3 years
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the PR-FQS Engine.
        
        Args:
            verbose: If True, print debug information during calculations.
        """
        self.verbose = verbose
        self._log = self._print if verbose else lambda *args, **kwargs: None
    
    @staticmethod
    def _print(*args, **kwargs):
        """Helper for verbose logging."""
        print(*args, **kwargs)
    
    def fetch_fund_data(self, fund_code: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical NAV data from mfapi.in for a single fund.
        
        Args:
            fund_code: AMFI fund code (e.g., '112961' for SBI Bluechip Fund).
        
        Returns:
            DataFrame with columns ['date', 'nav'] or None if fetch fails.
        """
        try:
            url = f"https://api.mfapi.in/mf/{fund_code}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Parse NAV data
            nav_data = data.get('data', [])
            if not nav_data:
                return None
            
            df = pd.DataFrame(nav_data)
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
            df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
            df = df[['date', 'nav']].dropna()
            df = df.sort_values('date').reset_index(drop=True)
            
            return df
        except Exception as e:
            self._log(f"Failed to fetch data for {fund_code}: {e}")
            return None
    
    def fetch_category_funds(self, category: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch all funds in a given SEBI category.
        
        This is a stub that should be connected to a live SEBI fund list API
        or a pre-cached fund master list. For production, integrate with:
        - SEBI's official fund registry
        - mfapi.in's fund list
        - Your proprietary fund database
        
        Args:
            category: SEBI category name (e.g., 'Large Cap', 'Small Cap').
        
        Returns:
            Dictionary mapping fund_code -> DataFrame with NAV history.
        """
        # TODO: Implement actual fund fetching from your data source
        # For now, return empty dict as placeholder
        self._log(f"Fetching funds for category: {category}")
        return {}
    
    def _apply_3year_gate(self, funds_dict: Dict[str, pd.DataFrame], 
                          cutoff_date: datetime) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """
        Apply the 3-Year Continuous History Gate.
        
        HARD GATE: Drop any fund without continuous, uninterrupted NAV data
        spanning the full 3-year lookback period. Only drop for incomplete data,
        never for bad performance.
        
        Args:
            funds_dict: Dictionary mapping fund_code -> DataFrame with NAV.
            cutoff_date: The T-3 (3 years ago) cutoff date.
        
        Returns:
            Tuple of (eligible_funds, dropped_fund_codes).
        """
        eligible = {}
        dropped = []
        
        for fund_code, df in funds_dict.items():
            if df is None or df.empty:
                dropped.append(fund_code)
                continue
            
            # Check: Does fund have data spanning T-3 to today?
            min_date = df['date'].min()
            max_date = df['date'].max()
            today = datetime.now()
            
            if min_date > cutoff_date or max_date < today - timedelta(days=7):
                dropped.append(fund_code)
                self._log(f"Dropped {fund_code}: incomplete 3-year history. "
                         f"Has {min_date} to {max_date}, needs {cutoff_date} to {today}")
                continue
            
            # Filter to exactly 3-year window
            df_filtered = df[(df['date'] >= cutoff_date) & (df['date'] <= today)].copy()
            
            if df_filtered.empty:
                dropped.append(fund_code)
                continue
            
            eligible[fund_code] = df_filtered
        
        self._log(f"After 3-Year Gate: {len(eligible)} eligible, {len(dropped)} dropped")
        return eligible, dropped
    
    MIN_PEER_COUNT = 10  # Requires at least 11 total funds so that N-1 >= 10 peers for leave-one-out

    def _apply_sample_size_gate(self, returns_df: pd.DataFrame) -> Tuple[bool, str, str]:
        """
        Apply the Sample Size Gate based on usable aligned return series.
        
        Because of leave-one-out benchmarking, each fund is evaluated against N-1 peers.
        We check the final usable column count in returns_df.
        """
        n_funds = returns_df.shape[1]
        peer_count = max(0, n_funds - 1)
        
        if peer_count < self.MIN_PEER_COUNT:
            msg = (
                f"❌ REJECTED: Only {peer_count} usable peers available "
                f"(after data alignment); minimum required is {self.MIN_PEER_COUNT}."
            )
            print(msg)
            self._log(msg)
            return False, "INSUFFICIENT_PEERS", msg
        
        msg = f"✓ Sample Size Gate Passed: {peer_count} usable peers >= {self.MIN_PEER_COUNT}"
        self._log(msg)
        return True, "VALID", msg
    
    def _build_returns_dataframe(self, eligible_funds: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Build a vectorized DataFrame of daily returns for all eligible funds.
        
        Args:
            eligible_funds: Dictionary of fund_code -> NAV DataFrame.
        
        Returns:
            DataFrame with shape (trading_days, num_funds), columns are fund codes,
            values are daily returns (log returns).
        """
        # Align all fund dates to a common index
        all_dates = set()
        for df in eligible_funds.values():
            all_dates.update(df['date'].unique())
        
        common_dates = sorted(all_dates)
        common_df = pd.DataFrame({'date': common_dates})
        
        # Merge each fund's NAV
        for fund_code, df in eligible_funds.items():
            df_copy = df[['date', 'nav']].copy()
            df_copy.columns = ['date', fund_code]
            common_df = common_df.merge(df_copy, on='date', how='left')
        
        # Forward-fill any small gaps, then drop rows with missing data
        common_df = common_df.ffill().dropna()
        common_df = common_df.set_index('date').sort_index()
        
        # Calculate log returns
        returns_df = np.log(common_df / common_df.shift(1)).dropna()
        
        # DEBUG: Print actual data shape
        print(f"[PR-FQS DATA] Returns DataFrame Shape: {returns_df.shape}")
        print(f"[PR-FQS DATA] Trading days: {returns_df.shape[0]}, Funds: {returns_df.shape[1]}")
        print(f"[PR-FQS DATA] Fund codes: {list(returns_df.columns)}")
        print(f"[PR-FQS DATA] Date range: {returns_df.index.min()} to {returns_df.index.max()}")
        
        self._log(f"Returns DataFrame: {returns_df.shape[0]} days × {returns_df.shape[1]} funds")
        return returns_df
    
    def _calculate_absolute_metrics(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Absolute Metrics for each fund.
        
        Metrics:
        - Sharpe Ratio: (mean_return - risk_free_rate) / std_dev
        - Sortino Ratio: (mean_return - risk_free_rate) / downside_std_dev
        - Volatility: Standard deviation of daily returns (annualized).
        
        Args:
            returns_df: DataFrame of daily returns (trading_days × num_funds).
        
        Returns:
            DataFrame with columns [fund_code, sharpe, sortino, volatility].
        """
        metrics = []
        
        for fund_code in returns_df.columns:
            r = returns_df[fund_code].values
            
            # Annualized metrics
            mean_return = r.mean() * 252
            volatility = r.std() * np.sqrt(252)
            
            # Sharpe Ratio
            sharpe = (mean_return - self.RISK_FREE_RATE_ANNUAL) / volatility if volatility > 0 else 0
            
            # Sortino Ratio: uses only downside volatility
            downside_returns = r[r < self.RISK_FREE_RATE_DAILY]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
            sortino = (mean_return - self.RISK_FREE_RATE_ANNUAL) / downside_vol if downside_vol > 0 else 0
            
            metrics.append({
                'fund_code': fund_code,
                'sharpe': sharpe,
                'sortino': sortino,
                'volatility': volatility,
            })
        
        df = pd.DataFrame(metrics)
        self._log(f"Absolute Metrics calculated for {len(df)} funds")
        return df
    
    def _calculate_peer_relative_metrics(self, returns_df: pd.DataFrame, 
                                          rolling_window: int = 63) -> pd.DataFrame:
        """
        Calculate Peer-Relative Metrics using Leave-One-Out benchmarking.
        
        For each fund, excludes itself from the peer benchmark to prevent self-inclusion bias.
        
        Metrics:
        - Rolling Return: 3-year annualized return.
        - Relative Return: Fund return - peer median return.
        - Consistency: % of periods where fund beat peer benchmark.
        - Beta: Covariance(fund, peer_bench) / Variance(peer_bench).
        
        Args:
            returns_df: DataFrame of daily returns (trading_days × num_funds).
            rolling_window: Window size for rolling calculations (default 63 days ≈ 3 months).
        
        Returns:
            DataFrame with columns [fund_code, rolling_return, relative_return, consistency, beta].
        """
        metrics = []
        
        for target_fund in returns_df.columns:
            # Leave-one-out benchmark: median return of all OTHER funds
            peer_funds = returns_df.drop(columns=[target_fund])
            
            # DEBUG: Check peer funds for NaN
            if peer_funds.empty:
                print(f"[PR-FQS] WARNING: No peer funds for {target_fund}")
                continue
                
            peer_benchmark = peer_funds.median(axis=1)
            
            # Check for NaN collapse in peer benchmark
            nan_count = peer_benchmark.isna().sum()
            if nan_count > 0:
                print(f"[PR-FQS] WARNING: Peer benchmark for {target_fund} has {nan_count} NaN values")
                peer_benchmark = peer_benchmark.fillna(peer_benchmark.mean())  # Forward fill with mean
            
            target_returns = returns_df[target_fund]
            
            # 1. Rolling Return (3-year annualized)
            rolling_return = (1 + target_returns).rolling(self.DATA_DAYS).apply(
                lambda x: (x.prod() ** (252 / self.DATA_DAYS) - 1) if x.notna().all() else np.nan
            ).iloc[-1] if len(target_returns) > self.DATA_DAYS else target_returns.mean() * 252
            
            if np.isnan(rolling_return):
                rolling_return = target_returns.mean() * 252
            
            # 2. Relative Return: cumulative outperformance vs benchmark
            # Handle potential NaN mismatches
            valid_mask = ~(target_returns.isna() | peer_benchmark.isna())
            if valid_mask.sum() > 0:
                relative_daily = (target_returns[valid_mask].values - peer_benchmark[valid_mask].values)
                relative_return = relative_daily.mean() * 252
            else:
                relative_return = 0
            
            # 3. Consistency: % of rolling windows where fund beat benchmark
            rolling_sizes = rolling_window
            consistency_periods = []
            for i in range(len(target_returns) - rolling_sizes):
                fund_roll_returns = target_returns.iloc[i:i+rolling_sizes]
                bench_roll_returns = peer_benchmark.iloc[i:i+rolling_sizes]
                
                # Check for valid data
                if fund_roll_returns.notna().all() and bench_roll_returns.notna().all():
                    fund_roll = (1 + fund_roll_returns).prod()
                    bench_roll = (1 + bench_roll_returns).prod()
                    if fund_roll > bench_roll:
                        consistency_periods.append(1)
                    else:
                        consistency_periods.append(0)
            
            consistency = np.mean(consistency_periods) if consistency_periods else 0.5
            
            # 4. Beta: Cov(fund, benchmark) / Var(benchmark)
            valid_mask_beta = ~(target_returns.isna() | peer_benchmark.isna())
            if valid_mask_beta.sum() > 1:
                covariance = np.cov(target_returns[valid_mask_beta].values, peer_benchmark[valid_mask_beta].values)[0, 1]
                benchmark_variance = peer_benchmark[valid_mask_beta].var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
            else:
                beta = 1.0
            
            # Validate metrics
            if any(np.isnan(x) for x in [rolling_return, relative_return, consistency, beta]):
                print(f"[PR-FQS] WARNING: NaN detected in metrics for {target_fund}")
                rolling_return = rolling_return if not np.isnan(rolling_return) else 0
                relative_return = relative_return if not np.isnan(relative_return) else 0
                consistency = consistency if not np.isnan(consistency) else 0.5
                beta = beta if not np.isnan(beta) else 1.0
            
            metrics.append({
                'fund_code': target_fund,
                'rolling_return': rolling_return,
                'relative_return': relative_return,
                'consistency': consistency,
                'beta': beta,
            })
        
        
        df = pd.DataFrame(metrics)
        self._log(f"Peer-Relative Metrics calculated for {len(df)} funds")
        return df
    
    def _normalize_metrics(self, absolute_metrics: pd.DataFrame,
                           peer_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize all metrics to 0-100 percentile scores (category-local).
        
        Uses pandas.rank(pct=True) * 100.
        
        Ascending=True: Sharpe, Sortino, Rolling Return, Relative Return, Consistency.
        Ascending=False: Volatility, Beta Fit.
        
        Args:
            absolute_metrics: DataFrame with sharpe, sortino, volatility.
            peer_metrics: DataFrame with rolling_return, relative_return, consistency, beta.
        
        Returns:
            DataFrame with normalized scores for all metrics.
        """
        combined = absolute_metrics.merge(peer_metrics, on='fund_code')
        
        print(f"[PR-FQS NORM] Combined metrics shape: {combined.shape}")
        print(f"[PR-FQS NORM] Sharpe values: min={combined['sharpe'].min():.4f}, max={combined['sharpe'].max():.4f}, mean={combined['sharpe'].mean():.4f}")
        print(f"[PR-FQS NORM] Volatility values: min={combined['volatility'].min():.4f}, max={combined['volatility'].max():.4f}, mean={combined['volatility'].mean():.4f}")
        
        # Normalize ascending metrics (higher is better)
        combined['sharpe_norm'] = combined['sharpe'].rank(pct=True) * 100
        combined['sortino_norm'] = combined['sortino'].rank(pct=True) * 100
        combined['rolling_return_norm'] = combined['rolling_return'].rank(pct=True) * 100
        combined['relative_return_norm'] = combined['relative_return'].rank(pct=True) * 100
        combined['consistency_norm'] = combined['consistency'].rank(pct=True) * 100
        
        # Normalize volatility (lower is better) - invert ranking
        combined['volatility_norm'] = combined['volatility'].rank(pct=True, ascending=False) * 100
        
        # Beta fit: distance from category median (lower distance = better)
        median_beta = combined['beta'].median()
        combined['beta_distance'] = (combined['beta'] - median_beta).abs()
        combined['beta_fit_norm'] = combined['beta_distance'].rank(pct=True, ascending=False) * 100
        
        print(f"[PR-FQS NORM] Normalized Sharpe: min={combined['sharpe_norm'].min():.2f}, max={combined['sharpe_norm'].max():.2f}")
        print(f"[PR-FQS NORM] Normalized Volatility: min={combined['volatility_norm'].min():.2f}, max={combined['volatility_norm'].max():.2f}")
        
        self._log("Metrics normalized to 0-100 percentile scores")
        return combined
    
    def _calculate_pr_fqs_score(self, normalized_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate final PR-FQS score using weighted combination.
        
        PR_FQS = (0.20 * Sharpe) + (0.15 * Sortino) + (0.15 * Rolling_Return)
               + (0.15 * Relative_Return) + (0.15 * Consistency) 
               + (0.10 * Beta_Fit) + (0.10 * Volatility_Inv)
        
        Args:
            normalized_metrics: DataFrame with normalized metric scores.
        
        Returns:
            DataFrame with fund_code, pr_fqs_score, and all component metrics.
        """
        normalized_metrics['pr_fqs_score'] = (
            self.WEIGHTS['sharpe'] * normalized_metrics['sharpe_norm'] +
            self.WEIGHTS['sortino'] * normalized_metrics['sortino_norm'] +
            self.WEIGHTS['rolling_return'] * normalized_metrics['rolling_return_norm'] +
            self.WEIGHTS['relative_return'] * normalized_metrics['relative_return_norm'] +
            self.WEIGHTS['consistency'] * normalized_metrics['consistency_norm'] +
            self.WEIGHTS['beta_fit'] * normalized_metrics['beta_fit_norm'] +
            self.WEIGHTS['volatility'] * normalized_metrics['volatility_norm']
        )
        
        print(f"[PR-FQS SCORE] Raw PR-FQS before sorting: min={normalized_metrics['pr_fqs_score'].min():.2f}, max={normalized_metrics['pr_fqs_score'].max():.2f}, mean={normalized_metrics['pr_fqs_score'].mean():.2f}")
        print(f"[PR-FQS SCORE] Sample scores: {normalized_metrics[['fund_code', 'pr_fqs_score']].head(3).to_string()}")
        
        return normalized_metrics.sort_values('pr_fqs_score', ascending=False)
    
    def calculate_category_pr_fqs(self, category: str, 
                                  fund_list: Optional[Dict[str, pd.DataFrame]] = None) -> PRFQSResult:
        """
        Main entry point: Calculate PR-FQS for all funds in a category.
        
        Enforces the complete data pipeline with all gates and produces
        a ranked list of funds with PR-FQS scores.
        
        Args:
            category: SEBI category name (e.g., 'Large Cap', 'Mid Cap').
            fund_list: Optional pre-fetched dictionary of fund_code -> NAV DataFrame.
                      If None, fetches from mfapi.in.
        
        Returns:
            PRFQSResult object containing category, eligible count, validity flag,
            error message (if any), and ranked DataFrame (if valid).
        """
        self._log(f"\n{'='*80}")
        self._log(f"PR-FQS Calculation: {category}")
        self._log(f"Timestamp: {datetime.now()}")
        self._log(f"{'='*80}\n")
        
        # Step 1: Fetch fund data (or use provided list)
        if fund_list is None:
            funds_dict = self.fetch_category_funds(category)
        else:
            funds_dict = fund_list
        
        if not funds_dict:
            return PRFQSResult(
                category=category,
                eligible_fund_count=0,
                is_valid=False,
                error_message="No fund data available for category."
            )
        
        self._log(f"Initial fund count: {len(funds_dict)}")

        # Step 2: Apply 3-Year Gate
        cutoff_date = datetime.now() - timedelta(days=3*365)
        eligible_funds, dropped = self._apply_3year_gate(funds_dict, cutoff_date)
        
        # Step 3: Build Returns DataFrame (vectorized alignment first)
        returns_df = self._build_returns_dataframe(eligible_funds)
        
        if returns_df.empty or returns_df.shape[1] < 2:
            return PRFQSResult(
                category=category,
                eligible_fund_count=len(eligible_funds),
                is_valid=False,
                status="INSUFFICIENT_HISTORY",
                error_message=f"Insufficient data for calculation: {returns_df.shape}"
            )

        # Step 4: Apply Strict Sample Size Gate on Usable Aligned Series
        gate_passed, status_code, gate_msg = self._apply_sample_size_gate(returns_df)
        if not gate_passed:
            return PRFQSResult(
                category=category,
                eligible_fund_count=returns_df.shape[1],
                is_valid=False,
                status=status_code,
                error_message=gate_msg
            )
        
        # Step 5: Calculate Absolute Metrics
        absolute_metrics = self._calculate_absolute_metrics(returns_df)
        
        # Step 6: Calculate Peer-Relative Metrics (Leave-One-Out)
        peer_metrics = self._calculate_peer_relative_metrics(returns_df)
        
        # Step 7: Normalize Metrics to 0-100 Percentile Scores
        normalized = self._normalize_metrics(absolute_metrics, peer_metrics)
        
        # Step 8: Calculate PR-FQS Score
        ranked_funds = self._calculate_pr_fqs_score(normalized)
        
        # Select output columns
        output_columns = [
            'fund_code', 'pr_fqs_score',
            'sharpe_norm', 'sortino_norm', 'rolling_return_norm',
            'relative_return_norm', 'consistency_norm', 'beta_fit_norm', 'volatility_norm',
            'sharpe', 'sortino', 'volatility',
            'rolling_return', 'relative_return', 'consistency', 'beta'
        ]
        
        ranked_funds = ranked_funds[output_columns].reset_index(drop=True)
        ranked_funds['rank'] = range(1, len(ranked_funds) + 1)
        
        print(f"[PR-FQS FINAL] Rankings for category '{category}':")
        print(ranked_funds[['rank', 'fund_code', 'pr_fqs_score']].to_string())
        print(f"[PR-FQS FINAL] ===============================================\n")
        
        self._log(f"\nFinal Rankings ({category}):")
        self._log(ranked_funds[['rank', 'fund_code', 'pr_fqs_score']].to_string())
        self._log(f"\n{'='*80}\n")
        
        return PRFQSResult(
            category=category,
            eligible_fund_count=returns_df.shape[1],
            is_valid=True,
            status="VALID",
            error_message=None,
            ranked_funds=ranked_funds
        )


# Module-level convenience function
def calculate_category_pr_fqs(category_name: str, fund_list: Optional[Dict] = None,
                              verbose: bool = False) -> PRFQSResult:
    """
    Convenience function to calculate PR-FQS for a category.
    
    Args:
        category_name: SEBI category name.
        fund_list: Optional pre-fetched fund data.
        verbose: Enable debug logging.
    
    Returns:
        PRFQSResult with ranked funds and metadata.
    """
    engine = PRFQSEngine(verbose=verbose)
    return engine.calculate_category_pr_fqs(category_name, fund_list)


if __name__ == "__main__":
    # Example usage (requires live fund data integration)
    engine = PRFQSEngine(verbose=True)
    
    # Example: Process a category with placeholder data
    print("PR-FQS Engine initialized. Ready for production integration.")
    print(f"Weights: {engine.WEIGHTS}")
    print(f"Minimum eligible funds: {engine.MIN_ELIGIBLE_FUNDS}")
    print(f"Data window: {engine.DATA_YEARS} years")
