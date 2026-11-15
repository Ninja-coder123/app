"""Mutual-fund categorization and portfolio health scoring."""

import pandas as pd


def categorize_mf(name: str) -> str:
    """Categorize mutual fund by name. Added Gold/Silver MF detection."""
    debt_keywords = ["debt", "liquid", "bond", "gilt", "corporate", "dynamic asset"]
    equity_keywords = ["equity", "midcap", "smallcap", "largecap", "bluechip", "flexi", 
        "index", "multicap", "multi cap", "mid cap", "small cap", "large cap",
        "contra", "value", "elss", "tax saver","Balanced Advantage", "tax saving", "focused", 
        "nifty", "sensex", "opportunities", "quant", "microcap", "infrastructure",
        "tech", "pharma", "consumption", "dividend yield", "thematic"]
    gold_keywords = ["gold", "silver", "precious"]
    
    nm = name.lower()
    
    if any(k.lower() in nm for k in gold_keywords):
        return "Gold/Silver MF"
    if any(k.lower() in nm for k in debt_keywords):
        return "Debt MF"
    if any(k.lower() in nm for k in equity_keywords):
        return "Equity MF"
    return "Unclassified MF"


def compute_health_score(balances: dict, equity_count: int, risk_profile: str, funds_df: pd.DataFrame, 
                         fri_df: pd.DataFrame, physical_gold_df: pd.DataFrame):
    """Compute health score using the original 11-point penalty engine."""
    total = sum(balances.values())
    score = 100
    penalties = []
    
    if total <= 0:
        return 0, [('No Assets', 'No assets found - score is 0')], []
    
    target_ranges = {
        'Ultra-Conservative': (0, 15),
        'Conservative': (20, 30),
        'Balanced': (40, 60),
        'Aggressive': (70, 85),
        'Maximum Growth': (90, 100)
    }
    
    equity_pct = (balances.get('Equity MF', 0) / total * 100) if total > 0 else 0
    target_min, target_max = target_ranges.get(risk_profile, (40, 60))
    if equity_pct < target_min - 10 or equity_pct > target_max + 10:
        score -= 15
        penalties.append(('Style Mismatch', f'Equity {equity_pct:.1f}% outside target {target_min}-{target_max}% for {risk_profile}'))
    
    fri_pct = (balances.get('FRI', 0) / total * 100) if total > 0 else 0
    if fri_pct > 40 and risk_profile in ['Aggressive', 'Maximum Growth']:
        score -= 10
        penalties.append(('Cash Drag', f'Fixed Return Instruments {fri_pct:.1f}% > 40% for {risk_profile}'))
    
    if balances.get('Equity MF', 0) == 0 and risk_profile != 'Ultra-Conservative':
        score -= 15
        penalties.append(('Zero Equity Trap', f'Total Equity is 0% but profile is {risk_profile}'))
    
    gold_silver_total = balances.get('Gold/Silver MF', 0)
    if not physical_gold_df.empty:
        gold_silver_total += physical_gold_df['value'].sum()
    gold_silver_pct = (gold_silver_total / total * 100) if total > 0 else 0
    if gold_silver_pct > 15:
        score -= 10
        penalties.append(('Heavy Metal Drag', f'Gold/Silver {gold_silver_pct:.1f}% > 15% of portfolio'))
    
    if equity_count > 7:
        score -= 10
        penalties.append(('Equity Over-Diversification', f'{equity_count} Equity MFs > 7'))
    
    debt_count = len(funds_df[funds_df['category'] == 'Debt MF']) if not funds_df.empty else 0
    if debt_count > 3:
        score -= 5
        penalties.append(('Debt Over-Diversification', f'{debt_count} Debt MFs > 3'))
    
    micro_threshold = total * 0.07
    has_micro = False
    if not funds_df.empty:
        micro_funds = funds_df[(funds_df['value'] > 0) & (funds_df['value'] < micro_threshold)]
        if not micro_funds.empty:
            has_micro = True
    if not fri_df.empty:
        micro_fri = fri_df[(fri_df['amount'] > 0) & (fri_df['amount'] < micro_threshold)]
        if not micro_fri.empty:
            has_micro = True
    if not physical_gold_df.empty:
        micro_gold = physical_gold_df[(physical_gold_df['value'] > 0) & (physical_gold_df['value'] < micro_threshold)]
        if not micro_gold.empty:
            has_micro = True
    if has_micro:
        score -= 5
        penalties.append(('Micro-Allocation', f'One or more holdings < 7% of total ({micro_threshold:.0f})'))
    
    if not funds_df.empty:
        equity_mfs = funds_df[funds_df['category'] == 'Equity MF']
        if not equity_mfs.empty:
            equity_total = equity_mfs['value'].sum()
            small_mid = equity_mfs[equity_mfs['name'].str.lower().str.contains('small|mid', na=False)]['value'].sum()
            if equity_total > 0:
                small_mid_pct = (small_mid / equity_total * 100)
                threshold = 50 if risk_profile == 'Maximum Growth' else 35
                if small_mid_pct > threshold:
                    score -= 15
                    penalties.append(('Small/Mid-Cap Overload', f'Small/Mid-Cap {small_mid_pct:.1f}% > {threshold}% of Equity bucket'))
    
    if balances.get('Debt MF', 0) == 0:
        score -= 15
        penalties.append(('Zero-Liquidity Danger', 'Total Debt MFs = 0 (high concentration risk)'))
    
    liquid_portfolio = balances.get('Equity MF', 0) + balances.get('Debt MF', 0) + balances.get('Gold/Silver MF', 0)
    if liquid_portfolio > 0 and not funds_df.empty:
        max_fund_value = funds_df['value'].max()
        max_fund_pct = (max_fund_value / liquid_portfolio * 100)
        if max_fund_pct > 30:
            max_fund_name = funds_df[funds_df['value'] == max_fund_value]['name'].values[0] if len(funds_df) > 0 else "Unknown"
            score -= 15
            penalties.append(('Single-Fund Risk', f'{max_fund_name} is {max_fund_pct:.1f}% of liquid portfolio (>30%)'))
    
    re_pct = (balances.get('Real Estate', 0) / total * 100) if total > 0 else 0
    if re_pct > 75:
        score -= 10
        penalties.append(('Real Estate Lock', f'Real Estate {re_pct:.1f}% > 75% of portfolio'))
    
    score = max(0, int(score))
    recommendations = []
    if any(p[0] == 'Style Mismatch' for p in penalties):
        recommendations.append(f"Rebalance equity allocation to match your {risk_profile} profile target range.")
    if any(p[0] == 'Cash Drag' for p in penalties):
        recommendations.append("Reduce Fixed Return Instruments to increase growth potential for your aggressive profile.")
    if any(p[0] == 'Zero Equity Trap' for p in penalties):
        recommendations.append("Add equity exposure suitable for your risk profile to achieve growth.")
    if any(p[0] == 'Heavy Metal Drag' for p in penalties):
        recommendations.append("Consider reducing Gold/Silver allocation to below 15% of your portfolio.")
    if any(p[0] == 'Equity Over-Diversification' for p in penalties):
        recommendations.append("Consolidate similar Equity MFs to reduce fees and simplify management.")
    if any(p[0] == 'Debt Over-Diversification' for p in penalties):
        recommendations.append("Reduce number of Debt MFs to improve focus and reduce fees.")
    if any(p[0] == 'Micro-Allocation' for p in penalties):
        recommendations.append("Review small holdings; consider consolidating or removing positions below 7% allocation.")
    if any(p[0] == 'Small/Mid-Cap Overload' for p in penalties):
        recommendations.append("Rebalance to reduce concentration in small-cap and mid-cap equities.")
    if any(p[0] == 'Zero-Liquidity Danger' for p in penalties):
        recommendations.append("Add Debt MF holdings for short-term liquidity and emergency reserves.")
    if any(p[0] == 'Single-Fund Risk' for p in penalties):
        recommendations.append("Diversify to reduce single-fund concentration below 30% of liquid portfolio.")
    if any(p[0] == 'Real Estate Lock' for p in penalties):
        recommendations.append("Diversify into liquid assets; Real Estate exceeds 75% threshold.")
    
    return score, penalties, recommendations
