"""
Wealth Management SaaS Backend Physics Engine.

A complete, parameter-driven, mathematically rigorous engine for financial planning,
portfolio allocation, tax optimization, and goal-based wealth management.
"""

from typing import Dict, Any, Optional, Tuple
import math
import random
from backend.schemas import MarketAssumptions, Goal, PortfolioAllocation, TaxConfig


# ============================================================================
# 1. GOAL-SPECIFIC GLIDEPATH ENGINE
# ============================================================================

def calculate_recommended_allocation(goal: Goal) -> PortfolioAllocation:
    """
    Calculate recommended portfolio allocation based on goal timeline and risk capacity.
    Mathematical mapping: $$RecommendedEquity = f(Time, Criticality, RiskCapacity)$$
    """
    if goal.years_to_goal >= 10.0:
        equity_pct = 85.0
    else:
        equity_pct = 85.0 * ((goal.years_to_goal / 10.0) ** 1.5)

    if goal.is_critical:
        equity_pct -= 15.0

    if goal.risk_capacity == "Low" or goal.is_critical:
        equity_pct = min(equity_pct, 40.0)
    elif goal.risk_capacity == "High":
        equity_pct = min(equity_pct + 10.0, 95.0)

    equity_pct = max(10.0, equity_pct)
    debt_pct = 100.0 - equity_pct

    return PortfolioAllocation(equity_pct=equity_pct, debt_pct=debt_pct)


# ============================================================================
# 3. STEP-UP SIP COMPOUNDING ENGINE
# ============================================================================

def calculate_step_up_sip_trajectory(
    initial_sip: float,
    allocation: PortfolioAllocation,
    assumptions: MarketAssumptions,
    years: int,
    current_equity: float = 0.0,
    current_debt: float = 0.0,
) -> Dict[str, Any]:
    """
    Calculate SIP trajectory with step-up and compound growth, including existing corpus.
    
    Computes MONTHLY accumulation of step-up SIP contributions with portfolio returns,
    plus compounds the existing equity and debt holdings separately.
    
    Args:
        initial_sip: Initial monthly SIP amount (in ₹).
        allocation: Portfolio allocation (equity/debt split).
        assumptions: Market assumptions (returns, step-up rate).
        years: Investment horizon (number of years).
        current_equity: Existing equity corpus to be compounded (in ₹).
        current_debt: Existing debt corpus to be compounded (in ₹).
        
    Returns:
        Dictionary containing complete corpus and gains breakdown.
    """
    # Monthly returns
    i_blended = (1.0 + (
        (allocation.equity_pct / 100.0) * assumptions.equity_return +
        (allocation.debt_pct / 100.0) * assumptions.debt_return
    )) ** (1.0 / 12.0) - 1.0
    
    i_equity = (1.0 + assumptions.equity_return) ** (1.0 / 12.0) - 1.0
    i_debt = (1.0 + assumptions.debt_return) ** (1.0 / 12.0) - 1.0
    base_equity_monthly_return = i_equity
    equity_weight = allocation.equity_pct / 100.0
    debt_weight = allocation.debt_pct / 100.0
    
    total_months = int(years * 12)
    
    # Compound existing corpus
    fv_existing_equity = current_equity * ((1.0 + i_equity) ** total_months)
    fv_existing_debt = current_debt * ((1.0 + i_debt) ** total_months)
    
    # Monthly SIP projection
    projected_corpus = current_equity + current_debt
    fv_equity_sip = 0.0
    fv_debt_sip = 0.0
    total_sip_invested = 0.0
    initial_corpus = current_equity + current_debt
    year_by_year = [{
        "year": 0,
        "sip": 0.0,
        "corpus_start": projected_corpus,
        "corpus_end": projected_corpus,
        "principal_invested": initial_corpus,
        "gains":0.0,
    }]
    annual_corpus = 0.0
    monthly_trajectory = []
    
    for month in range(1, total_months + 1):
        # Determine SIP amount (step up yearly)
        year_num = (month - 1) // 12
        monthly_sip = initial_sip * (1.0 + assumptions.step_up_rate) ** year_num

        # Calculate the theoretical expected return
        expected_blended_return = (equity_weight * base_equity_monthly_return) + (debt_weight * i_debt)

        # Calculate dynamic standard deviation (sigma)
        # Equity gets high variance (4% monthly), Debt gets extremely low variance (0.2% monthly)
        dynamic_sigma = (equity_weight * 0.04) + (debt_weight * 0.002)

        # Inject the Gaussian volatility directly into the blended return
        blended_monthly_return = expected_blended_return + random.gauss(0.0, dynamic_sigma)
        
        # Apply growth to current corpus
        projected_corpus *= (1.0 + blended_monthly_return)
        
        # Add new SIP contribution
        projected_corpus += monthly_sip
        total_sip_invested += monthly_sip
        
        # Track annually
        annual_corpus = projected_corpus
        if month % 12 == 0:
            year_num = month // 12
            annual_principal_invested = total_sip_invested
            gains_sip = projected_corpus - (current_equity + current_debt) - annual_principal_invested
            year_by_year.append({
                "year": year_num,
                "sip": (initial_sip * (1.0 + assumptions.step_up_rate) ** (year_num - 1)) * 12,
                "corpus_start": annual_corpus - (monthly_sip * (1.0 + blended_monthly_return)),
                "corpus_end": annual_corpus,
                "principal_invested": annual_principal_invested,
                "gains": gains_sip,
            })
        monthly_trajectory.append({
            "Month": month,
            "Gross Corpus": projected_corpus,
            "Principal Invested": total_sip_invested + current_equity + current_debt,
        })
    
    # Calculate component FVs
    # FV of equity SIPs: allocate each month's SIP and compound it
    temp_corpus = 0.0
    for month in range(1, total_months + 1):
        year_num = (month - 1) // 12
        monthly_sip = initial_sip * (1.0 + assumptions.step_up_rate) ** year_num
        equity_portion = monthly_sip * (allocation.equity_pct / 100.0)
        months_remaining = total_months - month
        temp_corpus += equity_portion * ((1.0 + i_equity) ** months_remaining)
    fv_equity_sip = temp_corpus
    
    # FV of debt SIPs
    temp_corpus = 0.0
    for month in range(1, total_months + 1):
        year_num = (month - 1) // 12
        monthly_sip = initial_sip * (1.0 + assumptions.step_up_rate) ** year_num
        debt_portion = monthly_sip * (allocation.debt_pct / 100.0)
        months_remaining = total_months - month
        temp_corpus += debt_portion * ((1.0 + i_debt) ** months_remaining)
    fv_debt_sip = temp_corpus
    
    # Total future values
    total_fv_equity = fv_existing_equity + fv_equity_sip
    total_fv_debt = fv_existing_debt + fv_debt_sip
    total_corpus = total_fv_equity + total_fv_debt
    
    # Principal invested = existing + SIPs
    principal_invested = current_equity + current_debt + total_sip_invested
    
    # Capital gains = final corpus - principal
    capital_gains = total_corpus - principal_invested
    
    # Cost basis for tax calculations
    cost_equity_total = current_equity + (total_sip_invested * (allocation.equity_pct / 100.0))
    cost_debt_total = current_debt + (total_sip_invested * (allocation.debt_pct / 100.0))
    
    return {
        "corpus_end": total_corpus,
        "principal_invested": principal_invested,
        "gains": capital_gains,
        "fv_equity_existing": fv_existing_equity,
        "fv_debt_existing": fv_existing_debt,
        "fv_equity_sip": fv_equity_sip,
        "fv_debt_sip": fv_debt_sip,
        "cost_equity_total": cost_equity_total,
        "cost_debt_total": cost_debt_total,
        "year_by_year": year_by_year,
        "monthly_trajectory": monthly_trajectory,
    }


# ============================================================================
# 4. ITERATIVE GROSS-UP TAX SOLVER (BINARY SEARCH)
# ============================================================================

def solve_gross_liquidation(
    target_net: float,
    fv_equity: float,
    cost_equity: float,
    fv_debt: float,
    cost_debt: float,
    tax_config: TaxConfig,
) -> Dict[str, Any]:
    """
    Solve for gross liquidation amount required to achieve target net proceeds.
    
    Uses iterative binary search to find the exact gross sale amount that,
    after taxes, yields the desired net proceeds.
    
    Args:
        target_net: Target net proceeds after taxes (in ₹).
        fv_equity: Future value of equity portfolio (in ₹).
        cost_equity: Cost basis of equity portfolio (in ₹).
        fv_debt: Future value of debt portfolio (in ₹).
        cost_debt: Cost basis of debt portfolio (in ₹).
        tax_config: Tax configuration (rates and exemptions).
        
    Returns:
        Dictionary containing:
            - gross_required: Gross sale amount needed.
            - net_received: Actual net proceeds (should match target ±₹1).
            - total_tax: Total tax liability.
            - equity_tax: Tax on equity portion.
            - debt_tax: Tax on debt portion.
            - portfolio_insufficient: True if target exceeds maximum liquidation value.
    """
    # Edge case: target_net <= 0
    if target_net <= 0:
        return {
            "gross_required": 0.0,
            "net_received": 0.0,
            "total_tax": 0.0,
            "equity_tax": 0.0,
            "debt_tax": 0.0,
            "portfolio_insufficient": False,
        }
    
    total_fv = fv_equity + fv_debt
    
    # Compute gain ratios
    gain_ratio_equity = 0.0
    if fv_equity > 1e-10:
        gain_ratio_equity = max(0.0, (fv_equity - cost_equity) / fv_equity)
    
    gain_ratio_debt = 0.0
    if fv_debt > 1e-10:
        gain_ratio_debt = max(0.0, (fv_debt - cost_debt) / fv_debt)
    
    def net_achieved_after_tax(gross_sale: float) -> float:
        """Calculate net proceeds for a given gross sale amount."""
        if total_fv < 1e-10:
            return 0.0
        
        # Pro-rata split
        sell_equity = gross_sale * (fv_equity / total_fv) if total_fv > 0 else 0.0
        sell_debt = gross_sale * (fv_debt / total_fv) if total_fv > 0 else 0.0
        
        # Realized gains
        realized_gain_equity = sell_equity * gain_ratio_equity
        realized_gain_debt = sell_debt * gain_ratio_debt
        
        # Taxes
        tax_equity = max(
            0.0,
            (realized_gain_equity - tax_config.equity_exemption) *
            tax_config.equity_ltcg_rate
        )
        tax_debt = realized_gain_debt * tax_config.debt_slab_rate
        
        total_tax = tax_equity + tax_debt
        net = gross_sale - total_tax
        
        return net
    
    max_net = net_achieved_after_tax(total_fv)
    portfolio_insufficient = target_net > max_net

    # Remove any artificial cap on the required gross amount. The theoretical
    # gross sale required to reach a target net amount may exceed the current
    # portfolio value when the goal requires more than the plan can produce.
    lower_bound = 0.0
    upper_bound = max(total_fv, target_net, 1.0)
    safety_iterations = 0
    while net_achieved_after_tax(upper_bound) < target_net and safety_iterations < 100:
        upper_bound *= 2.0
        safety_iterations += 1

    tolerance = 1.0  # ₹1 precision
    while upper_bound - lower_bound > tolerance:
        mid = (lower_bound + upper_bound) / 2.0
        net = net_achieved_after_tax(mid)
        
        if net < target_net:
            lower_bound = mid
        else:
            upper_bound = mid
    
    gross_required = (lower_bound + upper_bound) / 2.0
    
    # Final calculation with the determined gross amount
    if total_fv > 1e-10:
        sell_equity = gross_required * (fv_equity / total_fv)
        sell_debt = gross_required * (fv_debt / total_fv)
    else:
        sell_equity = 0.0
        sell_debt = 0.0
    
    realized_gain_equity = sell_equity * gain_ratio_equity
    realized_gain_debt = sell_debt * gain_ratio_debt
    
    equity_tax = max(
        0.0,
        (realized_gain_equity - tax_config.equity_exemption) *
        tax_config.equity_ltcg_rate
    )
    debt_tax = realized_gain_debt * tax_config.debt_slab_rate
    total_tax = equity_tax + debt_tax
    
    net_received = gross_required - total_tax
    
    return {
        "gross_required": gross_required,
        "net_received": net_received,
        "total_tax": total_tax,
        "equity_tax": equity_tax,
        "debt_tax": debt_tax,
        "portfolio_insufficient": portfolio_insufficient,
    }


# ============================================================================
# 5. CORRECTIVE ACTION ENGINE
# ============================================================================

def calculate_course_corrections(
    shortfall: float,
    goal: Goal,
    assumptions: MarketAssumptions,
    current_sip: float,
) -> Dict[str, Any]:
    """
    Calculate corrective actions to bridge a corpus shortfall.
    
    Determines two corrective strategies:
    1. Additional monthly SIP needed to bridge the gap.
    2. Additional months of compounding needed for the existing path to hit target.
    
    Args:
        shortfall: Shortfall amount (target - current corpus).
        goal: Goal specification with timeline and target.
        assumptions: Market assumptions for projections.
        current_sip: Current monthly SIP amount.
        
    Returns:
        Dictionary containing:
            - required_extra_sip: Additional monthly SIP needed.
            - required_extra_sip_monthly: Clarification of the above.
            - required_delay_months: Extra months of compounding needed.
            - is_feasible: Whether the shortfall can be bridged.
    """
    if shortfall <= 0:
        return {
            "required_extra_sip": 0.0,
            "required_extra_sip_monthly": 0.0,
            "required_delay_months": 0,
            "is_feasible": True,
        }
    
    # Strategy 1: Additional SIP to bridge gap
    # Calculate the future value multiplier for a monthly SIP over remaining years
    r_blended = assumptions.equity_return * 0.5 + assumptions.debt_return * 0.5  # Neutral assumption
    i = (1.0 + r_blended) ** (1.0 / 12.0) - 1.0
    annual_growth = (1.0 + i) ** 12
    
    remaining_years = max(1, goal.years_to_goal)
    
    # Future value of 1 rupee SIP over remaining years
    if i > 1e-10:
        fv_factor = ((annual_growth - 1.0) / i) * (annual_growth ** (remaining_years - 1)) if remaining_years > 0 else 1.0
    else:
        fv_factor = remaining_years * 12
    
    # Ensure fv_factor is at least 1
    fv_factor = max(1.0, fv_factor)
    
    # Required extra SIP = shortfall / fv_factor
    required_extra_sip_monthly = shortfall / max(1.0, fv_factor)
    
    # Strategy 2: Additional months of compounding
    # Estimate how many additional months are needed for growth to cover shortfall
    monthly_return = i
    
    # Simple approximation: shortfall / (current_sip * monthly_return)
    # This is a rough estimate; the actual calculation depends on monthly compounding
    if current_sip > 1e-10 and monthly_return > 1e-10:
        # Rough estimate: months needed for interest to cover shortfall
        required_delay_months = int(math.ceil(shortfall / (current_sip * monthly_return * 12)))
    else:
        required_delay_months = 0
    
    is_feasible = True  # Mathematically, any shortfall can be bridged with enough time/SIP
    
    return {
        "required_extra_sip": required_extra_sip_monthly,
        "required_extra_sip_monthly": required_extra_sip_monthly,
        "required_delay_months": required_delay_months,
        "is_feasible": is_feasible,
    }


# ============================================================================
# CONVENIENCE WRAPPERS & ANALYSIS FUNCTIONS
# ============================================================================

def full_goal_analysis(
    goal: Goal,
    initial_sip: float,
    assumptions: MarketAssumptions,
    tax_config: Optional[TaxConfig] = None,
    current_equity: float = 0.0,
    current_debt: float = 0.0,
) -> Dict[str, Any]:
    """
    Perform a complete analysis for a single goal.
    
    Orchestrates the entire physics engine: allocation recommendation,
    SIP trajectory projection, and tax impact analysis.
    
    Args:
        goal: Goal specification.
        initial_sip: Initial monthly SIP (in ₹).
        assumptions: Market assumptions.
        tax_config: Tax configuration (uses defaults if None).
        current_equity: Existing equity corpus (in ₹).
        current_debt: Existing debt corpus (in ₹).
        
    Returns:
        Comprehensive analysis dictionary with allocation, trajectory, and projections.
    """
    if tax_config is None:
        tax_config = TaxConfig()
    
    # Step 1: Determine allocation
    allocation = calculate_recommended_allocation(goal)
    
    # Step 2: Project SIP trajectory
    years = int(goal.years_to_goal)
    if years < 1:
        years = 1
    
    trajectory = calculate_step_up_sip_trajectory(
        initial_sip,
        allocation,
        assumptions,
        years,
        current_equity=current_equity,
        current_debt=current_debt,
    )
    
    final_corpus = trajectory["corpus_end"]
    shortfall = max(0, goal.target_amount - final_corpus)
    
    # Step 3: Corrective actions if shortfall
    corrections = {}
    if shortfall > 0:
        corrections = calculate_course_corrections(
            shortfall, goal, assumptions, initial_sip
        )
    
# Step 4: Tax analysis (if liquidation at end)
    fv_equity = trajectory["fv_equity_existing"] + trajectory["fv_equity_sip"]
    fv_debt = trajectory["fv_debt_existing"] + trajectory["fv_debt_sip"]
    equity_principal = trajectory["cost_equity_total"]
    debt_principal = trajectory["cost_debt_total"]
    
    inflated_target = goal.target_amount * ((1.0 + assumptions.inflation_rate) ** goal.years_to_goal)
    
    tax_analysis = solve_gross_liquidation(
        target_net=inflated_target,
        fv_equity=fv_equity,
        cost_equity=equity_principal,
        fv_debt=fv_debt,
        cost_debt=debt_principal,
        tax_config=tax_config,
    )
    return {
        "goal": {
            "target_amount": goal.target_amount,
            "years_to_goal": goal.years_to_goal,
            "is_critical": goal.is_critical,
        },
        "allocation": {
            "equity_pct": allocation.equity_pct,
            "debt_pct": allocation.debt_pct,
        },
        "sip": {
            "initial_monthly": initial_sip,
            "step_up_rate": assumptions.step_up_rate,
        },
        "projection": {
            "final_corpus": final_corpus,
            "principal_invested": trajectory["principal_invested"],
            "capital_gains": trajectory["gains"],
        },
        "shortfall": {
            "amount": shortfall,
            "required_extra_sip": corrections.get("required_extra_sip", 0.0),
            "required_extra_months": corrections.get("required_delay_months", 0),
        },
        "tax_on_liquidation": tax_analysis,
        "feasibility": shortfall == 0 or corrections.get("is_feasible", False),
    }
def calculate_plan_resilience(
    current_equity: float,
    current_debt: float,
    sip: float,
    goal: Goal,
    assumptions: MarketAssumptions,
    tax_config: TaxConfig
) -> Dict[str, Any]:
    """
    Simulates stress vectors to evaluate if the plan survives severe deviations.
    """
    allocation = calculate_recommended_allocation(goal)
    years = max(1, int(goal.years_to_goal))
    
    # Vector 1: Inflation Shock (+2% added to user's base inflation assumption)
    shocked_inflation = assumptions.inflation_rate + 0.02
    v1_target = goal.target_amount * ((1.0 + shocked_inflation) ** years)
    
    # Create shocked assumptions object
    shocked_assumptions = MarketAssumptions(
        inflation_rate=shocked_inflation,
        step_up_rate=assumptions.step_up_rate,
        equity_return=assumptions.equity_return,
        debt_return=assumptions.debt_return
    )
    
    v1_traj = calculate_step_up_sip_trajectory(sip, allocation, shocked_assumptions, years, current_equity, current_debt)
    v1_tax = solve_gross_liquidation(v1_target, v1_traj["fv_equity_existing"] + v1_traj["fv_equity_sip"], v1_traj["cost_equity_total"], v1_traj["fv_debt_existing"] + v1_traj["fv_debt_sip"], v1_traj["cost_debt_total"], tax_config)
    # Vector 2: Market Crash (-30% current equity)
    v2_traj = calculate_step_up_sip_trajectory(sip, allocation, assumptions, years, current_equity * 0.70, current_debt)
    v2_tax = solve_gross_liquidation(goal.target_amount, v2_traj["fv_equity_existing"] + v2_traj["fv_equity_sip"], v2_traj["cost_equity_total"], v2_traj["fv_debt_existing"] + v2_traj["fv_debt_sip"], v2_traj["cost_debt_total"], tax_config)
    
    # Vector 3: Income Disruption (Pause SIP for 12 months)
    v3_traj = calculate_step_up_sip_trajectory(sip, allocation, assumptions, years, current_equity, current_debt)
    # Approximate 12 months of missed SIP compounding from Year 2
    if years >= 2:
        missed_sip_fv = (sip * 12) * ((1 + assumptions.equity_return) ** (years - 2)) 
        v3_corpus = max(0.0, v3_traj["corpus_end"] - missed_sip_fv)
    else:
        v3_corpus = v3_traj["corpus_end"]
    
    return {
        "V1_Inflation_Shock": {
            "Final Corpus": v1_traj["corpus_end"],
            "Gross Requirement": v1_tax["gross_required"],
            "survives_stress": v1_traj["corpus_end"] >= v1_tax["gross_required"]
        },
        "V2_Market_Crash": {
            "Final Corpus": v2_traj["corpus_end"],
            "Gross Requirement": v2_tax["gross_required"],
            "survives_stress": v2_traj["corpus_end"] >= v2_tax["gross_required"]
        },
        "V3_Income_Disruption": {
            "Final Corpus": v3_corpus,
            "Gross Requirement": tax_config.equity_exemption, # Simplified threshold check
            "survives_stress": v3_corpus >= goal.target_amount 
        }
    }