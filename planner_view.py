"""
Wealth Management SaaS Frontend: Planner View.

A dumb frontend following strict "Dumb Frontend / Smart Backend" architecture.
All financial calculations are delegated to backend/math_engine.py.
This module is strictly a UI layer: it collects inputs, calls backend methods,
and visualizes results using Streamlit.
"""

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Import backend data structures and functions
from backend.schemas import (
    MarketAssumptions,
    Goal,
    PortfolioAllocation,
    TaxConfig,
)
from backend.math_engine import (
    calculate_recommended_allocation,
    calculate_step_up_sip_trajectory,
    solve_gross_liquidation,
    calculate_course_corrections,
    full_goal_analysis,
    calculate_plan_resilience,
)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize session state to prevent widget reset on script reruns."""
    defaults = {
        # Goal Settings
        "goal_target": 5000000,  # ₹50 lakhs
        "goal_years": 10,
        "goal_is_critical": False,
        # Current Portfolio
        "current_equity": 1000000,  # ₹10 lakhs
        "current_debt": 500000,  # ₹5 lakhs
        "initial_sip": 50000,  # ₹50k monthly
        # Market Assumptions
        "preset_name": "Normal",
        "inflation_rate": 0.06,
        "step_up_rate": 0.07,
        "equity_return": 0.12,
        "debt_return": 0.07,
        # Asset Allocation
        "use_smart_glidepath": True,
        "manual_equity_pct": 60.0,
        # Tax Config
        "equity_ltcg_rate": 0.125,
        "equity_exemption": 125000.0,
        "debt_slab_rate": 0.30,
        "engine_triggered": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_health_score(
    current_equity: float,
    net_achieved: float,
    target_net: float,
    years_remaining: float,
    is_critical: bool,
) -> int:
    """
    Calculate a Health Score (0-100) based on Asset Allocation and Funding Risks.
    """
    if target_net <= 0:
        return 0

    # 1. Asset Allocation Penalty (Max 50 points)
    recommended_equity = 85.0 if years_remaining >= 10.0 else 85.0 * ((years_remaining / 10.0) ** 1.5)
    if is_critical:
        recommended_equity -= 15.0
    recommended_equity = max(10.0, recommended_equity)

    excess_equity = max(0.0, current_equity - recommended_equity)
    allocation_penalty = min(excess_equity * 1.5, 50.0)

    # 2. Funding Shortfall Penalty (Max 50 points)
    funding_ratio = min(net_achieved / target_net, 1.0)
    shortfall_pct = 1.0 - funding_ratio
    funding_penalty = min(shortfall_pct * 100.0, 50.0)

    # 3. Final Score Calculation
    return int(max(0.0, 100.0 - allocation_penalty - funding_penalty))


def format_currency(value: float) -> str:
    """
    Format a number as Indian currency with proper decimal handling and sign placement.
    
    Examples: ₹50.01L, ₹50.01k, -₹28.50k (NOT ₹-28.50k)
    """
    is_negative = value < 0
    abs_value = abs(value)
    
    if abs_value >= 1_00_00_000:  # Crore
        formatted = f"₹{abs_value / 1_00_00_000:.2f}Cr"
    elif abs_value >= 1_00_000:  # Lakh
        formatted = f"₹{abs_value / 1_00_000:.2f}L"
    elif abs_value >= 1_000:  # Thousand
        formatted = f"₹{abs_value / 1_000:.2f}k"
    else:
        formatted = f"₹{abs_value:.2f}"
    
    # Apply negative sign at the start
    return f"-{formatted}" if is_negative else formatted


def get_market_assumptions() -> MarketAssumptions:
    """Build MarketAssumptions from session state."""
    return MarketAssumptions(
        inflation_rate=st.session_state.inflation_rate,
        step_up_rate=st.session_state.step_up_rate,
        equity_return=st.session_state.equity_return,
        debt_return=st.session_state.debt_return,
    )


def get_goal() -> Goal:
    """Build Goal from session state."""
    return Goal(
        target_amount=st.session_state.goal_target,
        years_to_goal=st.session_state.goal_years,
        is_critical=st.session_state.goal_is_critical,
    )


def get_tax_config() -> TaxConfig:
    """Build TaxConfig from session state."""
    return TaxConfig(
        equity_ltcg_rate=st.session_state.equity_ltcg_rate,
        equity_exemption=st.session_state.equity_exemption,
        debt_slab_rate=st.session_state.debt_slab_rate,
    )


def get_allocation() -> PortfolioAllocation:
    """Get portfolio allocation (smart or manual)."""
    if st.session_state.use_smart_glidepath:
        goal = get_goal()
        return calculate_recommended_allocation(goal)
    else:
        equity = st.session_state.manual_equity_pct
        debt = 100.0 - equity
        return PortfolioAllocation(equity_pct=equity, debt_pct=debt)


# ============================================================================
# LEFT COLUMN: CONTROL ROOM
# ============================================================================

def render_control_room(col_left):
    """Render input controls in the left column."""
    with col_left:
        st.subheader("🎯 Control Room")
        with st.form("engine_controls"):
            with st.expander("📊 Goal Settings", expanded=True):
                st.number_input(
                    "Target Net Amount (₹)",
                    min_value=10_000,
                    max_value=100_000_000,
                    step=100_000,
                    key="goal_target",
                    help="The corpus you want to accumulate by the goal date.",
                )
                st.number_input(
                    "Years to Goal",
                    min_value=0,
                    max_value=50,
                    step=1, 
                    key="goal_years",
                    help="Timeline to achieve the goal (affects allocation).",
                )
                st.checkbox(
                    "🚨 Is this a Critical Goal?",
                    key="goal_is_critical",
                    help="Critical goals reduce equity allocation by 10% as a safety buffer.",
                )

            with st.expander("💼 Current Portfolio", expanded=True):
                st.session_state.current_equity = st.number_input(
                    "Current Invested Equity (₹)",
                    min_value=0,
                    value=st.session_state.current_equity,
                    step=50_000,
                )
                st.session_state.current_debt = st.number_input(
                    "Current Invested Debt (₹)",
                    min_value=0,
                    value=st.session_state.current_debt,
                    step=25_000,
                )
                current_total = st.session_state.current_equity + st.session_state.current_debt
                st.caption(f"Total Current Corpus: {format_currency(current_total)}")

            with st.expander("💰 SIP Settings", expanded=True):
                st.session_state.initial_sip = st.number_input(
                    "Initial Monthly SIP (₹)",
                    min_value=100,
                    max_value=500_000,
                    value=st.session_state.initial_sip,
                    step=100,
                    help="Monthly systematic investment plan.",
                )

            with st.expander("📈 Market Assumptions", expanded=False):
                selected_preset = st.selectbox(
                    "Quick Preset",
                    options=["Conservative", "Normal", "Strong", "Custom"],
                    index=0 if st.session_state.preset_name == "Conservative"
                    else (1 if st.session_state.preset_name == "Normal"
                          else (2 if st.session_state.preset_name == "Strong" else 3)),
                )
                if selected_preset != "Custom":
                    try:
                        preset_obj = MarketAssumptions.from_preset(selected_preset)
                        st.session_state.preset_name = selected_preset
                        st.session_state.inflation_rate = preset_obj.inflation_rate
                        st.session_state.step_up_rate = preset_obj.step_up_rate
                        st.session_state.equity_return = preset_obj.equity_return
                        st.session_state.debt_return = preset_obj.debt_return
                    except ValueError:
                        pass

                col_ax, col_bx = st.columns(2)
                with col_ax:
                    st.session_state.inflation_rate = st.number_input(
                        "Inflation Rate (%)", min_value=0.0, max_value=20.0,
                        value=st.session_state.inflation_rate * 100, step=0.5,
                    ) / 100
                    st.session_state.equity_return = st.number_input(
                        "Equity Return (%)", min_value=0.0, max_value=30.0,
                        value=st.session_state.equity_return * 100, step=0.5,
                    ) / 100
                with col_bx:
                    st.session_state.step_up_rate = st.number_input(
                        "SIP Step-Up Rate (%)", min_value=0.0, max_value=20.0,
                        value=st.session_state.step_up_rate * 100, step=0.5,
                    ) / 100
                    st.session_state.debt_return = st.number_input(
                        "Debt Return (%)", min_value=0.0, max_value=15.0,
                        value=st.session_state.debt_return * 100, step=0.5,
                    ) / 100

            with st.expander("🎲 Asset Allocation", expanded=False):
                st.checkbox(
                    "✨ Use Smart Goal-Specific Glidepath",
                    key="use_smart_glidepath",
                    help="Automatically calculate equity % based on years-to-goal.",
                )
                st.info(
                    "💡 **Note:** If Smart Glidepath is checked, the engine will safely "
                    "override your manual slider below."
                )
                st.slider(
                    "Manual Equity %",
                    min_value=0.0,
                    max_value=100.0,
                    step=5.0,
                    key="manual_equity_pct",
                    help="Debt % auto-adjusts to sum to 100%.",
                )
                debt_pct = 100.0 - st.session_state.manual_equity_pct
                st.caption(f"Manual Debt: **{debt_pct:.1f}%**")

            with st.expander("🏛️ Tax Configuration", expanded=False):
                col_tx1, col_tx2 = st.columns(2)
                with col_tx1:
                    st.session_state.equity_ltcg_rate = st.number_input(
                        "Equity LTCG Rate (%)", min_value=0.0, max_value=50.0,
                        value=st.session_state.equity_ltcg_rate * 100, step=0.5,
                    ) / 100
                with col_tx2:
                    st.session_state.equity_exemption = st.number_input(
                        "Equity Exemption Limit (₹)", min_value=0,
                        value=int(st.session_state.equity_exemption), step=10_000,
                        help="Annual tax-free exemption on equity gains.",
                    )
                st.session_state.debt_slab_rate = st.number_input(
                    "Debt Slab Rate (%)", min_value=0.0, max_value=50.0,
                    value=st.session_state.debt_slab_rate * 100, step=1.0,
                ) / 100

            submitted = st.form_submit_button(
                "⚙️ Run Physics Engine",
                type="primary",
                use_container_width=True,
            )
            st.session_state.engine_triggered = submitted


# ============================================================================
# RIGHT COLUMN: LIVE INTELLIGENCE DISPLAY
# ============================================================================

def render_intelligence_display(col_right):
    """Render analysis and results in the right column."""
    with col_right:
        if not st.session_state.get("engine_triggered", False):
            st.info("Configure the controls and run the Physics Engine to view your analysis.")
            return

        st.subheader("🧠 Live Intelligence Display")
        
        try:
            # Get user inputs as backend objects
            goal = get_goal()
            assumptions = get_market_assumptions()
            allocation = get_allocation()
            tax_config = get_tax_config()
            # 🔧 FIX: Inflate target for tomorrow's prices before UI rendering cascades
            goal.target_amount = goal.target_amount * ((1.0 + assumptions.inflation_rate) ** goal.years_to_goal)
            # ===== Calculate Projections =====
            # Backend: calculate_step_up_sip_trajectory
            trajectory = calculate_step_up_sip_trajectory(
                st.session_state.initial_sip,
                allocation,
                assumptions,
                int(goal.years_to_goal),
                current_equity=st.session_state.current_equity,
                current_debt=st.session_state.current_debt,
            )
            
            final_corpus = trajectory["corpus_end"]
            principal = trajectory["principal_invested"]
            gains = trajectory["gains"]
            
            # ===== Health Score (Prominent Display) =====
            shortfall = max(0, goal.target_amount - final_corpus)
            health_score = calculate_health_score(
                allocation.equity_pct,
                final_corpus,
                goal.target_amount,
                goal.years_to_goal,
                goal.is_critical,
            )
            
            # Color-coded health score
            if health_score >= 80:
                color = "green"
            elif health_score >= 60:
                color = "orange"
            else:
                color = "red"
            
            st.markdown(
                f"<h2 style='text-align: center; color: {color};'>"
                f"Health Score: {health_score}/100</h2>",
                unsafe_allow_html=True,
            )
            
            # ===== Wealth Trajectory Chart =====
            st.markdown("**📈 Wealth Trajectory**")
            
            # Prepare chart data directly from the high-resolution monthly engine
            df_trajectory = pd.DataFrame(trajectory["monthly_trajectory"])
        #Only render the chart if the DataFrame actually has data
            if not df_trajectory.empty:
                # Create area chart using Altair
                chart = alt.Chart(df_trajectory).mark_area(
                    line=True, point=False, opacity=0.7
                ).encode(
                x=alt.X("Month:Q", title="Month", scale=alt.Scale(domain=[0, df_trajectory["Month"].max()], nice=False)),
                y=alt.Y("Gross Corpus:Q", axis=alt.Axis(title="Amount (₹)", format=",.0f")),
                tooltip=["Month:Q", "Principal Invested:Q", "Gross Corpus:Q"]
            ).properties(
                height=300,
                width=400,
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            # ===== Key Metrics Row =====
            st.markdown("**💡 Key Metrics**")
            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
            
            with metric_col1:
                st.metric(
                    "Principal Invested",
                    format_currency(principal),
                )
            
            with metric_col2:
                st.metric(
                    "Capital Gains",
                    format_currency(gains),
                )
            
            with metric_col3:
                st.metric(
                    "Final Corpus",
                    format_currency(final_corpus),
                    delta=format_currency(final_corpus - goal.target_amount),
                )
            
            # Calculate Gain % (with division by zero check)
            gain_percentage = 0.0
            if principal > 1e-10:
                gain_percentage = (gains / principal) * 100
            
            with metric_col4:
                st.metric(
                    "Gain %",
                    f"{gain_percentage:.2f}%",
                )
            
            # Calculate XIRR/CAGR proxy (with division by zero check)
            cagr_percentage = 0.0
            if principal > 1e-10:
                if goal.years_to_goal == 0:
                    cagr_percentage = 0.0
                else:
                    cagr_percentage = (((final_corpus / principal) ** (1 / goal.years_to_goal)) - 1) * 100
            
            with metric_col5:
                st.metric(
                    "XIRR (Extended Internal Rate of Return)",
                    f"{cagr_percentage:.2f}%",
                )
            
            # ===== Tax Solver: Gross-Up Analysis =====
            st.markdown("**🏛️ Tax Impact Analysis**")
            
            # Get equity and debt future values from trajectory
            fv_equity = trajectory["fv_equity_existing"] + trajectory["fv_equity_sip"]
            fv_debt = trajectory["fv_debt_existing"] + trajectory["fv_debt_sip"]
            cost_equity = trajectory["cost_equity_total"]
            cost_debt = trajectory["cost_debt_total"]
            
            # Backend: solve_gross_liquidation
            tax_result = solve_gross_liquidation(
                goal.target_amount,
                fv_equity,
                cost_equity,
                fv_debt,
                cost_debt,
                tax_config,
            )
            
            # Display tax metrics
            tax_col1, tax_col2, tax_col3 = st.columns(3)
            
            with tax_col1:
                st.metric(
                    "Net Goal Required",
                    format_currency(goal.target_amount),
                )
            
            with tax_col2:
                st.metric(
                    "Estimated Total Tax",
                    format_currency(tax_result["total_tax"]),
                )
            
            with tax_col3:
                st.metric(
                    "Gross Liquidation Needed",
                    format_currency(tax_result["gross_required"]),
                )
            
            # Tax breakdown (smaller)
            st.caption(
                f"Equity Tax: {format_currency(tax_result['equity_tax'])} | "
                f"Debt Tax: {format_currency(tax_result['debt_tax'])}"
            )
            
            # ===== Mutually Exclusive Status Alert =====
            gross_liquidation_required = tax_result["gross_required"]
            plan_on_track = final_corpus >= gross_liquidation_required
            
            if plan_on_track:
                st.success("✅ Plan on Track: Your corpus will meet the goal!")
            else:
                st.error(
                    "⚠️ Portfolio Insufficient: The projected corpus cannot "
                    "meet your gross liquidation requirement after taxes."
                )
                
                # Show course corrections only if plan is not on track
                st.markdown("**🔧 Course Correction Advisory**")
                
                # Backend: calculate_course_corrections
                corrections = calculate_course_corrections(
                    shortfall,
                    goal,
                    assumptions,
                    st.session_state.initial_sip,
                )
                
                col_corr1, col_corr2 = st.columns(2)
                
                with col_corr1:
                    st.metric(
                        "💪 Extra Monthly SIP Needed",
                        format_currency(corrections["required_extra_sip"]),
                        help="Start this additional SIP today to bridge the gap.",
                    )
                
                with col_corr2:
                    st.metric(
                        "⏰ Extra Months to Wait",
                        f"{corrections['required_delay_months']} months",
                        help="Delay the goal by this many months to let growth catch up.",
                    )
                
                st.warning(
                    f"**Shortfall Detected**: {format_currency(shortfall)} | "
                    f"Bridge by increasing SIP or extending timeline."
                )
            
           # ===== Allocation Display =====
            st.markdown("**🎯 Recommended Allocation**")
            alloc_col1, alloc_col2 = st.columns(2)
            with alloc_col1:
                st.metric("Equity", f"{allocation.equity_pct:.1f}%")
            with alloc_col2:
                st.metric("Debt", f"{allocation.debt_pct:.1f}%")

            # ===== Adaptive Stress & Resilience Engine =====
            from backend.math_engine import calculate_plan_resilience
            
            st.divider()
            with st.expander("🌪️ Stress Test Analysis", expanded=True):
                resilience_data = calculate_plan_resilience(
                    st.session_state.current_equity,
                    st.session_state.current_debt,
                    st.session_state.initial_sip,
                    goal,
                    assumptions,
                    tax_config
                )
                
                st.markdown("How badly can reality deviate before the plan breaks?")
                
                metrics = []
                for scenario, data in resilience_data.items():
                    status = "✅ Survives" if data["survives_stress"] else "❌ Breaks"
                    metrics.append({
                        "Scenario": scenario.replace("_", " "),
                        "Required Gross": format_currency(data["Gross Requirement"]),
                        "Outcome": status
                    })
                
                st.table(metrics)
        
        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_planner_view():
    """Main function to run the planner view."""
    st.set_page_config(
        page_title="Wealth Planner",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    
    # Initialize session state
    init_session_state()
    
    # Page title
    st.title("💰 Wealth Management Planner")
    st.subheader("Smart Financial Planning with AI-Driven Portfolio Optimization")
    
    # Main layout: two columns
    col_left, col_right = st.columns([1, 1.5])
    
    # Render control room (left) and intelligence display (right)
    render_control_room(col_left)
    render_intelligence_display(col_right)
    
    # Footer
    st.divider()
    st.caption(
        "🔒 **Privacy**: All calculations are performed locally. "
        "No data is stored or transmitted."
    )


def render_planner():
    """Backward compatibility wrapper."""
    run_planner_view()


if __name__ == "__main__":
    run_planner_view()
