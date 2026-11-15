"""
Portfolio Generator Frontend View
Handles all Streamlit UI for psychometric risk assessment and portfolio visualization.
Clean separation: NO financial math logic here, only presentation.
"""

import asyncio
import json
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from backend.portfolio_engine import (
    RISK_ASSESSMENT_QUESTIONS,
    generate_portfolio,
)


# ============================================================================
# PSYCHOMETRIC QUESTIONS DEFINITION
# ============================================================================

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


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "portfolio_assessment_complete" not in st.session_state:
        st.session_state.portfolio_assessment_complete = False
    if "portfolio_generated" not in st.session_state:
        st.session_state.portfolio_generated = None
    if "portfolio_assessment_answers" not in st.session_state:
        st.session_state.portfolio_assessment_answers = {}
    if "portfolio_monthly_sip" not in st.session_state:
        st.session_state.portfolio_monthly_sip = 10000
    if "portfolio_reset_counter" not in st.session_state:
        st.session_state.portfolio_reset_counter = 0
# ============================================================================
# UI COMPONENTS
# ============================================================================


def render_psychometric_assessment():
    """Render psychometric risk assessment UI with persistent state."""
    st.subheader("📋 Psychometric Risk Assessment")
    st.write(
        "Complete this 7-question assessment to determine your risk profile. "
        "Your answers are saved automatically."
    )

    col1, col2 = st.columns(2)
    reset_suffix = st.session_state.portfolio_reset_counter

    with col1:
        st.write("**Questions 1-4:**")

        for question_data in RISK_ASSESSMENT_QUESTIONS[:4]:
            q_id = question_data["id"]
            question_text = question_data["question"]
            options = question_data["options"]

            st.write(f"\n**Q{q_id}: {question_text}**")
            selected = st.radio(
                "Select your answer:",
                options,
                format_func=lambda x: x[0],
                key=f"portfolio_q_{q_id}_reset_{reset_suffix}",
                index=None,
                label_visibility="collapsed",
            )
            if selected is not None:
                st.session_state.portfolio_assessment_answers[q_id] = selected[1]

    with col2:
        st.write("**Questions 5-7:**")

        for question_data in RISK_ASSESSMENT_QUESTIONS[4:]:
            q_id = question_data["id"]
            question_text = question_data["question"]
            options = question_data["options"]

            st.write(f"\n**Q{q_id}: {question_text}**")
            selected = st.radio(
                "Select your answer:",
                options,
                format_func=lambda x: x[0],
                key=f"portfolio_q_{q_id}_reset_{reset_suffix}",
                index=None,
                label_visibility="collapsed",
            )
            if selected is not None:
                st.session_state.portfolio_assessment_answers[q_id] = selected[1]

                # Dynamic Q6-to-SIP linking: Auto-update SIP based on Question 6 answer
                if q_id == 6:
                    q6_value = selected[1]
                    if q6_value <= 2:
                        new_sip = 5000
                    elif q6_value <= 4:
                        new_sip = 25000
                    elif q6_value <= 6:
                        new_sip = 75000
                    elif q6_value <= 8:
                        new_sip = 250000
                    else:
                        new_sip = 500000
                    st.session_state.portfolio_monthly_sip = new_sip

    st.divider()

    # SIP Input Section
    st.write("**💰 Monthly Investment Details:**")
    col_sip, col_spacer = st.columns([2, 3])
    with col_sip:
        # Bind directly to session state without conflicting key parameter
        monthly_sip = st.number_input(
            "Target Monthly SIP Amount (₹)",
            min_value=1000,
            max_value=100000000,
            step=5000,
            value=st.session_state.portfolio_monthly_sip,
        )
        # Update session state dynamically when user changes input
        st.session_state.portfolio_monthly_sip = monthly_sip

    # Action Buttons
    col_generate, col_reset = st.columns(2)
    with col_generate:
        if st.button(
            "🚀 Generate Optimized Portfolio",
            key="portfolio_generate_btn",
            use_container_width=True,
        ):
            # Validation: Check all 7 questions answered
            if len(st.session_state.portfolio_assessment_answers) < 7:
                st.error("⚠️ Please answer all 7 questions before generating the portfolio.")
            # Strict SIP validation: Enforce minimum ₹1,000
            elif st.session_state.get("portfolio_monthly_sip", 0) < 1000:
                st.error(
                    "⚠️ Please enter a valid monthly SIP amount (minimum ₹1,000)."
                )
            else:
                st.session_state.portfolio_assessment_complete = True
                st.rerun()

    with col_reset:
        if st.button(
            "🔄 Reset Assessment",
            key="portfolio_reset_btn",
            use_container_width=True,
        ):
            # Complete hard reset: increment counter, clear all state
            st.session_state.portfolio_reset_counter += 1
            st.session_state.portfolio_assessment_complete = False
            st.session_state.portfolio_generated = None
            st.session_state.portfolio_assessment_answers = {}
            st.session_state.portfolio_monthly_sip = 10000
            st.rerun()
def render_portfolio_results(portfolio):
    """Render portfolio results with visualizations and data."""
    st.subheader("📈 Portfolio Results")

    # Risk Profile Summary
    col_score, col_tier, col_sip = st.columns(3)
    with col_score:
        risk_score = portfolio.client_profile["risk_score"]
        st.metric("Risk Score", f"{risk_score} / 70")

    with col_tier:
        st.metric("Risk Tier", portfolio.risk_tier)

    with col_sip:
        st.metric(
            "Total Monthly SIP",
            f"₹{portfolio.total_sip:,.0f}",
        )

    st.divider()

    # Visualizations
    col_chart, col_metrics = st.columns([1.5, 1])

    with col_chart:
        st.write("**Asset Allocation Breakdown:**")
        allocation_data = pd.DataFrame(
            [
                {
                    "Asset Class": "Equity",
                    "Percentage": portfolio.allocation_breakdown["equity"],
                },
                {
                    "Asset Class": "Debt",
                    "Percentage": portfolio.allocation_breakdown["debt"],
                },
            ]
        )

        chart = (
            alt.Chart(allocation_data)
            .mark_arc(innerRadius=60)
            .encode(
                theta="Percentage",
                color=alt.Color(
                    "Asset Class:N",
                    scale=alt.Scale(
                        domain=["Equity", "Debt"], range=["#1f77b4", "#ff7f0e"]
                    ),
                ),
                tooltip=["Asset Class", "Percentage:Q"],
            )
            .properties(width=350, height=300, title="")
        )

        st.altair_chart(chart, use_container_width=True)

    with col_metrics:
        st.write("**Capital Allocation (₹):**")
        equity_amount = portfolio.total_sip * portfolio.allocation_breakdown["equity"] / 100
        debt_amount = portfolio.total_sip * portfolio.allocation_breakdown["debt"] / 100

        st.metric(
            "Equity",
            f"₹{equity_amount:,.0f}",
            delta=f"{portfolio.allocation_breakdown['equity']}%",
            delta_color="off",
        )
        st.metric(
            "Debt",
            f"₹{debt_amount:,.0f}",
            delta=f"{portfolio.allocation_breakdown['debt']}%",
            delta_color="off",
        )

    st.divider()

    # Fund Recommendations
    st.write("**🎯 Recommended Funds (Sorted by PR-FQS):**")

    if portfolio.recommended_portfolio:
        fund_data = []
        for alloc in sorted(
            portfolio.recommended_portfolio, key=lambda x: x.pr_fqs_score, reverse=True
        ):
            fund_data.append(
                {
                    "Fund Name": alloc.fund_name,
                    "Category": alloc.category,
                    "Monthly SIP (₹)": f"{alloc.sip_amount:,.2f}",
                    "Allocation %": f"{(alloc.sip_amount / portfolio.total_sip * 100):.1f}%",
                    "PR-FQS Score (out of 100)": f"{alloc.pr_fqs_score:.1f}",
                }
            )

        df = pd.DataFrame(fund_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ No funds found for the selected categories. Please try again.")

    st.divider()

    # JSON Export (download button only, no raw display)
    portfolio_json = {
        "client_profile": portfolio.client_profile,
        "recommended_portfolio": [
            {
                "scheme_code": alloc.scheme_code,
                "fund_name": alloc.fund_name,
                "category": alloc.category,
                "sip_amount": round(alloc.sip_amount, 2),
                "pr_fqs_score": round(alloc.pr_fqs_score, 2),
                "expected_return_percent": round(alloc.expected_return, 2),
            }
            for alloc in portfolio.recommended_portfolio
        ],
        "total_sip_amount": round(portfolio.total_sip, 2),
        "risk_tier": portfolio.risk_tier,
        "allocation_breakdown": portfolio.allocation_breakdown,
    }

    json_str = json.dumps(portfolio_json, indent=2)
    st.download_button(
        label="📥 Download Portfolio JSON",
        data=json_str,
        file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )

    st.divider()

    # Back Button
    if st.button("🔙 Back to Assessment", use_container_width=True):
        st.session_state.portfolio_assessment_complete = False
        st.session_state.portfolio_generated = None
        st.rerun()


# ============================================================================
# MAIN RENDERER
# ============================================================================


def render_portfolio_generator():
    """Main renderer for the portfolio generator section."""
    st.title("🎯 Portfolio Generator (Step 3)")
    st.markdown(
        """
    **SEBI-Compliant Mutual Fund Portfolio Builder**
    
    Get a personalized, deterministic portfolio based on your risk profile
    and investment goals using live market data.
    """
    )

    initialize_session_state()

    if not st.session_state.portfolio_assessment_complete:
        # Assessment Phase
        render_psychometric_assessment()

    else:
        # Generation & Results Phase
        if st.session_state.portfolio_generated is None:
            with st.spinner(
                "⏳ Fetching live data and calculating optimal fund selection... This may take 30-60 seconds."
            ):
                try:
                    # Run async portfolio generation
                    portfolio = generate_portfolio(
                            st.session_state.portfolio_assessment_answers,
                            st.session_state.portfolio_monthly_sip,
                        )
                    
                    st.session_state.portfolio_generated = portfolio
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error generating portfolio: {str(e)}")
                    st.info("💡 Tip: Ensure you have an active internet connection.")
        else:
            # Display Results
            render_portfolio_results(st.session_state.portfolio_generated)
