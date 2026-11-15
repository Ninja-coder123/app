"""Master router for the Wealth Manager application."""

import streamlit as st

from frontend.dashboard_view import render_dashboard
from frontend.planner_view import render_planner
from frontend.portfolio_view import render_portfolio_generator
from frontend.screener_view import render_screener


st.set_page_config(layout="wide")

header_left, header_right = st.columns([4, 1])
with header_right:
    selected_view = st.selectbox(
        "Navigate",
        ["📊 Dashboard (Cockpit)", "📈 Financial Planning (Step 2)", "🎯 Portfolio Generator (Step 3)", "🏛️ Institutional Mutual Fund Screener"],
        label_visibility="collapsed",
    )

if selected_view == "📊 Dashboard (Cockpit)":
    render_dashboard()
elif selected_view == "📈 Financial Planning (Step 2)":
    render_planner()
elif selected_view == "🎯 Portfolio Generator (Step 3)":
    render_portfolio_generator()
elif selected_view == "🏛️ Institutional Mutual Fund Screener":
    render_screener()