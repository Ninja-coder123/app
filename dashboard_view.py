"""Step 1 portfolio diagnostic dashboard."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend.parser_module import parse_pdf_for_funds
from backend.scoring_module import categorize_mf, compute_health_score


def render_dashboard():
    """Capture dashboard inputs, call backend services, and render results."""
    st.title("💼 Wealth Manager Command Center")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        risk_profile = st.selectbox(
            "Client Risk Profile",
            ["Ultra-Conservative", "Conservative", "Balanced", "Aggressive", "Maximum Growth"],
        )
    with col2:
        fri_input = st.number_input("Fixed Returns (₹)", min_value=0.0, step=10000.0)
    with col3:
        gold_input = st.number_input("Physical Gold (₹)", min_value=0.0, step=10000.0)
    with col4:
        re_input = st.number_input("Real Estate (₹)", min_value=0.0, step=10000.0)

    gold_override_enabled = st.checkbox(
        "Use a target Gold allocation",
        help="Override the intelligent gold allocation guidance with your target percentage.",
    )
    gold_target_pct = st.number_input(
        "Target Gold Allocation (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        disabled=not gold_override_enabled,
    )
    if gold_override_enabled:
        st.caption(f"Gold allocation target: **{gold_target_pct:.1f}%**")
    else:
        st.caption("Gold allocation target: **Intelligent default**")

    funds_df = pd.DataFrame(columns=["name", "value", "Include", "category"])
    raw_text = ""
    with st.expander("📝 Upload & Edit Mutual Funds", expanded=False):
        uploaded = st.file_uploader("Upload CAS PDF", type=["pdf"])
        pdf_password = st.text_input("PDF Password", type="password")
        if uploaded is not None:
            funds_df, raw_text = parse_pdf_for_funds(uploaded.read(), pdf_password)

        if funds_df.empty:
            st.info("No funds detected. Enter manually (Name,Value per line):")
            manual = st.text_area(
                "Manual fund entries",
                placeholder="Axis Bluechip,150000\nHDFC Smallcap,80000",
                key="manual_funds",
            )
            if manual:
                rows = []
                for ln in manual.splitlines():
                    parts = ln.split(",")
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        try:
                            val = float(parts[1].replace(",", ""))
                        except Exception:
                            val = 0.0
                        rows.append((name, val))
                funds_df = pd.DataFrame(rows, columns=["name", "value"])

        if not funds_df.empty:
            if "Include" not in funds_df.columns:
                funds_df["Include"] = True
            funds_df["Include"] = funds_df["Include"].fillna(True).astype(bool)
            funds_df["value"] = pd.to_numeric(funds_df["value"], errors="coerce").fillna(0)
            funds_df["category"] = funds_df["name"].fillna("").apply(categorize_mf)
            st.write("Fund holdings (uncheck Include to exclude a holding):")
            editor_columns = ["Include", "name", "category", "value"]
            try:
                funds_df = st.data_editor(
                    funds_df[editor_columns],
                    num_rows="dynamic",
                    key="funds_editor",
                    disabled=["category"],
                )
            except Exception:
                try:
                    funds_df = st.experimental_data_editor(
                        funds_df[editor_columns],
                        num_rows="dynamic",
                        key="funds_editor",
                        disabled=["category"],
                    )
                except Exception:
                    pass

            funds_df["Include"] = funds_df["Include"].fillna(True).astype(bool)
            funds_df["value"] = pd.to_numeric(funds_df["value"], errors="coerce").fillna(0)
            funds_df["category"] = funds_df["name"].fillna("").apply(categorize_mf)
#show raw text in an expander if available
        """if raw_text:
            with st.expander("Raw PDF Text"):
                st.text_area("Extracted text", raw_text, height=200, disabled=True)"""

    fri_df = pd.DataFrame([{"amount": fri_input, "rate": 0}]) if fri_input > 0 else pd.DataFrame(columns=["amount", "rate"])
    physical_gold_df = pd.DataFrame([{"description": "Manual Entry", "value": gold_input}]) if gold_input > 0 else pd.DataFrame(columns=["description", "value"])
    re_df = pd.DataFrame([{"desc": "Manual Entry", "amount": re_input}]) if re_input > 0 else pd.DataFrame(columns=["desc", "amount"])

    if not funds_df.empty:
        funds_df["category"] = funds_df["name"].fillna("").apply(categorize_mf)
        funds_df["value"] = pd.to_numeric(funds_df["value"], errors="coerce").fillna(0)
        funds_df["Include"] = funds_df["Include"].fillna(True).astype(bool)

    if not funds_df.empty:
        funds_df = funds_df[funds_df["Include"]].copy()

    equity_total = float(funds_df.loc[funds_df["category"] == "Equity MF", "value"].sum()) if not funds_df.empty else 0.0
    debt_total = float(funds_df.loc[funds_df["category"] == "Debt MF", "value"].sum()) if not funds_df.empty else 0.0
    gold_silver_mf_total = float(funds_df.loc[funds_df["category"] == "Gold/Silver MF", "value"].sum()) if not funds_df.empty else 0.0
    unclassified_total = float(funds_df.loc[funds_df["category"] == "Unclassified MF", "value"].sum()) if not funds_df.empty else 0.0
    fri_total = float(fri_df["amount"].sum()) if not fri_df.empty else 0.0
    physical_gold_total = float(physical_gold_df["value"].sum()) if not physical_gold_df.empty else 0.0
    re_total = float(re_df["amount"].sum()) if not re_df.empty else 0.0
    portfolio_total = equity_total + debt_total + gold_silver_mf_total + unclassified_total + fri_total + physical_gold_total + re_total
    balances = {
        "Equity MF": equity_total,
        "Debt MF": debt_total,
        "Gold/Silver MF": gold_silver_mf_total,
        "FRI": fri_total,
        "Physical Gold/Silver": physical_gold_total,
        "Real Estate": re_total,
    }
    equity_count = int(funds_df.loc[funds_df["category"] == "Equity MF"].shape[0]) if not funds_df.empty else 0
    score, penalties, recommendations = compute_health_score(
        balances, equity_count, risk_profile, funds_df, fri_df, physical_gold_df
    )
# --- FIX: Adjust or suppress Heavy Metal Drag penalty if custom gold target override is active ---
    if gold_override_enabled and portfolio_total > 0:
         total_gold_value = gold_silver_mf_total + physical_gold_total
         actual_gold_pct = (total_gold_value / portfolio_total) * 100
        
         updated_penalties = []
         for p in penalties:
             if "Heavy Metal Drag" in p[0]:
                 if actual_gold_pct <= gold_target_pct:
                     continue
                 else:
                     updated_penalties.append((p[0], f"Gold/Silver {actual_gold_pct:.1f}% > {gold_target_pct:.1f}% target portfolio allocation."))
             else:
                updated_penalties.append(p)
         penalties = updated_penalties
    color_map = {
        "Equity MF": "#0047AB", "Debt MF": "#00A86B", "Gold/Silver MF": "#FFD700",
        "Physical Gold/Silver": "#DAA520", "FRI": "#708090", "Real Estate": "#8B4513",
        "Unclassified MF": "#808080",
    }
    with st.container():
        col_left, col_mid, col_right = st.columns([1, 1.2, 1.2])
        with col_left:
            st.metric("💰 Portfolio Value", f"₹{portfolio_total:,.0f}")
            st.write("**Asset Breakdown:**")
            breakdown = pd.DataFrame([balances]).T.reset_index()
            breakdown.columns = ["Asset", "Value"]
            breakdown["Value"] = breakdown["Value"].map(lambda value: f"₹{value:,.0f}")
            st.table(breakdown)
        with col_mid:
            fig_gauge = go.Figure(data=[go.Indicator(
                mode="gauge+number", value=score, domain={"x": [0, 1], "y": [0, 1]}, title={"text": "Health Score"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#0047AB"}, "steps": [
                    {"range": [0, 40], "color": "#ffcccc"}, {"range": [40, 75], "color": "#ffffcc"}, {"range": [75, 100], "color": "#ccffcc"}
                ]},
            )])
            fig_gauge.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font={"size": 14, "color": "white"}, height=350, margin={"l": 20, "r": 20, "t": 60, "b": 20})
            st.plotly_chart(fig_gauge, use_container_width=True)
        with col_right:
            pie_df = pd.DataFrame({"asset": list(balances.keys()), "value": list(balances.values())})
            pie_df = pie_df[pie_df["value"] > 0]
            if not pie_df.empty:
                fig_donut = px.pie(pie_df, names="asset", values="value", hole=0.4, title="Asset Allocation", color="asset", color_discrete_map=color_map)
                fig_donut.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font={"size": 12, "color": "white"}, height=350, margin={"l": 20, "r": 20, "t": 60, "b": 20})
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.info("No asset allocation data")

    st.markdown("---")
    col_penalties, col_recommendations = st.columns(2)
    with col_penalties:
        st.subheader("🚨 Penalties Applied")
        if penalties:
            for penalty_name, penalty_desc in penalties:
                st.warning(f"**{penalty_name}**\n{penalty_desc}")
        else:
            st.success("✅ No penalties — well-balanced portfolio!")
    with col_recommendations:
        st.subheader("✅ Action Items")
        if recommendations:
            for i, recommendation in enumerate(recommendations, 1):
                st.info(f"**{i}.** {recommendation}")
        else:
            st.success("No recommendations needed.")

    if not funds_df.empty:
        st.markdown("---")
        st.subheader("Individual Mutual Funds")
        funds_plot = funds_df.sort_values("value", ascending=False)
        fig_bar = px.bar(funds_plot, x="name", y="value", color="category", title="Fund Holdings Breakdown", color_discrete_map=color_map)
        fig_bar.update_layout(xaxis_title="Fund Name", yaxis_title="Value (₹)", xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.caption("🎯 Top-Bar Command Center: Real-time wealth analytics for live client consultations")
