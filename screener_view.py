import streamlit as st
import duckdb

@st.cache_resource
def get_db_connection():
    # Connect in read-only mode so it doesn't lock the database from the background updater
    return duckdb.connect("data/mf_terminal.duckdb", read_only=True)

def render_screener():
    st.title("🏛️ Institutional Mutual Fund Screener")
    st.markdown("Offline-first risk engine with Leave-One-Out (LOO) peer benchmarking and confidence scoring.")

    try:
        conn = get_db_connection()
    except duckdb.IOException:
        st.error("Database not found. Please run `python run_mf_terminal.py --bootstrap` in your terminal first.")
        return

    st.sidebar.header("Screener Filters")
    categories = conn.execute("SELECT DISTINCT canonical_category FROM scheme_master WHERE is_active = TRUE").df()
    selected_category = st.sidebar.selectbox("Select SEBI Category", categories['canonical_category'].tolist())

    query = """
        SELECT 
            m.scheme_name AS "Fund Name",
            m.amc_name AS "AMC",
            ROUND(f.cagr_3y * 100, 2) AS "3Y CAGR (%)",
            ROUND(f.volatility_annualized * 100, 2) AS "Volatility (%)",
            ROUND(f.max_drawdown * 100, 2) AS "Max Drawdown (%)",
            ROUND(f.sharpe_ratio, 2) AS "Sharpe",
            s.category_rank AS "Rank",
            ROUND(s.research_score, 1) AS "PR-FQS Score",
            ROUND(s.confidence_score, 1) AS "Data Confidence",
            s.ranking_status
        FROM scheme_master m
        JOIN fund_features f ON m.scheme_code = f.scheme_code
        JOIN fund_scores s ON m.scheme_code = s.scheme_code
        WHERE m.canonical_category = ? AND m.is_active = TRUE
        ORDER BY s.category_rank ASC NULLS LAST
    """
    df = conn.execute(query, [selected_category]).df()

    if df.empty:
        st.warning("No active funds found for this category.")
    else:
        # Split into Tier A (Ranked) and Tier B (Valid but Unranked)
        tier_a = df[df['ranking_status'] == 'VALID'].drop(columns=['ranking_status'])
        tier_b = df[df['ranking_status'] == 'INSUFFICIENT_PEERS'].drop(columns=['ranking_status', 'Rank', 'PR-FQS Score'])

        # Display Tier A: Quant Ranked
        st.subheader(f"🟢 Tier A: Quant Ranked ({len(tier_a)} Funds)")
        if not tier_a.empty:
            st.dataframe(tier_a, use_container_width=True, hide_index=True)
        else:
            st.info("No funds in this category passed the sample-size gate (N-1 ≥ 10) for quantitative ranking.")

        st.divider()

        # Display Tier B: Valid but Unranked
        st.subheader(f"🟡 Tier B: Valid but Unranked ({len(tier_b)} Funds)")
        st.markdown("*Limited peer sample — quantitative PR-FQS ranking unavailable. Raw metrics provided below.*")
        if not tier_b.empty:
            st.dataframe(tier_b, use_container_width=True, hide_index=True)
        else:
            st.write("No unranked funds in this category.")