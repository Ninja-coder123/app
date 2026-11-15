from mf_research_terminal.database import get_connection

def export_to_parquet(conn):
    # This pure SQL query bypasses analytics.py entirely. 
    # It calculates real risk metrics directly from your raw downloaded data.
    conn.execute("""
        COPY (
            WITH daily_returns AS (
                SELECT 
                    scheme_code,
                    nav_date,
                    nav,
                    (nav - LAG(nav) OVER (PARTITION BY scheme_code ORDER BY nav_date)) / 
                    NULLIF(LAG(nav) OVER (PARTITION BY scheme_code ORDER BY nav_date), 0) AS d_ret
                FROM nav_history
            ),
            metrics AS (
                SELECT 
                    scheme_code,
                    (AVG(d_ret) * 252) AS annualized_return,
                    (STDDEV_SAMP(d_ret) * SQRT(252)) AS volatility
                FROM daily_returns
                GROUP BY scheme_code
            ),
            latest_navs AS (
                SELECT scheme_code, nav
                FROM (
                    SELECT scheme_code, nav, ROW_NUMBER() OVER (PARTITION BY scheme_code ORDER BY nav_date DESC) as rn
                    FROM nav_history
                ) WHERE rn = 1
            )
            SELECT 
                m.scheme_name,
                m.canonical_category AS category,
                85.0 AS pr_fqs_score, -- Mocked peer score for UI
                ((metrics.annualized_return - 0.07) / NULLIF(metrics.volatility, 0)) AS sharpe,
                1.5 AS sortino,       -- Mocked metric for UI
                (metrics.volatility * 100) AS volatility,
                1.0 AS beta,          -- Mocked metric for UI
                l.nav AS latest_nav
            FROM scheme_master m
            JOIN metrics ON m.scheme_code = metrics.scheme_code
            JOIN latest_navs l ON m.scheme_code = l.scheme_code
        ) TO 'data/screener_database.parquet' (FORMAT PARQUET)
    """)

def main():
    print("1/2: Connecting to DuckDB...")
    conn = get_connection()
    
    print("2/2: Calculating metrics via SQL & building Parquet export...")
    export_to_parquet(conn)
    
    print("\n✅ Test Processing Complete! You can now launch Streamlit.")

if __name__ == "__main__":
    main()