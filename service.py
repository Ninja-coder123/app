from __future__ import annotations

from typing import Dict, List

from .analytics import compute_all_fund_features
from .database import get_connection, insert_pipeline_run
from .ingest import bootstrap_history, ingest_master_scheme_directory, run_daily_pipeline
from .scoring import compute_peer_scores


def export_to_parquet(conn):
    """Exports the calculated Tier A funds to the Parquet file used by the frontend."""
    conn.execute("""
        COPY (
            SELECT 
                m.scheme_name,
                m.canonical_category AS category,
                s.research_score AS pr_fqs_score,
                f.sharpe_ratio AS sharpe,
                f.sortino_ratio AS sortino,
                f.volatility_annualized AS volatility,
                f.beta,
                (SELECT nav FROM nav_history h WHERE h.scheme_code = m.scheme_code ORDER BY nav_date DESC LIMIT 1) AS latest_nav
            FROM scheme_master m
            JOIN fund_features f ON m.scheme_code = f.scheme_code
            JOIN fund_scores s ON m.scheme_code = s.scheme_code
            WHERE s.ranking_status = 'VALID'
        ) TO 'data/screener_database.parquet' (FORMAT PARQUET)
    """)


def initialize_and_bootstrap() -> Dict[str, int]:
    conn = get_connection()
    schemes_count = ingest_master_scheme_directory()
    rows = 0
    resume_query = """
        SELECT scheme_code FROM scheme_master 
        WHERE scheme_code NOT IN (SELECT DISTINCT scheme_code FROM nav_history)
        ORDER BY scheme_code
    """
    for row in conn.execute(resume_query).fetchall():
        scheme_code = row[0]
        rows += bootstrap_history(scheme_code)
    
    compute_all_fund_features()
    compute_peer_scores()
    
    # Generate the Parquet file for the frontend
    export_to_parquet(conn)
    
    insert_pipeline_run(conn, "bootstrap-run", "BOOTSTRAP", "SUCCESS", schemes_processed=schemes_count, rows_inserted=rows)
    return {"schemes_processed": schemes_count, "rows_inserted": rows}


def daily_run() -> Dict[str, object]:
    result = run_daily_pipeline()
    compute_all_fund_features()
    compute_peer_scores()
    
    # Update the Parquet file daily
    conn = get_connection()
    export_to_parquet(conn)
    
    return result