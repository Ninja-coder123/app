from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb

from .config import DB_PATH, ensure_data_directory

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scheme_master (
    scheme_code INTEGER PRIMARY KEY,
    isin VARCHAR,
    amc_name VARCHAR NOT NULL,
    scheme_name VARCHAR NOT NULL,
    canonical_category VARCHAR NOT NULL,
    plan VARCHAR NOT NULL,
    option_type VARCHAR NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    first_nav_date DATE,
    latest_nav_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS nav_history (
    scheme_code INTEGER NOT NULL,
    nav_date DATE NOT NULL,
    nav DOUBLE NOT NULL,
    source VARCHAR NOT NULL,
    source_hash VARCHAR,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (scheme_code, nav_date)
);

CREATE TABLE IF NOT EXISTS scheme_lifecycle (
    event_id VARCHAR PRIMARY KEY,
    scheme_code INTEGER NOT NULL,
    event_type VARCHAR NOT NULL,
    event_date DATE NOT NULL,
    details VARCHAR
);

CREATE TABLE IF NOT EXISTS benchmark_history (
    benchmark_id VARCHAR NOT NULL,
    trade_date DATE NOT NULL,
    index_value DOUBLE NOT NULL,
    PRIMARY KEY (benchmark_id, trade_date)
);

CREATE TABLE IF NOT EXISTS scheme_benchmark_map (
    scheme_code INTEGER NOT NULL,
    benchmark_id VARCHAR NOT NULL,
    effective_from DATE NOT NULL,
    effective_to DATE,
    PRIMARY KEY (scheme_code, benchmark_id, effective_from)
);

CREATE TABLE IF NOT EXISTS fund_features (
    scheme_code INTEGER NOT NULL,
    as_of_date DATE NOT NULL,
    history_days INTEGER,
    history_coverage DOUBLE,
    cagr_1m DOUBLE, cagr_3m DOUBLE, cagr_6m DOUBLE,
    cagr_1y DOUBLE, cagr_3y DOUBLE, cagr_5y DOUBLE, cagr_10y DOUBLE,
    rolling_3y_mean DOUBLE,
    rolling_3y_min DOUBLE,
    rolling_3y_max DOUBLE,
    rolling_3y_positive_pct DOUBLE,
    volatility_annualized DOUBLE,
    downside_deviation DOUBLE,
    max_drawdown DOUBLE,
    max_drawdown_duration_days INTEGER,
    recovery_duration_days INTEGER,
    ulcer_index DOUBLE,
    var_95 DOUBLE, cvar_95 DOUBLE,
    skewness DOUBLE, kurtosis DOUBLE,
    sharpe_ratio DOUBLE,
    sortino_ratio DOUBLE,
    calmar_ratio DOUBLE,
    omega_ratio DOUBLE,
    beta DOUBLE,
    alpha_annualized DOUBLE,
    tracking_error DOUBLE,
    information_ratio DOUBLE,
    upside_capture DOUBLE,
    downside_capture DOUBLE,
    outperformance_consistency DOUBLE,
    PRIMARY KEY (scheme_code, as_of_date)
);

CREATE TABLE IF NOT EXISTS fund_scores (
    scheme_code INTEGER NOT NULL,
    as_of_date DATE NOT NULL,
    category VARCHAR NOT NULL,
    peer_count INTEGER NOT NULL,
    research_score DOUBLE,
    category_rank INTEGER,
    confidence_score DOUBLE,
    ranking_status VARCHAR NOT NULL,
    PRIMARY KEY (scheme_code, as_of_date)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id VARCHAR PRIMARY KEY,
    run_type VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    schemes_processed INTEGER,
    rows_inserted INTEGER,
    error_log VARCHAR,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS nav_reconciliation (
    scheme_code INTEGER,
    nav_date DATE,
    stored_nav DOUBLE,
    incoming_nav DOUBLE,
    discrepancy DOUBLE,
    status VARCHAR,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def get_connection(db_path: Optional[Path | str] = None) -> duckdb.DuckDBPyConnection:
    target_path = Path(db_path) if db_path is not None else DB_PATH
    ensure_data_directory()
    connection = duckdb.connect(str(target_path))
    connection.execute("PRAGMA enable_progress_bar = false")
    initialize_database(connection)
    return connection


def initialize_database(connection: Optional[duckdb.DuckDBPyConnection] = None):
    conn = connection or get_connection()
    conn.execute(SCHEMA_SQL)
    conn.commit()
    return conn


def insert_pipeline_run(
    connection: duckdb.DuckDBPyConnection,
    run_id: str,
    run_type: str,
    status: str,
    schemes_processed: int | None = None,
    rows_inserted: int | None = None,
    error_log: str | None = None,
):
    connection.execute(
        """
        INSERT INTO pipeline_runs (
            run_id, run_type, status, schemes_processed, rows_inserted, error_log
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (run_id) DO UPDATE SET
            run_type = excluded.run_type,
            status = excluded.status,
            schemes_processed = excluded.schemes_processed,
            rows_inserted = excluded.rows_inserted,
            error_log = excluded.error_log,
            executed_at = now()
        """,
        [run_id, run_type, status, schemes_processed, rows_inserted, error_log],
    )
    connection.commit()


def fetch_one(connection: duckdb.DuckDBPyConnection, query: str, params=()):
    return connection.execute(query, params).fetchone()
