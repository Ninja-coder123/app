"""Mutual Fund Research Terminal and Analytics Engine."""

from .database import get_connection, initialize_database
from .ingest import bootstrap_history, fetch_daily_amfi_file, upsert_scheme_master, run_daily_pipeline
from .analytics import compute_fund_features_for_scheme, compute_all_fund_features
from .scoring import compute_peer_scores

__all__ = [
    "get_connection",
    "initialize_database",
    "bootstrap_history",
    "fetch_daily_amfi_file",
    "upsert_scheme_master",
    "run_daily_pipeline",
    "compute_fund_features_for_scheme",
    "compute_all_fund_features",
    "compute_peer_scores",
]
