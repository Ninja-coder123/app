from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .database import get_connection


def _percentile_rank(value: float, series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float((series < value).mean() * 100.0)


def compute_peer_scores(as_of_date: str | None = None):
    conn = get_connection()
    if as_of_date is None:
        as_of_date = pd.Timestamp.utcnow().date().isoformat()

    feature_df = conn.execute(
        """
        SELECT ff.scheme_code, ff.as_of_date, sm.canonical_category AS category,
               ff.sharpe_ratio, ff.sortino_ratio, ff.volatility_annualized,
               ff.max_drawdown, ff.cagr_1y, ff.alpha_annualized, ff.downside_deviation,
               ff.beta, ff.tracking_error, ff.information_ratio, ff.omega_ratio,
               ff.outperformance_consistency
        FROM fund_features ff
        INNER JOIN scheme_master sm ON ff.scheme_code = sm.scheme_code
        WHERE ff.as_of_date = ?
        """,
        [as_of_date],
    ).fetchdf()

    if feature_df.empty:
        return pd.DataFrame(columns=["scheme_code", "as_of_date", "category", "peer_count", "research_score", "category_rank", "confidence_score", "ranking_status"])

    rows: List[Dict[str, object]] = []
    for category, group in feature_df.groupby("category", dropna=False):
        if len(group) < 3:
            for _, row in group.iterrows():
                rows.append({
                    "scheme_code": int(row["scheme_code"]),
                    "as_of_date": as_of_date,
                    "category": category,
                    "peer_count": int(len(group)),
                    "research_score": None,
                    "category_rank": None,
                    "confidence_score": 0.0,
                    "ranking_status": "INSUFFICIENT_PEERS",
                })
            continue

        # Model a category-specific weighted score built from percentile ranks.
        for metric in ["cagr_1y", "sharpe_ratio", "sortino_ratio", "omega_ratio", "outperformance_consistency", "alpha_annualized"]:
            group[metric] = pd.to_numeric(group[metric], errors="coerce")

        score_df = group.copy()
        score_df["research_num"] = 0.0
        for metric in ["cagr_1y", "sharpe_ratio", "sortino_ratio", "omega_ratio", "outperformance_consistency", "alpha_annualized"]:
            val_series = score_df[metric].replace([np.inf, -np.inf], np.nan)
            score_df[metric] = val_series
            score_df["research_num"] += val_series.rank(pct=True, na_option="keep") * (1.0 / 6.0)

        score_df["research_score"] = (score_df["research_num"] * 100.0).clip(lower=0.0, upper=100.0)
        score_df = score_df.sort_values("research_score", ascending=False).reset_index(drop=True)
        score_df["category_rank"] = np.arange(1, len(score_df) + 1)
        score_df["peer_count"] = len(score_df)
        score_df["ranking_status"] = "VALID"
        score_df["confidence_score"] = 100.0

        for _, row in score_df.iterrows():
            rows.append({
                "scheme_code": int(row["scheme_code"]),
                "as_of_date": as_of_date,
                "category": category,
                "peer_count": int(row["peer_count"]),
                "research_score": float(row["research_score"]),
                "category_rank": int(row["category_rank"]),
                "confidence_score": float(row["confidence_score"]),
                "ranking_status": str(row["ranking_status"]),
            })

    output = pd.DataFrame(rows)
    if output.empty:
        return output

    conn.execute("DELETE FROM fund_scores WHERE as_of_date = ?", [as_of_date])
    conn.executemany(
        """
        INSERT INTO fund_scores (
            scheme_code, as_of_date, category, peer_count, research_score,
            category_rank, confidence_score, ranking_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        output[["scheme_code", "as_of_date", "category", "peer_count", "research_score", "category_rank", "confidence_score", "ranking_status"]].itertuples(index=False, name=None),
    )
    conn.commit()
    return output
