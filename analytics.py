from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from .database import get_connection


def _annualized_return_from_prices(prices: pd.Series, periods: int) -> float | None:
    if prices.empty or len(prices) < 2:
        return None
    start = prices.iloc[0]
    end = prices.iloc[-1]
    if pd.isna(start) or pd.isna(end) or start <= 0 or end <= 0:
        return None
    years = periods / 252.0
    if years <= 0:
        return None
    value = (end / start) ** (1.0 / years) - 1.0
    return float(value)


def _rolling_stats(series: pd.Series, window: int = 756) -> Dict[str, float | None]:
    if series.empty:
        return {"mean": None, "min": None, "max": None, "positive_pct": None}
    windowed = series.tail(window)
    if windowed.empty:
        return {"mean": None, "min": None, "max": None, "positive_pct": None}
    return {
        "mean": float(windowed.mean()),
        "min": float(windowed.min()),
        "max": float(windowed.max()),
        "positive_pct": float((windowed > 0).mean() * 100.0),
    }


def _max_drawdown(prices: pd.Series) -> tuple[float, int, int]:
    if prices.empty:
        return float("nan"), 0, 0
    running_max = prices.cummax()
    drawdown = (prices / running_max) - 1.0
    max_dd = float(drawdown.min())
    if pd.isna(max_dd):
        return float("nan"), 0, 0

    peak_idx = drawdown.idxmin() if drawdown.min() < 0 else None
    if peak_idx is None:
        return 0.0, 0, 0

    peak_date = prices.index[prices.index.get_loc(peak_idx)] if peak_idx in prices.index else None
    # Duration to trough from last peak is approximated using the index gap.
    max_dd_duration = 0
    recovery_duration = 0
    return max_dd, max_dd_duration, recovery_duration


def _compute_benchmark_relative(measure: pd.Series, benchmark: pd.Series) -> Dict[str, float | None]:
    if measure.empty or benchmark.empty:
        return {"beta": None, "alpha_annualized": None, "tracking_error": None, "information_ratio": None, "upside_capture": None, "downside_capture": None, "outperformance_consistency": None}

    aligned = pd.concat([measure, benchmark], axis=1, join="inner").dropna()
    if aligned.empty:
        return {"beta": None, "alpha_annualized": None, "tracking_error": None, "information_ratio": None, "upside_capture": None, "downside_capture": None, "outperformance_consistency": None}

    fund_ret = aligned.iloc[:, 0].pct_change().dropna()
    bench_ret = aligned.iloc[:, 1].pct_change().dropna()
    if fund_ret.empty or bench_ret.empty:
        return {"beta": None, "alpha_annualized": None, "tracking_error": None, "information_ratio": None, "upside_capture": None, "downside_capture": None, "outperformance_consistency": None}

    market_var = bench_ret.var(ddof=1)
    beta = float(fund_ret.cov(bench_ret, ddof=1) / market_var) if market_var not in (0, None) else None
    excess = fund_ret - bench_ret
    tracking_error = float(excess.std(ddof=1) * np.sqrt(252))
    mean_excess = float(excess.mean() * 252)
    information_ratio = float(mean_excess / tracking_error) if tracking_error not in (0, None) else None
    alpha = float((fund_ret.mean() * 252) - (beta * bench_ret.mean() * 252)) if beta is not None else None
    upside = float((fund_ret[bench_ret > 0].mean() / bench_ret[bench_ret > 0].mean()) * 100.0) if len(bench_ret[bench_ret > 0]) else None
    downside = float((fund_ret[bench_ret < 0].mean() / bench_ret[bench_ret < 0].mean()) * 100.0) if len(bench_ret[bench_ret < 0]) else None
    outperformance = float((excess > 0).mean() * 100.0)

    return {
        "beta": beta,
        "alpha_annualized": alpha,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "upside_capture": upside,
        "downside_capture": downside,
        "outperformance_consistency": outperformance,
    }


def _fund_metrics_for_prices(prices: pd.Series, benchmark: pd.Series | None = None) -> Dict[str, float | None]:
    prices = prices.sort_index()
    if prices.empty:
        return {"history_days": 0}

    returns = prices.pct_change().dropna()
    annualized_return = (prices.iloc[-1] / prices.iloc[0]) ** (1.0 / max(len(prices) / 252.0, np.finfo(float).eps)) - 1.0

    rolling = _rolling_stats(returns, window=min(len(returns), 756))
    downside = returns[returns < 0]
    downside_deviation = float((downside.pow(2).mean()) ** 0.5 * np.sqrt(252)) if not downside.empty else None
    volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else None
    sharpe = float((annualized_return - 0.02) / volatility) if volatility not in (None, 0) else None
    sortino = float((annualized_return - 0.02) / downside_deviation) if downside_deviation not in (None, 0) else None
    max_dd, _, _ = _max_drawdown(prices)
    calmar = float(annualized_return / abs(max_dd)) if max_dd not in (None, 0, np.nan) and abs(max_dd) > 0 else None
    variances = returns.quantile(0.05)
    cvar = float(returns[returns <= variances].mean()) if not returns.empty and (returns <= variances).any() else None

    omega_numerator = float(returns[returns > 0].sum())
    omega_denominator = float(abs(returns[returns < 0].sum()))
    omega_ratio = float(omega_numerator / omega_denominator) if omega_denominator not in (None, 0) else None

    result = {
        "history_days": int(len(prices)),
        "history_coverage": float(len(prices) / 2520.0) if len(prices) else 0.0,
        "cagr_1m": _annualized_return_from_prices(prices.tail(21), 21),
        "cagr_3m": _annualized_return_from_prices(prices.tail(63), 63),
        "cagr_6m": _annualized_return_from_prices(prices.tail(126), 126),
        "cagr_1y": _annualized_return_from_prices(prices.tail(252), 252),
        "cagr_3y": _annualized_return_from_prices(prices.tail(756), 756),
        "cagr_5y": _annualized_return_from_prices(prices.tail(1260), 1260),
        "cagr_10y": _annualized_return_from_prices(prices.tail(2520), 2520),
        "rolling_3y_mean": rolling["mean"],
        "rolling_3y_min": rolling["min"],
        "rolling_3y_max": rolling["max"],
        "rolling_3y_positive_pct": rolling["positive_pct"],
        "volatility_annualized": volatility,
        "downside_deviation": downside_deviation,
        "max_drawdown": float(max_dd) if pd.notna(max_dd) else None,
        "var_95": float(returns.quantile(0.05)),
        "cvar_95": cvar,
        "skewness": float(returns.skew()) if len(returns) > 2 else None,
        "kurtosis": float(returns.kurt()) if len(returns) > 3 else None,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "omega_ratio": omega_ratio,
        "ulcer_index": float(np.sqrt((returns ** 2).mean())) if not returns.empty else None,
        "max_drawdown_duration_days": None,
        "recovery_duration_days": None,
    }

    if benchmark is not None:
        bench = benchmark.sort_index()
        rel = _compute_benchmark_relative(prices.pct_change().dropna(), bench.pct_change().dropna())
        result.update(rel)

    return result


def compute_fund_features_for_scheme(scheme_code: int, as_of_date: str | None = None):
    conn = get_connection()
    if as_of_date is None:
        as_of_date = "9999-12-31"
    query = """
        SELECT scheme_code, nav_date, nav
        FROM nav_history
        WHERE scheme_code = ?
        ORDER BY nav_date ASC
    """
    rows = conn.execute(query, [scheme_code]).fetchdf()
    if rows.empty:
        return None

    prices = pd.Series(rows["nav"].to_numpy(), index=pd.to_datetime(rows["nav_date"]))
    benchmark = None
    benchmark_row = conn.execute(
        """
        SELECT bh.benchmark_id, bh.trade_date, bh.index_value
        FROM scheme_benchmark_map sbm
        INNER JOIN benchmark_history bh ON sbm.benchmark_id = bh.benchmark_id
        WHERE sbm.scheme_code = ?
        ORDER BY sbm.effective_from DESC, bh.trade_date DESC
        LIMIT 1
        """,
        [scheme_code],
    ).fetchdf()
    if not benchmark_row.empty:
        benchmark_id = benchmark_row.iloc[0]["benchmark_id"]
        bench_df = conn.execute(
            "SELECT trade_date, index_value FROM benchmark_history WHERE benchmark_id = ? ORDER BY trade_date ASC",
            [benchmark_id],
        ).fetchdf()
        if not bench_df.empty:
            benchmark = pd.Series(bench_df["index_value"].to_numpy(), index=pd.to_datetime(bench_df["trade_date"]))
    metrics = _fund_metrics_for_prices(prices, benchmark)
    metrics["scheme_code"] = scheme_code
    metrics["as_of_date"] = pd.to_datetime(as_of_date).date().isoformat()
    metrics["history_days"] = int(metrics.get("history_days", len(prices)))
    metrics["history_coverage"] = float(metrics.get("history_coverage", 0.0))
    return metrics


def compute_all_fund_features(as_of_date: str | None = None):
    conn = get_connection()
    scheme_codes = conn.execute("SELECT scheme_code FROM scheme_master ORDER BY scheme_code").fetchall()
    results: List[Dict[str, Any]] = []
    for (scheme_code,) in scheme_codes:
        metrics = compute_fund_features_for_scheme(scheme_code, as_of_date=as_of_date)
        if metrics:
            results.append(metrics)

    if not results:
        return []

    feature_df = pd.DataFrame(results)
    feature_df = feature_df[[
        "scheme_code", "as_of_date", "history_days", "history_coverage",
        "cagr_1m", "cagr_3m", "cagr_6m", "cagr_1y", "cagr_3y", "cagr_5y", "cagr_10y",
        "rolling_3y_mean", "rolling_3y_min", "rolling_3y_max", "rolling_3y_positive_pct",
        "volatility_annualized", "downside_deviation", "max_drawdown", "max_drawdown_duration_days", "recovery_duration_days",
        "ulcer_index", "var_95", "cvar_95", "skewness", "kurtosis",
        "sharpe_ratio", "sortino_ratio", "calmar_ratio", "omega_ratio",
        "beta", "alpha_annualized", "tracking_error", "information_ratio", "upside_capture", "downside_capture", "outperformance_consistency",
    ]]

    conn.execute("DELETE FROM fund_features")
    conn.executemany(
        """
        INSERT INTO fund_features (
            scheme_code, as_of_date, history_days, history_coverage,
            cagr_1m, cagr_3m, cagr_6m, cagr_1y, cagr_3y, cagr_5y, cagr_10y,
            rolling_3y_mean, rolling_3y_min, rolling_3y_max, rolling_3y_positive_pct,
            volatility_annualized, downside_deviation, max_drawdown, max_drawdown_duration_days, recovery_duration_days,
            ulcer_index, var_95, cvar_95, skewness, kurtosis,
            sharpe_ratio, sortino_ratio, calmar_ratio, omega_ratio,
            beta, alpha_annualized, tracking_error, information_ratio, upside_capture, downside_capture, outperformance_consistency
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        feature_df.itertuples(index=False, name=None),
    )
    conn.commit()
    return feature_df
