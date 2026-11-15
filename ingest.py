from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Iterable, List, Sequence

import pandas as pd
import requests

from .config import AMFI_NAV_URL, MFAPI_BASE_URL
from .database import get_connection, insert_pipeline_run


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def upsert_scheme_master(connection, scheme_record: dict):
    scheme_code = _safe_int(scheme_record.get("scheme_code") or scheme_record.get("schemeCode"))
    if scheme_code is None:
        return None

    row = {
        "scheme_code": scheme_code,
        "isin": scheme_record.get("isin") or scheme_record.get("isin_div_growth") or None,
        "amc_name": scheme_record.get("amc_name") or scheme_record.get("amcName") or "UNKNOWN",
        "scheme_name": scheme_record.get("scheme_name") or scheme_record.get("schemeName") or "UNKNOWN",
        "canonical_category": (scheme_record.get("canonical_category") or scheme_record.get("category") or "UNCLASSIFIED").strip(),
        "plan": (scheme_record.get("plan") or "Direct").strip(),
        "option_type": (scheme_record.get("option_type") or "Growth").strip(),
        "is_active": bool(scheme_record.get("is_active", True)),
        "first_nav_date": scheme_record.get("first_nav_date"),
        "latest_nav_date": scheme_record.get("latest_nav_date"),
    }

    connection.execute(
        """
        INSERT INTO scheme_master (
            scheme_code, isin, amc_name, scheme_name, canonical_category,
            plan, option_type, is_active, first_nav_date, latest_nav_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (scheme_code) DO UPDATE SET
            isin = excluded.isin,
            amc_name = excluded.amc_name,
            scheme_name = excluded.scheme_name,
            canonical_category = excluded.canonical_category,
            plan = excluded.plan,
            option_type = excluded.option_type,
            is_active = excluded.is_active,
            first_nav_date = COALESCE(excluded.first_nav_date, scheme_master.first_nav_date),
            latest_nav_date = COALESCE(excluded.latest_nav_date, scheme_master.latest_nav_date),
            updated_at = now()
        """,
        [
            row["scheme_code"],
            row["isin"],
            row["amc_name"],
            row["scheme_name"],
            row["canonical_category"],
            row["plan"],
            row["option_type"],
            row["is_active"],
            row["first_nav_date"],
            row["latest_nav_date"],
        ],
    )
    connection.commit()
    return scheme_code


def fetch_master_index():
    response = requests.get(f"{MFAPI_BASE_URL}/mf", timeout=30)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return payload["data"]
        return list(payload.values())
    return []


def bootstrap_history(scheme_code: int | str, years: int = 13):
    conn = get_connection()
    url = f"{MFAPI_BASE_URL}/mf/{scheme_code}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else []
    if not isinstance(data, list):
        raise ValueError(f"No historical NAV dataset returned for scheme_code={scheme_code}")

    rows = []
    for entry in data:
        nav_date = entry.get("date") or entry.get("nav_date")
        nav_value = entry.get("nav") or entry.get("value")
        if nav_date is None or nav_value is None:
            continue
        nav_dt = pd.to_datetime(nav_date, errors="coerce")
        if pd.isna(nav_dt):
            continue
        rows.append((int(scheme_code), nav_dt.date().isoformat(), float(nav_value), "MFAPI_BOOTSTRAP", _hash_bytes(json.dumps(entry, sort_keys=True).encode("utf-8"))))

    if rows:
        conn.executemany(
            """
            INSERT INTO nav_history (scheme_code, nav_date, nav, source, source_hash)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (scheme_code, nav_date) DO UPDATE SET
                nav = excluded.nav,
                source = excluded.source,
                source_hash = excluded.source_hash,
                ingested_at = now()
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def _parse_amfi_nav_lines(file_content: str) -> List[tuple[int, str, float]]:
    rows: List[tuple[int, str, float]] = []
    for line in str(file_content).splitlines():
        if not line.strip():
            continue
        cleaned = line.strip()
        for delim in (";", "|", ","):
            if delim in cleaned:
                parts = [part.strip() for part in cleaned.split(delim)]
                break
        else:
            parts = [cleaned]

        if len(parts) < 4:
            continue

        scheme_code = _safe_int(parts[0])
        if scheme_code is None:
            continue

        nav_date = None
        nav_value = None
        for candidate in reversed(parts):
            parsed_dt = pd.to_datetime(candidate, errors="coerce")
            if pd.notna(parsed_dt):
                nav_date = parsed_dt.date().isoformat()
                break

        for candidate in reversed(parts):
            val = _safe_float(candidate)
            if val is not None:
                nav_value = val
                break

        if nav_date and nav_value is not None:
            rows.append((scheme_code, nav_date, nav_value))
    return rows


def fetch_daily_amfi_file():
    response = requests.get(AMFI_NAV_URL, timeout=30)
    response.raise_for_status()
    payload = response.content
    return payload, _hash_bytes(payload)


def run_daily_pipeline():
    conn = get_connection()
    run_id = datetime.utcnow().strftime("AMFI-%Y%m%d-%H%M%S")
    try:
        payload, file_hash = fetch_daily_amfi_file()
        file_content = payload.decode("utf-8", errors="replace")
        rows = _parse_amfi_nav_lines(file_content)

        inserted = 0
        for scheme_code, nav_date, nav_value in rows:
            existing = conn.execute(
                "SELECT 1 FROM nav_history WHERE scheme_code = ? AND nav_date = ?",
                [scheme_code, nav_date],
            ).fetchone()
            if existing:
                continue
            conn.execute(
                "INSERT INTO nav_history (scheme_code, nav_date, nav, source, source_hash) VALUES (?, ?, ?, 'AMFI_DAILY', ?)",
                [scheme_code, nav_date, nav_value, file_hash],
            )
            inserted += 1

        conn.commit()
        insert_pipeline_run(conn, run_id, "DAILY_AMFI", "SUCCESS", schemes_processed=len(rows), rows_inserted=inserted)
        return {"run_id": run_id, "rows_inserted": inserted, "schemes_processed": len(rows)}
    except Exception as exc:  # pragma: no cover - runtime guardrail
        insert_pipeline_run(conn, run_id, "DAILY_AMFI", "FAILED", error_log=str(exc))
        raise


def ingest_master_scheme_directory():
    conn = get_connection()
    payload = fetch_master_index()
    for scheme in payload:
        record = {
            "scheme_code": scheme.get("schemeCode") if isinstance(scheme, dict) else None,
            "scheme_name": scheme.get("schemeName") if isinstance(scheme, dict) else None,
            "amc_name": scheme.get("amc") or scheme.get("amcName") if isinstance(scheme, dict) else None,
            "canonical_category": scheme.get("category") if isinstance(scheme, dict) else "UNCLASSIFIED",
            "plan": "Direct",
            "option_type": "Growth",
            "is_active": True,
        }
        upsert_scheme_master(conn, record)
    return len(payload)
