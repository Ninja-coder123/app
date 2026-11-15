"""PDF extraction service for mutual-fund holdings."""

import os
import re
import tempfile

import pandas as pd
import pdfplumber


def parse_pdf_for_funds(pdf_bytes: bytes, password: str | None = None):
    """Extract fund names and current market values from a CAS PDF."""
    text = ""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpf:
            tmpf.write(pdf_bytes)
            tmpf.flush()
            tmp_path = tmpf.name

        try:
            if password:
                pdf = pdfplumber.open(tmp_path, password=password)
            else:
                pdf = pdfplumber.open(tmp_path)
        except Exception as e:
            return pd.DataFrame(columns=["name", "value"]), f"PDF open error: {e}"

        for p in pdf.pages:
            try:
                page_text = p.extract_text(layout=True) or ""
                text += page_text + "\n"
            except Exception:
                continue

        try:
            pdf.close()
        except Exception:
            pass

        pattern = re.compile(r"^(\S+)\s+(.*?)\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{1,4})\s+(\d{2}-[A-Za-z]{3}-\d{4})\s+([\d,]+\.\d{1,4})\s+([\d,]+\.\d{2})")

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        portfolio_dict = {}
        pending_name = ""

        for ln in lines:
            try:
                junk_keywords = ["Consolidated Account Summary", "Page", "Gain/Loss", "Folio No", "(INR)", "SoA Holdings"]
                if any(keyword in ln for keyword in junk_keywords):
                    continue

                match = pattern.search(ln)
                if match:
                    fund_name = match.group(2).strip()
                    if fund_name == "" and pending_name != "":
                        fund_name = pending_name
                    if fund_name:
                        market_value = float(match.group(7).replace(",", ""))
                        portfolio_dict[fund_name] = portfolio_dict.get(fund_name, 0.0) + market_value
                    pending_name = ""
                else:
                    if len(ln) > 10 and not ln.startswith("(+"):
                        pending_name = re.sub(r'\s+[\d\.,]+$', '', ln).strip()
            except Exception:
                continue

        rows = [(name, val) for name, val in portfolio_dict.items()]
        df = pd.DataFrame(rows, columns=["name", "value"]) if rows else pd.DataFrame(columns=["name", "value"])
        return df, text
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
