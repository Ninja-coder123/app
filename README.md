# Mutual Fund Research Terminal

This project implements the institutional-grade mutual fund analytics engine described in the design brief stored at `Act as an expert Python Developer..txt`.

## Architecture

- DuckDB-backed storage in `data/mf_terminal.duckdb`
- Historical bootstrap ingestion from `https://api.mfapi.in/mf/{scheme_code}`
- Daily production ingestion from the AMFI NAV file
- Feature engineering and risk metrics in `mf_research_terminal/analytics.py`
- Peer scoring and ranking in `mf_research_terminal/scoring.py`

## Run

```bash
python run_mf_terminal.py --bootstrap
python run_mf_terminal.py --daily
```

The CLI uses a fail-safe pipeline. Daily AMFI ingestion aborts without wiping the previous database if the feed fails.
