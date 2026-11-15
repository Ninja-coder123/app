from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "mf_terminal.duckdb"

MFAPI_BASE_URL = "https://api.mfapi.in"
AMFI_NAV_URL = "https://www.amfiindia.com/spages/NAVAll.txt"

DEFAULT_AS_OF_DATE = "latest"


def ensure_data_directory() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
