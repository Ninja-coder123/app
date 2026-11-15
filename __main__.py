from __future__ import annotations

import argparse

from .service import daily_run, initialize_and_bootstrap


def main():
    parser = argparse.ArgumentParser(description="Mutual Fund Research Terminal")
    parser.add_argument("--bootstrap", action="store_true", help="Fetch historical MFAPI data and initialize the terminal")
    parser.add_argument("--daily", action="store_true", help="Fetch the latest AMFI NAV file and ingest only new records")
    args = parser.parse_args()

    if args.bootstrap:
        result = initialize_and_bootstrap()
        print(result)
        return

    if args.daily:
        result = daily_run()
        print(result)
        return

    print("No action selected. Use --bootstrap or --daily.")


if __name__ == "__main__":
    main()
