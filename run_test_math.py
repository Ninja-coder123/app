import pandas as pd
import os

def create_mock_parquet():
    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Create fake institutional funds
    data = {
        "fund_code": [120000, 120001, 120002, 120003],
        "scheme_name": [
            "HDFC Top 100 Fund - Direct Plan", 
            "SBI Liquid Fund - Direct Plan",
            "ICICI Prudential Bluechip Fund - Direct",
            "Axis Liquid Fund - Direct"
        ],
        "category": [
            "Equity Scheme - Large Cap Fund", 
            "Debt Scheme - Liquid Fund",
            "Equity Scheme - Large Cap Fund",
            "Debt Scheme - Liquid Fund"
        ],
        "pr_fqs_score": [88.5, 92.0, 85.0, 90.5],
        "sharpe": [1.45, 0.85, 1.25, 0.90],
        "sortino": [1.80, 1.10, 1.60, 1.15],
        "volatility": [12.5, 1.2, 13.0, 1.0],
        "beta": [0.95, 0.05, 1.02, 0.03],
        "latest_nav": [150.25, 1050.10, 180.00, 1100.50]
    }
    
    # Build dataframe and export directly to Parquet
    df = pd.DataFrame(data)
    df.to_parquet("data/screener_database.parquet", index=False)
    print("✅ Hardcoded mock Parquet file created successfully!")

if __name__ == "__main__":
    create_mock_parquet()