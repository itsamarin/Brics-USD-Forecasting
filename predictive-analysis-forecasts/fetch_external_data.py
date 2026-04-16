"""
Fetch external reference data required for accuracy improvements.

Downloads (via yfinance — no API key needed):
  - DXY_Monthly.csv        US Dollar Index monthly close
  - Gold_Spot_Monthly.csv  Gold spot price USD/troy oz monthly average
  - WTI_Monthly.csv        WTI crude oil spot price USD/bbl monthly average

Builds from embedded SWIFT RMB Tracker public reports:
  - SWIFT_USD_Share.csv    USD share of international SWIFT payments (%)

Run once before predictive_analysis_forecast.py:
    pip install yfinance
    python3 fetch_external_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

START_DATE = '2020-01-01'
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Yahoo Finance downloader
# ---------------------------------------------------------------------------

def fetch_yahoo_monthly(ticker, label, start=START_DATE):
    """
    Download monthly prices from Yahoo Finance and return a clean DataFrame.
    Returns None if yfinance is not installed or download fails.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed. Run: pip install yfinance>=0.2.0")
        return None

    end = datetime.today().strftime('%Y-%m-%d')
    try:
        raw = yf.download(ticker, start=start, end=end,
                          interval='1mo', auto_adjust=True, progress=False)
        if raw.empty:
            print(f"  No data returned for {ticker}")
            return None

        # Flatten multi-level columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        monthly = raw[['Close']].copy()
        monthly.index = pd.to_datetime(monthly.index).to_period('M').to_timestamp()
        monthly.columns = [label]
        monthly.index.name = 'Date'
        return monthly.reset_index()

    except Exception as exc:
        print(f"  Download failed for {ticker}: {exc}")
        return None


# ---------------------------------------------------------------------------
# SWIFT data (embedded — from SWIFT RMB Tracker public monthly reports)
# ---------------------------------------------------------------------------

def build_swift_data():
    """
    USD share of international SWIFT payments by value (%).
    Source: SWIFT RMB Tracker monthly reports (publicly available).
    Quarterly anchor values are linearly interpolated to monthly frequency.
    """
    # (year, month, usd_share_pct)
    anchors = [
        (2021,  1, 38.23), (2021,  4, 38.91), (2021,  7, 39.45), (2021, 10, 39.82),
        (2022,  1, 40.51), (2022,  4, 41.73), (2022,  7, 42.88), (2022, 10, 41.35),
        (2023,  1, 41.14), (2023,  4, 42.69), (2023,  7, 46.46), (2023, 10, 47.23),
        (2024,  1, 47.35), (2024,  4, 47.82), (2024,  7, 48.03), (2024, 10, 47.54),
        (2025,  1, 47.20), (2025,  4, 46.90),
    ]
    df = pd.DataFrame(anchors, columns=['Year', 'Month', 'USD_Share_Pct'])
    df['Date'] = pd.to_datetime(dict(year=df.Year, month=df.Month, day=1))
    df = df[['Date', 'USD_Share_Pct']].sort_values('Date').set_index('Date')

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq='MS')
    df = df.reindex(full_idx).interpolate(method='linear')
    df.index.name = 'Date'
    df['USD_Share_Pct'] = df['USD_Share_Pct'].round(2)
    return df.reset_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Fetching external reference data for USD dominance model")
    print("=" * 60)

    tasks = [
        ('DX-Y.NYB',  'DXY',          'DXY_Monthly.csv',        'DXY (US Dollar Index)'),
        ('GC=F',       'Gold_Spot_USD', 'Gold_Spot_Monthly.csv',  'Gold spot price USD/oz'),
        ('CL=F',       'WTI_Spot_USD',  'WTI_Monthly.csv',        'WTI crude oil USD/bbl'),
    ]

    for ticker, label, filename, description in tasks:
        print(f"\n  Downloading {description} ({ticker})...")
        df = fetch_yahoo_monthly(ticker, label)
        if df is not None:
            path = os.path.join(OUT_DIR, filename)
            df.to_csv(path, index=False)
            print(f"  Saved {filename} ({len(df)} months, "
                  f"{df['Date'].min()} – {df['Date'].max()})")
        else:
            print(f"  Skipped {filename} — will use no-deflation fallback in main script")

    print("\n  Building SWIFT USD share data (embedded from public reports)...")
    swift = build_swift_data()
    path  = os.path.join(OUT_DIR, 'SWIFT_USD_Share.csv')
    swift.to_csv(path, index=False)
    print(f"  Saved SWIFT_USD_Share.csv ({len(swift)} months)")

    print("\n" + "=" * 60)
    print("Done. Now run: python3 predictive_analysis_forecast.py")
    print("=" * 60)
