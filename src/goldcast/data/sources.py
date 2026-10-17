"""Fetchers for free sources. Each returns a float Series indexed by OBSERVATION date.

Publication lag is not applied here; `goldcast.data.store` keeps raw observations
and `goldcast.features.build` moves them to their availability dates.
"""

from __future__ import annotations

import io
import json
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
GPR_DAILY_XLS = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
TIMEOUT = 60


def _clean(s: pd.Series, name: str) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    s.name = name
    s.index.name = "date"
    return s.astype(float)


def fetch_yahoo(ticker: str, start: str) -> pd.Series:
    import yfinance as yf

    df = yf.download(ticker, start=start, auto_adjust=False, progress=False, threads=False)
    if df.empty:
        raise RuntimeError(f"yahoo returned no rows for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns a ticker column level
        close = close.iloc[:, 0]
    return _clean(close, ticker)


def fetch_fred(series_id: str, start: str) -> pd.Series:
    r = requests.get(FRED_CSV.format(id=series_id), timeout=TIMEOUT)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    date_col = df.columns[0]
    s = df.set_index(date_col)[series_id].replace(".", None)
    s = _clean(s, series_id)
    return s[s.index >= pd.Timestamp(start)]


def fetch_gpr(column: str, start: str) -> pd.Series:
    r = requests.get(GPR_DAILY_XLS, timeout=TIMEOUT)
    r.raise_for_status()
    # Parse from a temp file outside the repository; the download is data only.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gpr.xls"
        path.write_bytes(r.content)
        df = pd.read_excel(path)
    # The sheet has both `date` (datetime) and `DAY` (yyyymmdd integer); use either explicitly.
    if "date" in df.columns:
        idx = pd.to_datetime(df["date"])
    else:
        idx = pd.to_datetime(df["DAY"].astype(str), format="%Y%m%d")
    s = _clean(df.set_index(idx)[column], column)
    return s[s.index >= pd.Timestamp(start)]


# --- R-004 sources. Raw downloads are cached under data/cache so a re-run does not refetch.

CACHE = Path(__file__).resolve().parents[3] / "data" / "cache"
UA = {"User-Agent": "goldcast-research/0.1 (personal research)"}
CFTC_URL = "https://www.cftc.gov/files/dea/history/{file}"
# (history file, yearly file pattern, first yearly year) per report family; futures only.
CFTC_FILES = {
    "disagg": ("fut_disagg_txt_hist_2006_2016.zip", "fut_disagg_txt_{y}.zip", 2017),
    "legacy": ("deacot1986_2016.zip", "deacot{y}.zip", 2017),
}
# (long, short, open interest) columns, matched after lower-casing and keeping only [a-z0-9].
CFTC_COLS = {
    "disagg": ("mmoneypositionslongall", "mmoneypositionsshortall", "openinterestall"),
    "legacy": ("noncommercialpositionslongall", "noncommercialpositionsshortall", "openinterestall"),
}
_last_request = [0.0]


def _polite_get(url: str) -> requests.Response:
    """GET with a named User-Agent and at least 2 s between requests."""
    wait = 2.0 - (time.time() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    _last_request[0] = time.time()
    r.raise_for_status()
    return r


def _cached(name: str, url: str, refresh: bool = False) -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if path.exists() and not refresh:
        return path.read_bytes()
    data = _polite_get(url).content
    path.write_bytes(data)
    return data


def _norm(c: str) -> str:
    return "".join(ch for ch in c.lower() if ch.isalnum())


def fetch_cftc(spec_id: str, start: str) -> pd.Series:
    """Net position as a share of open interest for one contract, by report (Tuesday) date.

    `spec_id` is "<family>:<contract code>", e.g. "disagg:088691" (COMEX gold).
    The current year's file is refetched each time; older files come from the cache.
    """
    family, code = spec_id.split(":")
    hist, yearly, first = CFTC_FILES[family]
    year = pd.Timestamp.today().year
    files = [hist] + [yearly.format(y=y) for y in range(first, year + 1)]
    long_c, short_c, oi_c = CFTC_COLS[family]
    frames = []
    for f in files:
        blob = _cached(f"cftc_{f}", CFTC_URL.format(file=f), refresh=str(year) in f)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for member in z.namelist():
                df = pd.read_csv(z.open(member), dtype=str, low_memory=False)
                df.columns = [_norm(c) for c in df.columns]
                code_col = next(c for c in df.columns if c.startswith("cftccontractmarketcode"))
                date_col = next(c for c in df.columns if "yyyy" in c and "date" in c)
                df = df[df[code_col].str.strip() == code]
                frames.append(df[[date_col, long_c, short_c, oi_c]].set_axis(["date", "l", "s", "oi"], axis=1))
    df = pd.concat(frames)
    num = df[["l", "s", "oi"]].apply(lambda c: pd.to_numeric(c.str.strip(), errors="coerce"))
    s = pd.Series(((num["l"] - num["s"]) / num["oi"]).values, index=pd.to_datetime(df["date"].str.strip()))
    s = _clean(s, spec_id)
    return s[s.index >= pd.Timestamp(start)]


def _trends_window(keyword: str, tf: str) -> pd.Series:
    from pytrends.request import TrendReq

    tr = TrendReq(hl="en-US", tz=0, timeout=(10, 30), requests_args={"headers": UA})
    for attempt in range(4):
        time.sleep(5 + 30 * attempt)  # Google answers bursts with 429; back off
        try:
            tr.build_payload([keyword], timeframe=tf, geo="")
            return tr.interest_over_time()[keyword].astype(float)
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise


def fetch_trends(keyword: str, start: str) -> pd.Series:
    """Weekly Google Trends interest, chained across overlapping 4-year windows.

    Google scales each window to 0-100, so only log changes inside one window compare.
    Each week's log change comes from the earliest window holding both weeks; the
    result is exp(cumsum), meaningful only through its log changes. Observation date
    is the Saturday that ends the Google week (weeks start on Sunday).
    """
    first = max(pd.Timestamp(start), pd.Timestamp("2004-01-01"))
    today = pd.Timestamp.today().normalize()
    windows, a = [], first
    while a < today:
        windows.append((a, min(a + pd.DateOffset(years=4), today)))
        a = a + pd.DateOffset(years=3)
    CACHE.mkdir(parents=True, exist_ok=True)
    dlogs = []
    for a, b in windows:
        tf = f"{a.date()} {b.date()}"
        path = CACHE / f"trends_{_norm(keyword)}_{a.date()}_{b.date()}.json"
        if path.exists():
            d = pd.Series(json.loads(path.read_text()))
            d.index = pd.to_datetime(d.index)
        else:
            d = _trends_window(keyword, tf)
            path.write_text(json.dumps({str(k.date()): v for k, v in d.items()}))
        if (d.index[1] - d.index[0]).days != 7:
            raise RuntimeError(f"trends window {tf} is not weekly")
        dlogs.append(np.log(d.where(d > 0)).diff().dropna())
    dl = pd.concat(dlogs)
    dl = dl[~dl.index.duplicated(keep="first")].sort_index()  # earliest window wins
    chain = np.exp(dl.cumsum())
    chain.index = chain.index + pd.Timedelta(days=6)
    s = _clean(chain, keyword)
    return s[s.index >= pd.Timestamp(start)]


FETCHERS = {"yahoo": fetch_yahoo, "fred": fetch_fred, "gpr": fetch_gpr,
            "cftc": fetch_cftc, "trends": fetch_trends}
