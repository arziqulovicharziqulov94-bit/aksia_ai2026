# -*- coding: utf-8 -*-
"""
AKSIYA AI v4 — bozorni oʻzi skaner qiladigan va oʻzini oʻzi tekshiradigan ML tizimi
===================================================================================

Nima qiladi
-----------
1. ~170 ta yirik AQSh aksiyasi (NYSE/NASDAQ) va SPY narx tarixini yuklaydi (Yahoo Finance, zaxira: Stooq).
2. Har bir aksiya uchun texnik belgilar (momentum, volatillik, trend, Wyckoff fazasi, sham modellari,
   Gann burchagi, hajm ...) hisoblaydi — faqat OʻTMISH maʼlumotidan (kelajakka qarash yoʻq).
3. Gradient boosting modelini "purged walk-forward" usulida sinaydi: har bir test davri uchun model faqat
   undan OLDINGI maʼlumotda oʻqitiladi; etiketlari ustma-ust tushgan oraliq (embargo) olib tashlanadi.
4. Modelning tarixiy ustunligini oʻlchaydi (AUC, IC, t-statistika, backtest) va natijani HALOL koʻrsatadi:
   ustunlik topilmasa — "dalil yoʻq" deydi.
5. Eng yuqori ball olgan aksiyalar roʻyxatini va birinchisining chuqur tahlilini beradi; istalgan tikerni
   alohida tahlil qiladi (narx diapazoni, support/resistance, Wyckoff, Gann, sham, fundamental, yangilik).

Ishga tushirish
---------------
    pip install yfinance scikit-learn pandas numpy matplotlib vaderSentiment requests "gradio>=6,<7"
    python app.py             # http://localhost:7860
    python app.py --share     # vaqtinchalik ochiq havola (Colab uchun)
    python app.py --selftest  # sintetik maʼlumotda metodologiya tekshiruvi

Muhim: bu moliyaviy maslahat emas. Hech bir model kelajakni kafolatlamaydi.
"""
from __future__ import annotations

import os
import sys
import io
import re
import math
import time
import json
import base64
import pickle
import threading
import tempfile
import warnings
import argparse
import html as _html
import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Kichik serverlarda (masalan, 512 MB) xotirani tejash: BLAS/OpenMP oqimlarini cheklaymiz.
# numpy/sklearn import qilinishidan OLDIN oʻrnatilishi shart; foydalanuvchi oʻzi bergan qiymat ustun turadi.
for _k, _v in (("OPENBLAS_NUM_THREADS", "2"), ("MKL_NUM_THREADS", "2"), ("OMP_NUM_THREADS", "4"),
               ("GRADIO_ANALYTICS_ENABLED", "False")):
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings("ignore")

# ======================================================================================
# 0. SOZLAMALAR
# ======================================================================================
APP_VERSION = "4.1"
BENCH = "SPY"                      # benchmark (bozor)
DATA_YEARS = 8                     # necha yillik tarix yuklanadi
WARMUP = 260                       # belgilar toʻliq hisoblanishi uchun kerakli kunlar (52 hafta + zaxira)
STRIDE = 5                         # namunalar har 5-savdo kunida olinadi (haftalik)
HORIZONS = [5, 10, 20, 40, 60, 90, 125]          # savdo kunlarida
HORIZON_UZ = {5: "1 hafta", 10: "2 hafta", 20: "1 oy", 40: "2 oy", 60: "3 oy", 90: "4,5 oy", 125: "6 oy"}
PRIMARY_H = 20                     # asosiy ufq (reyting va backtest shu boʻyicha)
N_FOLDS = 4                        # walk-forward test davrlari soni
TOP_N = 5                          # tavsiya roʻyxati va backtest portfeli hajmi
COST_BPS = 10.0                    # har bir tomonga tranzaksiya xarajati (baza punkti)
MIN_NAMES = 20                     # sana boʻyicha kamida shuncha aksiya boʻlsa, kesim hisobga olinadi
T_WEAK = 2.0                       # IC t-statistikasi: zaif dalil chegarasi
T_STRONG = 2.7                     # kuchli dalil chegarasi (7 ufq sinalgani uchun qatʼiyroq)
REFRESH_HOURS = 12                 # tizim har necha soatda yangilanadi
MC_PATHS = 20000                   # Monte-Karlo yoʻllari
MC_DRIFT = 0.0003                  # kunlik "bozor-prior" drift (~7,8% yillik); aksiyaning oʻz oʻtmish oʻrtachasi EMAS
MC_NU = 4                          # t-taqsimot erkinlik darajasi (ogʻir dumlar)
HGB_PARAMS = dict(max_depth=2, learning_rate=0.03, max_iter=150, min_samples_leaf=500,
                  l2_regularization=20.0, early_stopping=False, random_state=0)   # juda qatʼiy regullarizatsiya
LOGIT_C = 0.05                     # logistik regressiya regullarizatsiyasi (kichik = kuchli)
CACHE_DIR = os.environ.get("AKSIYA_CACHE", os.path.join(tempfile.gettempdir(), "aksiya_ai_cache"))

_UNIVERSE_TXT = """
AAPL MSFT NVDA GOOGL AMZN META TSLA AVGO ORCL ADBE CRM AMD INTC CSCO QCOM TXN IBM NOW INTU AMAT MU LRCX KLAC
PANW SNPS CDNS ANET ADI NFLX UBER PYPL FTNT MRVL NXPI ADSK WDAY TTD CRWD DDOG NET ZS TEAM MELI SPOT SHOP
HD MCD NKE SBUX LOW TJX BKNG MAR ORLY AZO CMG ROST LULU F GM DIS
WMT COST KO PEP PG PM MDLZ TGT CL MO KMB GIS KHC
JPM BAC WFC GS MS C V MA AXP BLK SCHW BRK-B SPGI CB PGR MMC USB PNC TFC COF ICE CME AON
UNH JNJ LLY PFE MRK ABBV TMO ABT DHR BMY AMGN GILD ISRG VRTX REGN MDT CVS CI ELV ZTS SYK BSX HCA DXCM IDXX MRNA BIIB
CAT DE BA HON UPS UNP LMT RTX GE MMM ETN ADP FDX CSX NOC GD WM ITW EMR PH
XOM CVX COP SLB EOG OXY MPC PSX NEE DUK SO D AEP T VZ TMUS CMCSA LIN SHW APD FCX NEM NUE DOW AMT PLD CCI EQIX O SPG
"""
UNIVERSE: List[str] = list(dict.fromkeys(_UNIVERSE_TXT.split()))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def trim_memory() -> None:
    """Ishlatilmayotgan xotirani operatsion tizimga qaytaradi (arzon serverlarda 512 MB chegarasi uchun)."""
    try:
        import gc
        import ctypes
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _limit_malloc_arenas() -> None:
    """glibc: koʻp oqimli dasturda xotira parchalanishini kamaytiradi (M_ARENA_MAX=2). Boshqa tizimlarda jim oʻtadi."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").mallopt(-8, 2)
    except Exception:
        pass


_limit_malloc_arenas()


class DataError(Exception):
    """Foydalanuvchiga tushunarli xabar bilan maʼlumot xatosi."""


# ======================================================================================
# 1. MAʼLUMOT YUKLASH (Yahoo Finance -> zaxira: Stooq)
# ======================================================================================
_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper().replace(" ", "").replace(".", "-")
    t = re.sub(r"[^A-Z0-9\-\^=]", "", t)
    if not t or len(t) > 12:
        raise DataError("Tiker notoʻgʻri. Misol: AAPL, NVDA, TSLA, BRK-B")
    return t


def _clean_ohlcv(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Turli manbalardan kelgan jadvalni yagona OHLCV koʻrinishiga keltiradi."""
    if df is None or len(df) == 0:
        return None
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    ren = {}
    for c in d.columns:
        cl = str(c).strip().lower()
        for k in _OHLCV:
            if cl == k.lower():
                ren[c] = k
    d = d.rename(columns=ren)
    if "Close" not in d.columns:
        return None
    for k in _OHLCV:
        if k not in d.columns:
            d[k] = d["Close"] if k != "Volume" else 0.0
    d = d[_OHLCV].apply(pd.to_numeric, errors="coerce")
    idx = pd.to_datetime(d.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    d.index = idx.normalize()
    d = d[~d.index.duplicated(keep="last")].sort_index()
    d = d.dropna(subset=["Close"])
    d = d[d["Close"] > 0]
    for k in ("Open", "High", "Low"):
        d[k] = d[k].fillna(d["Close"])
    d["High"] = d[["Open", "High", "Low", "Close"]].max(axis=1)
    d["Low"] = d[["Open", "High", "Low", "Close"]].min(axis=1)
    d["Volume"] = d["Volume"].fillna(0.0).clip(lower=0)
    d = d.astype(float)
    return d if len(d) >= 30 else None


def _drop_partial_today(d: pd.DataFrame) -> pd.DataFrame:
    """Bozor ochiq paytida Yahoo bugungi tugallanmagan shamni ham beradi — uni olib tashlaymiz."""
    try:
        from zoneinfo import ZoneInfo
        now = dt.datetime.now(ZoneInfo("America/New_York"))
        if d.index[-1].date() == now.date() and (now.hour, now.minute) < (16, 20):
            return d.iloc[:-1]
    except Exception:
        pass
    return d


def _start_date(years: float) -> str:
    return (dt.date.today() - dt.timedelta(days=int(365.25 * years) + 5)).isoformat()


def _split_bulk(raw: Optional[pd.DataFrame], tickers: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if raw is None or len(raw) == 0:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        l0 = set(map(str, raw.columns.get_level_values(0)))
        l1 = set(map(str, raw.columns.get_level_values(1)))
        for t in tickers:
            try:
                if t in l0:
                    sub = raw[t]
                elif t in l1:
                    sub = raw.xs(t, axis=1, level=1)
                else:
                    continue
                d = _clean_ohlcv(sub)
                if d is not None:
                    out[t] = d
            except Exception:
                continue
    elif len(tickers) == 1:
        d = _clean_ohlcv(raw)
        if d is not None:
            out[tickers[0]] = d
    return out


def _yf_bulk(tickers: List[str], years: float, batch: int = 40,
             cb: Optional[Callable[[float, str], None]] = None) -> Dict[str, pd.DataFrame]:
    import yfinance as yf
    out: Dict[str, pd.DataFrame] = {}
    start = _start_date(years)
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        raw = None
        for attempt in range(2):
            try:
                raw = yf.download(chunk, start=start, interval="1d", auto_adjust=True,
                                  group_by="ticker", threads=True, progress=False)
                if raw is not None and len(raw) > 0:
                    break
            except Exception as e:  # tarmoq / rate-limit
                log("yfinance xato:", type(e).__name__, str(e)[:120])
            time.sleep(1.5)
        out.update(_split_bulk(raw, chunk))
        if cb:
            cb(min(1.0, (i + len(chunk)) / max(1, len(tickers))),
               f"Narx tarixi yuklanmoqda: {min(i + batch, len(tickers))}/{len(tickers)}")
    return out


def _stooq_one(ticker: str, years: float) -> Optional[pd.DataFrame]:
    import requests
    sym = ticker.lower().replace("^", "") + ".us"
    d2 = dt.date.today()
    d1 = d2 - dt.timedelta(days=int(365.25 * years) + 10)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d&d1={d1:%Y%m%d}&d2={d2:%Y%m%d}"
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200 or not r.text.lstrip().lower().startswith("date"):
        raise DataError(f"Stooq javobi yaroqsiz (HTTP {r.status_code})")
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"], index_col="Date")
    return _clean_ohlcv(df)


def fetch_one(ticker: str, years: float = DATA_YEARS) -> pd.DataFrame:
    """Bitta aksiya tarixini oladi: avval Yahoo, soʻng Stooq."""
    ticker = normalize_ticker(ticker)
    errs: List[str] = []
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(start=_start_date(years), interval="1d", auto_adjust=True)
        d = _clean_ohlcv(df)
        if d is not None:
            return _drop_partial_today(d)
        errs.append("Yahoo: maʼlumot boʻsh")
    except Exception as e:
        errs.append(f"Yahoo: {type(e).__name__}: {str(e)[:100]}")
    try:
        d = _stooq_one(ticker, years)
        if d is not None:
            return _drop_partial_today(d)
        errs.append("Stooq: maʼlumot boʻsh")
    except Exception as e:
        errs.append(f"Stooq: {type(e).__name__}: {str(e)[:100]}")
    raise DataError(f"'{ticker}' uchun narx maʼlumoti olinmadi (tiker toʻgʻriligini va internetni tekshiring). "
                    + " | ".join(errs))


def load_universe_data(years: float = DATA_YEARS, cb: Optional[Callable[[float, str], None]] = None,
                       force: bool = False) -> Dict[str, pd.DataFrame]:
    tickers = list(dict.fromkeys(UNIVERSE + [BENCH]))
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"universe_{int(years)}y_{dt.date.today().isoformat()}.pkl")
    if not force and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if BENCH in obj and len(obj) > 40:
                log(f"Kesh topildi: {len(obj)} ta tiker")
                return obj
        except Exception:
            pass
    data: Dict[str, pd.DataFrame] = {}
    try:
        data = _yf_bulk(tickers, years, cb=cb)
    except Exception as e:
        log("Yahoo ommaviy yuklash xatosi:", type(e).__name__, str(e)[:150])
    missing = [t for t in tickers if t not in data]
    # Yahoo cheklagan boʻlsa (yoki SPY yoʻq boʻlsa) — har birini alohida (Yahoo/Stooq) urinib koʻramiz
    if missing and (len(data) < 0.6 * len(tickers) or BENCH in missing):
        t0 = time.time()
        budget = float(os.environ.get("AKSIYA_FALLBACK_SECONDS", "240"))     # GitHub Actions da kattaroq qiymat beriladi
        for j, t in enumerate(missing):
            if time.time() - t0 > budget:
                break
            try:
                data[t] = fetch_one(t, years)
            except Exception:
                pass
            if cb:
                cb(min(1.0, (j + 1) / len(missing)), f"Zaxira manbadan yuklanmoqda: {j + 1}/{len(missing)}")
    if BENCH not in data:
        raise DataError("Bozor indeksi (SPY) maʼlumoti olinmadi — Yahoo Finance va Stooq bilan aloqa yoʻq. "
                        "Bir necha daqiqadan keyin qayta urinib koʻring.")
    spy_last = data[BENCH].index[-1]
    data = {t: _drop_partial_today(d) if t == BENCH else d for t, d in data.items()}
    keep = {t: d for t, d in data.items()
            if len(d) >= WARMUP + 60 and d.index[-1] >= spy_last - pd.Timedelta(days=10)}
    if len(keep) < 40:
        raise DataError(f"Faqat {len(keep)} ta aksiya uchun yetarli tarix yuklandi (kamida 40 kerak). "
                        "Internet yoki Yahoo cheklovini tekshiring.")
    try:
        with open(path, "wb") as f:
            pickle.dump(keep, f)
        for fn in os.listdir(CACHE_DIR):          # eski keshlarni tozalash
            fp = os.path.join(CACHE_DIR, fn)
            if fn.startswith("universe_") and fp != path and time.time() - os.path.getmtime(fp) > 3 * 86400:
                os.remove(fp)
    except Exception:
        pass
    log(f"Yuklandi: {len(keep)} ta tiker, oxirgi sana {spy_last.date()}")
    return keep


# ======================================================================================
# 2. BELGILAR (faqat oʻtmishdan hisoblanadi — kelajakka qarash yoʻq)
# ======================================================================================
FEATURES_CONT = ["ret1", "ret5", "ret10", "ret20", "ret60", "ret120", "mom_12_1",
                 "vol20", "vol60", "vol_ratio_20_60", "atr_pct", "beta60",
                 "dist_sma10", "dist_sma30", "dist_sma50", "dist_sma200",
                 "pos60", "dist_52w_high", "dist_52w_low", "rsi14",
                 "vol_ratio10", "vol_trend", "gann_slope"]
FEATURES_DISC = ["candle_dir", "wy_accum", "wy_markup", "wy_distrib", "wy_markdown"]
FEATURES = FEATURES_CONT + FEATURES_DISC
EXTRA = ["sig_mc", "rel20", "rel60"]               # modelga kirmaydigan yordamchi ustunlar
MODEL_COLS = ["r_" + f for f in FEATURES_CONT] + FEATURES_DISC

FEATURE_GROUPS: Dict[str, List[str]] = {
    "Momentum (1–12 oy)": ["ret20", "ret60", "ret120", "mom_12_1"],
    "Qisqa muddat (1–10 kun, RSI)": ["ret1", "ret5", "ret10", "rsi14"],
    "Volatillik va risk": ["vol20", "vol60", "vol_ratio_20_60", "atr_pct", "beta60"],
    "Trend (SMA masofalari)": ["dist_sma10", "dist_sma30", "dist_sma50", "dist_sma200"],
    "Diapazon / 52 hafta": ["pos60", "dist_52w_high", "dist_52w_low"],
    "Hajm": ["vol_ratio10", "vol_trend"],
    "Wyckoff fazasi": ["wy_accum", "wy_markup", "wy_distrib", "wy_markdown"],
    "Sham modellari": ["candle_dir"],
    "Gann burchagi": ["gann_slope"],
}

WYCKOFF_UZ = {0: "Aniqlanmadi", 1: "Accumulation (toʻplanish)", 2: "Markup (oʻsish)",
              3: "Distribution (taqsimlanish)", 4: "Markdown (pasayish)"}


def rsi_series(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    au = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = au / ad
        val = np.where(ad == 0, np.where(au == 0, 50.0, 100.0), 100.0 - 100.0 / (1.0 + rs))
    out = pd.Series(val, index=close.index)
    return out.where(au.notna() & ad.notna())


def atr_series(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def wyckoff_code(df: pd.DataFrame) -> pd.Series:
    """Qoidaga asoslangan Wyckoff fazasi (1 Accum., 2 Markup, 3 Distrib., 4 Markdown; NaN = aniqlanmadi).

    Trend: narx>SMA50>SMA200 va SMA50 oʻsmoqda -> Markup; teskarisi -> Markdown.
    Trend yoʻq (diapazon): narx 52 haftalik diapazonning pastki yarmida -> Accumulation, yuqori yarmida -> Distribution.
    Bu — klassik Wyckoff tahlilining soddalashtirilgan, qoidaga asoslangan yaqinlashuvi (trader koʻzi bilan emas).
    """
    c, h, l = df["Close"], df["High"], df["Low"]
    s50 = c.rolling(50).mean()
    s200 = c.rolling(200).mean()
    hh = h.rolling(252, min_periods=200).max()
    ll = l.rolling(252, min_periods=200).min()
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = (c - ll) / (hh - ll).replace(0, np.nan)
    up = (c > s50) & (s50 > s200) & (s50 > s50.shift(10))
    dn = (c < s50) & (s50 < s200) & (s50 < s50.shift(10))
    code = np.where(up, 2.0, np.where(dn, 4.0, np.where(pos < 0.5, 1.0, 3.0)))
    valid = s200.notna() & pos.notna() & s50.shift(10).notna()
    return pd.Series(np.where(valid, code, np.nan), index=c.index)


_CANDLE_DIR = {"Hammer": 1.0, "Bullish Marubozu": 1.0, "Bullish Engulfing": 1.0,
               "Hanging Man": -1.0, "Shooting Star": -1.0, "Bearish Marubozu": -1.0,
               "Bearish Engulfing": -1.0, "Doji": 0.0, "": 0.0}
CANDLE_UZ = {"Hammer": "Bolgʻa (Hammer) — pastdan qaytish ehtimoli",
             "Hanging Man": "Osilgan odam (Hanging Man) — yuqoridan qaytish xavfi",
             "Shooting Star": "Uchar yulduz (Shooting Star) — yuqoridan qaytish xavfi",
             "Bullish Marubozu": "Bullish Marubozu — xaridorlar toʻliq nazoratda",
             "Bearish Marubozu": "Bearish Marubozu — sotuvchilar toʻliq nazoratda",
             "Bullish Engulfing": "Bullish Engulfing — oʻsishga burilish signali",
             "Bearish Engulfing": "Bearish Engulfing — pasayishga burilish signali",
             "Doji": "Doji — ikkilanish, muvozanat"}


def candle_names_series(df: pd.DataFrame) -> pd.Series:
    o, h, l, c = (df[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    rng = h - l
    body = (c - o).abs()
    top = pd.concat([o, c], axis=1).max(axis=1)
    bot = pd.concat([o, c], axis=1).min(axis=1)
    upper = h - top
    lower = bot - l
    with np.errstate(divide="ignore", invalid="ignore"):
        bf, uf, lf = body / rng, upper / rng, lower / rng
    ok = rng > 0
    down_ctx = c.shift(1) < c.shift(6)
    up_ctx = c.shift(1) > c.shift(6)
    doji = ok & (bf <= 0.1)
    hs = ok & (lf >= 0.6) & (uf <= 0.15) & (bf <= 0.35)          # kichik tana tepada, uzun pastki soya
    inv = ok & (uf >= 0.6) & (lf <= 0.15) & (bf <= 0.35)          # kichik tana pastda, uzun yuqori soya
    maru = ok & (bf >= 0.9)
    po, pc = o.shift(1), c.shift(1)
    pbody = (pc - po).abs()
    bull_eng = (pc < po) & (c > o) & (c >= po) & (o <= pc) & (body > pbody)
    bear_eng = (pc > po) & (c < o) & (c <= po) & (o >= pc) & (body > pbody)
    names = pd.Series("", index=df.index, dtype=object)
    names[doji] = "Doji"
    names[hs & down_ctx] = "Hammer"
    names[hs & up_ctx] = "Hanging Man"
    names[inv & up_ctx] = "Shooting Star"
    names[maru & (c > o)] = "Bullish Marubozu"
    names[maru & (c < o)] = "Bearish Marubozu"
    names[bull_eng] = "Bullish Engulfing"
    names[bear_eng] = "Bearish Engulfing"
    return names


def candle_dir_series(df: pd.DataFrame) -> pd.Series:
    return candle_names_series(df).map(_CANDLE_DIR).astype(float)


GANN_UNIT_ATR = 0.25      # Gann 1x1 chizigʻi: har bar uchun 0.25 ATR (ixtiyoriy, lekin barqaror ulchov)
GANN_WIN = 126            # langar (pivot low) izlanadigan oyna (~6 oy)
GANN_RATIOS = [("8x1", 8.0), ("4x1", 4.0), ("3x1", 3.0), ("2x1", 2.0), ("1x1", 1.0),
               ("1x2", 0.5), ("1x3", 1.0 / 3), ("1x4", 0.25), ("1x8", 0.125)]


def gann_anchor(df: pd.DataFrame, win: int = GANN_WIN) -> pd.DataFrame:
    """Har bir sana uchun oxirgi `win` kundagi eng past past (langar) — vektorlashtirilgan, faqat oʻtmish."""
    low = df["Low"].values.astype(float)
    n = len(low)
    pos = np.full(n, -1, dtype=np.int64)
    if n >= win:
        w = sliding_window_view(low, win)
        end = np.arange(win - 1, n)
        pos[end] = end - (win - 1) + w.argmin(axis=1)
    out = pd.DataFrame(index=df.index)
    out["anchor_pos"] = pos
    out["anchor_low"] = np.where(pos >= 0, low[np.clip(pos, 0, n - 1)], np.nan)
    out["bars"] = np.where(pos >= 0, np.arange(n) - pos, np.nan)
    return out


def gann_slope_series(df: pd.DataFrame, atr: Optional[pd.Series] = None, win: int = GANN_WIN) -> pd.Series:
    """Narx langar pastdan beri qanchalik tez oʻsgan: 1.0 = aynan Gann 1x1 chizigʻida; 0 = langarda; >1 = tikroq."""
    if atr is None:
        atr = atr_series(df)
    g = gann_anchor(df, win)
    c = df["Close"].values.astype(float)
    unit = GANN_UNIT_ATR * atr.values
    bars = np.maximum(g["bars"].values, 5.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (c - g["anchor_low"].values) / (bars * unit)
    slope = np.where(np.isfinite(slope), np.clip(slope, 0.0, 4.0), np.nan)
    return pd.Series(slope, index=df.index)


def compute_features(df: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """Barcha belgilarni hisoblaydi. Hamma narsa TRAILING (faqat t sanagacha boʻlgan maʼlumot)."""
    c = df["Close"].astype(float)
    h, l = df["High"].astype(float), df["Low"].astype(float)
    v = df["Volume"].astype(float)
    mk = market.reindex(df.index).ffill()
    f = pd.DataFrame(index=df.index)
    for n in (1, 5, 10, 20, 60, 120):
        f[f"ret{n}"] = c.pct_change(n)
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1.0
    lr = np.log(c).diff()
    f["vol20"] = lr.rolling(20).std()
    f["vol60"] = lr.rolling(60).std()
    f["vol_ratio_20_60"] = f["vol20"] / f["vol60"]
    atr = atr_series(df)
    f["atr_pct"] = atr / c
    for n in (10, 30, 50, 200):
        f[f"dist_sma{n}"] = c / c.rolling(n).mean() - 1.0
    hh60, ll60 = h.rolling(60).max(), l.rolling(60).min()
    with np.errstate(divide="ignore", invalid="ignore"):
        f["pos60"] = (c - ll60) / (hh60 - ll60).replace(0, np.nan)
    f["dist_52w_high"] = c / h.rolling(252, min_periods=200).max() - 1.0
    f["dist_52w_low"] = c / l.rolling(252, min_periods=200).min() - 1.0
    f["rsi14"] = rsi_series(c)
    with np.errstate(divide="ignore", invalid="ignore"):
        f["vol_ratio10"] = v.rolling(10).mean() / v.rolling(60).mean().replace(0, np.nan)
        f["vol_trend"] = np.log(v.rolling(20).mean().replace(0, np.nan) / v.rolling(120).mean().replace(0, np.nan))
    r = c.pct_change()
    mr = mk.pct_change()
    f["beta60"] = r.rolling(60).cov(mr) / mr.rolling(60).var().replace(0, np.nan)
    f["gann_slope"] = gann_slope_series(df, atr)
    wy = wyckoff_code(df)
    for k, name in ((1, "wy_accum"), (2, "wy_markup"), (3, "wy_distrib"), (4, "wy_markdown")):
        f[name] = (wy == k).astype(float).where(wy.notna())
    f["candle_dir"] = candle_dir_series(df)
    # yordamchi (modelga kirmaydi)
    ew = lr.ewm(alpha=0.06, min_periods=40).std()
    lt = lr.rolling(252, min_periods=120).std()
    f["sig_mc"] = np.sqrt(0.5 * ew ** 2 + 0.5 * lt ** 2)
    f["rel20"] = f["ret20"] - mk.pct_change(20)
    f["rel60"] = f["ret60"] - mk.pct_change(60)
    return f.replace([np.inf, -np.inf], np.nan)


# ======================================================================================
# 3. DATASET (kesimli etiketlar + kesim-reytingli belgilar)
# ======================================================================================
@dataclass
class Dataset:
    S: pd.DataFrame                 # namunalar: (di, date, ticker) + belgilar + etiketlar
    dates: pd.DatetimeIndex         # namuna sanalari (di -> sana)
    latest: pd.DataFrame            # oxirgi sana uchun belgilar (index = tiker)
    latest_close: pd.Series
    ref: Dict[str, np.ndarray]      # oxirgi kesimdagi saralangan qiymatlar (reyting uchun)
    last_date: pd.Timestamp
    horizons: List[int]
    min_names: int
    stride: int


def build_dataset(data: Dict[str, pd.DataFrame], horizons: List[int] = HORIZONS, stride: int = STRIDE,
                  warm: int = WARMUP, min_names: int = MIN_NAMES, bench: str = BENCH,
                  cb: Optional[Callable[[float, str], None]] = None) -> Dataset:
    if bench not in data:
        raise DataError("Benchmark (SPY) topilmadi")
    market = data[bench]["Close"].astype(float)
    master = market.index
    n = len(master)
    if n < warm + 60:
        raise DataError("Tarix juda qisqa")
    pos = np.arange(n - 1, warm - 1, -stride)[::-1]       # oxirgi sana har doim namunada
    dates = master[pos]
    mfw = {H: (market.shift(-H) / market - 1.0).reindex(dates).values for H in horizons}
    parts: List[pd.DataFrame] = []
    latest_rows: Dict[str, Dict[str, float]] = {}
    latest_close: Dict[str, float] = {}
    tick = [t for t in data if t != bench]
    for i, t in enumerate(tick):
        df = data[t]
        if len(df) < warm + 30:
            continue
        F = compute_features(df, market)
        c = df["Close"].astype(float)
        X = F.reindex(dates)
        for H in horizons:
            X[f"fwd{H}"] = (c.shift(-H) / c - 1.0).reindex(dates)
        X["ticker"] = t
        X["di"] = np.arange(len(dates))
        X["date"] = dates
        parts.append(X)
        if df.index[-1] >= master[-1] - pd.Timedelta(days=7):
            last = F.iloc[-1]
            if last[FEATURES].notna().all():
                latest_rows[t] = {k: float(last[k]) for k in FEATURES + EXTRA if pd.notna(last[k])}
                latest_close[t] = float(c.iloc[-1])
        if cb and i % 20 == 0:
            cb(i / max(1, len(tick)), f"Belgilar hisoblanmoqda: {i}/{len(tick)}")
    if not parts:
        raise DataError("Belgilar hisoblash uchun yetarli maʼlumot yoʻq")
    S = pd.concat(parts, ignore_index=True)
    del parts, X                                               # oraliq nusxalar xotirani band qilmasin
    S = S.dropna(subset=FEATURES)
    cnt = S.groupby("di")["ticker"].transform("count")
    S = S[cnt >= min_names].copy().reset_index(drop=True)
    g = S.groupby("di")
    mfw_map = {H: pd.Series(mfw[H], index=np.arange(len(dates))) for H in horizons}
    beta_adj = 0.67 * S["beta60"] + 0.33                       # Blume tuzatmasi
    for H in horizons:
        col = f"fwd{H}"
        S[f"mfwd{H}"] = S["di"].map(mfw_map[H])
        # BETA-NEYTRAL kesimli etiket: bozor yoʻnalishidan tozalangan qoldiq daromad, sana medianasiga nisbatan.
        # (Shunda yuqori-beta aksiyalar bull bozorda tasodifan "yaxshi" koʻrinib, sunʼiy dalil yaratmaydi.)
        resid = S[col] - beta_adj * S[f"mfwd{H}"]
        grp = resid.groupby(S["di"])
        nn = grp.transform("count")
        med = grp.transform("median")
        ex = (resid - med).where(nn >= min_names)
        S[f"ex{H}"] = ex
        S[f"y{H}"] = np.where(ex.notna(), (ex > 0).astype(float), np.nan)
    for f_ in FEATURES_CONT:
        S["r_" + f_] = g[f_].rank(pct=True)
    latest = pd.DataFrame.from_dict(latest_rows, orient="index")
    if len(latest) < min_names:
        raise DataError(f"Joriy kesim uchun faqat {len(latest)} ta aksiya belgisi hisoblandi")
    latest = latest.dropna(subset=FEATURES)
    ref = {f_: np.sort(latest[f_].dropna().values) for f_ in FEATURES_CONT}
    return Dataset(S=S, dates=dates, latest=latest, latest_close=pd.Series(latest_close),
                   ref=ref, last_date=master[-1], horizons=list(horizons), min_names=min_names, stride=stride)


def model_matrix_from_features(feat: pd.DataFrame, ref: Dict[str, np.ndarray]) -> np.ndarray:
    """Xom belgilar -> model kirishi (kesim-reytingi universum bilan solishtirib)."""
    cols = []
    for f_ in FEATURES_CONT:
        a = ref[f_]
        cols.append(np.searchsorted(a, feat[f_].values.astype(float), side="right") / float(len(a)))
    for f_ in FEATURES_DISC:
        cols.append(feat[f_].values.astype(float))
    return np.column_stack(cols)


# ======================================================================================
# 4. MODEL VA HALOL SINOV (purged walk-forward, IC, backtest)
# ======================================================================================
class EnsembleModel:
    """Ikki model oʻrtachasi: (1) juda qatʼiy regullarizatsiyalangan sayoz gradient boosting (nochiziqli oʻzaro taʼsirlar),
    (2) kuchli regullarizatsiyali logistik regressiya (barqaror, kam namunada ham ishlaydi).
    Moliyaviy maʼlumotda signal/shovqin nisbati juda past — shuning uchun oddiy va qatʼiy modellar odatda yaxshiroq."""

    def __init__(self, hgb_params: Optional[dict] = None, C: float = LOGIT_C):
        self.hgb_params = hgb_params or HGB_PARAMS
        self.C = C
        self.hgb = None
        self.lr = None

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        self.hgb = HistGradientBoostingClassifier(**self.hgb_params).fit(X, y)
        self.lr = LogisticRegression(C=self.C, max_iter=500).fit(X, y)
        return self

    def predict_proba(self, X):
        p = 0.5 * self.hgb.predict_proba(X)[:, 1] + 0.5 * self.lr.predict_proba(X)[:, 1]
        return np.column_stack([1.0 - p, p])


def _make_model(params: Optional[dict] = None):
    return EnsembleModel(hgb_params=params)


def _auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(y) < 20 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def per_date_ic(di: np.ndarray, p: np.ndarray, ex: np.ndarray, min_names: int = 8) -> pd.Series:
    """Har bir sana uchun Spearman IC: model balli bilan keyingi (kesimli) daromad orasidagi reyting korrelyatsiyasi."""
    d = pd.DataFrame({"di": di, "p": p, "ex": ex})
    d["rp"] = d.groupby("di")["p"].rank()
    d["re"] = d.groupby("di")["ex"].rank()
    d["pe"] = d["rp"] * d["re"]
    d["pp"] = d["rp"] ** 2
    d["ee"] = d["re"] ** 2
    g = d.groupby("di")
    n = g["rp"].count().astype(float)
    mp, me = g["rp"].mean(), g["re"].mean()
    cov = g["pe"].mean() - mp * me
    vp = g["pp"].mean() - mp ** 2
    ve = g["ee"].mean() - me ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = cov / np.sqrt(vp * ve)
    return ic.where((n >= min_names) & (vp > 1e-12) & (ve > 1e-12))


def nonoverlap_t(s: pd.Series, k: int) -> Tuple[float, float, float]:
    """Ustma-ust tushmaydigan kuzatuvlar (har k-sana) boʻyicha t-statistika; barcha siljishlar oʻrtachasi.
    Qaytaradi: (barcha sanalar oʻrtachasi, t, mustaqil kuzatuvlar soni)."""
    s = s.dropna().sort_index()
    k = max(1, int(k))
    if len(s) == 0:
        return float("nan"), float("nan"), 0.0
    ts = []
    for o in range(k):
        x = s.values[o::k]
        if len(x) >= 5 and np.std(x, ddof=1) > 0:
            ts.append(float(np.mean(x) / (np.std(x, ddof=1) / math.sqrt(len(x)))))
    t = float(np.mean(ts)) if ts else float("nan")
    return float(s.mean()), t, len(s) / k


def quintile_stats(di, p, ex, y, k: int) -> Dict[str, float]:
    d = pd.DataFrame({"di": di, "p": p, "ex": ex, "y": y})
    d["q"] = d.groupby("di")["p"].rank(pct=True)
    top, bot = d[d["q"] > 0.8], d[d["q"] <= 0.2]
    ts, bs = top.groupby("di")["ex"].mean(), bot.groupby("di")["ex"].mean()
    sp = (ts - bs).dropna()
    m, t, neff = nonoverlap_t(sp, k)
    return dict(hit=float(top["y"].mean()) if len(top) else float("nan"), spread=m, spread_t=t,
                top_ex=float(ts.mean()) if len(ts) else float("nan"),
                bot_ex=float(bs.mean()) if len(bs) else float("nan"))


def _group_col_idx() -> Dict[str, List[int]]:
    idx = {}
    for g, feats in FEATURE_GROUPS.items():
        cols = [("r_" + f if f in FEATURES_CONT else f) for f in feats]
        idx[g] = [MODEL_COLS.index(c) for c in cols]
    return idx


def group_permutation_importance(model, X: np.ndarray, y: np.ndarray, di: np.ndarray,
                                 rng: np.random.Generator, reps: int = 2) -> Dict[str, float]:
    """Belgilar guruhini HAR SANA ichida aralashtirib, AUC qanchaga tushishini oʻlchaydi (musbat = guruh foydali)."""
    base = _auc(y, model.predict_proba(X)[:, 1])
    date_idx = [np.where(di == u)[0] for u in np.unique(di)]
    out: Dict[str, float] = {}
    for gname, cidx in _group_col_idx().items():
        drops = []
        for _ in range(reps):
            Xp = X.copy()
            for idx in date_idx:
                perm = rng.permutation(idx)
                Xp[np.ix_(idx, cidx)] = X[np.ix_(perm, cidx)]
            drops.append(base - _auc(y, model.predict_proba(Xp)[:, 1]))
        out[gname] = float(np.nanmean(drops))
    return out


@dataclass
class EvalResult:
    H: int
    purged: bool
    n_rows: int
    n_dates: int
    auc: float
    ic_mean: float
    ic_t: float
    n_eff: float
    hit_top: float
    spread: float
    spread_t: float
    top_ex: float
    folds: List[Dict[str, Any]]
    pos_folds: int
    verdict: str
    verdict_text: str
    importance: Dict[str, float]
    oos: pd.Series


def verdict_from(auc: float, ic_mean: float, ic_t: float, n_eff: float, pos_folds: int, n_folds: int) -> Tuple[str, str]:
    if not np.isfinite(ic_t) or n_eff < 8:
        return "none", "Statistik xulosa uchun mustaqil kuzatuvlar yetarli emas."
    if ic_t >= T_STRONG and ic_mean > 0 and auc > 0.5 and pos_folds >= max(1, n_folds - 1):
        return "strong", "Kuchli dalil: ustunlik koʻp marta tekshiruvdan keyin ham statistik ahamiyatli va barqaror."
    if ic_t >= T_WEAK and ic_mean > 0 and auc > 0.5:
        return "weak", "Zaif dalil: ustunlik bor koʻrinadi, lekin 7 ufq sinalgani uchun tasodif boʻlishi ham mumkin."
    if ic_t <= -T_WEAK:
        return "inverse", "Model teskari ishlamoqda (ball yuqori boʻlgan aksiyalar ortda qolgan)."
    return "none", "Dalil yoʻq: model tarixiy sinovda tasodifdan ishonchli darajada yaxshi emas."


def walk_forward_eval(ds: Dataset, H: int, n_folds: int = N_FOLDS, purge: bool = True,
                      importance: bool = True, params: Optional[dict] = None, seed: int = 0) -> EvalResult:
    """Kengayuvchi oynali walk-forward: har bir test davri uchun model FAQAT undan oldingi maʼlumotda oʻqitiladi.

    purge=True: test davri boshidan `ceil(H/stride)+1` namuna orqaga qaytib, oʻsha oraliqdagi
    oʻqitish namunalari olib tashlanadi — chunki ularning etiketlari (keyingi H kun daromadi) test davriga tutashadi.
    """
    S = ds.S
    ycol, ecol = f"y{H}", f"ex{H}"
    V = S.loc[S[ycol].notna(), ["di", ycol, ecol] + MODEL_COLS]
    if len(V) < 1500:
        raise DataError(f"{HORIZON_UZ.get(H, H)} ufqi uchun yetarli namuna yoʻq ({len(V)} ta)")
    dates_u = np.sort(V["di"].unique())
    nd = len(dates_u)
    n0 = int(nd * 0.4)
    edges = np.linspace(n0, nd, n_folds + 1).round().astype(int)
    emb = int(math.ceil(H / ds.stride)) if purge else 0
    oos = pd.Series(np.nan, index=S.index)
    Xall, yall = V[MODEL_COLS].values, V[ycol].values
    dall, eall = V["di"].values, V[ecol].values
    rng = np.random.default_rng(seed)
    folds: List[Dict[str, Any]] = []
    imps: List[Dict[str, float]] = []
    for k in range(n_folds):
        te_dates = dates_u[edges[k]:edges[k + 1]]
        if len(te_dates) == 0:
            continue
        t0 = te_dates[0]
        te = (dall >= te_dates[0]) & (dall <= te_dates[-1])
        tr = (dall <= t0 - emb - 1) if purge else (dall < t0)
        if tr.sum() < 800 or te.sum() < 200:
            continue
        m = _make_model(params).fit(Xall[tr], yall[tr])
        p = m.predict_proba(Xall[te])[:, 1]
        oos.loc[V.index[te]] = p
        ic = per_date_ic(dall[te], p, eall[te])
        folds.append(dict(fold=k + 1, n_train=int(tr.sum()), n_test=int(te.sum()),
                          start=ds.dates[te_dates[0]], end=ds.dates[te_dates[-1]],
                          auc=_auc(yall[te], p), ic=float(ic.mean())))
        if importance:
            imps.append(group_permutation_importance(m, Xall[te], yall[te], dall[te], rng))
    msk = oos.notna()
    if msk.sum() < 300:
        raise DataError(f"{HORIZON_UZ.get(H, H)}: test namunalari yetarli emas")
    di_o = S.loc[msk, "di"].values
    p_o = oos[msk].values
    ex_o = S.loc[msk, ecol].values
    y_o = S.loc[msk, ycol].values
    k_ov = max(1, int(math.ceil(H / ds.stride)))
    ic_all = per_date_ic(di_o, p_o, ex_o)
    ic_mean, ic_t, n_eff = nonoverlap_t(ic_all, k_ov)
    q = quintile_stats(di_o, p_o, ex_o, y_o, k_ov)
    auc = _auc(y_o, p_o)
    pos_folds = int(sum(1 for f in folds if f["ic"] > 0))
    verdict, vtext = verdict_from(auc, ic_mean, ic_t, n_eff, pos_folds, len(folds))
    imp = {g: float(np.mean([d[g] for d in imps])) for g in FEATURE_GROUPS} if imps else {}
    return EvalResult(H=H, purged=purge, n_rows=int(msk.sum()), n_dates=int(len(ic_all.dropna())),
                      auc=auc, ic_mean=ic_mean, ic_t=ic_t, n_eff=n_eff, hit_top=q["hit"], spread=q["spread"],
                      spread_t=q["spread_t"], top_ex=q["top_ex"], folds=folds, pos_folds=pos_folds,
                      verdict=verdict, verdict_text=vtext, importance=imp, oos=oos)


def naive_split_auc(ds: Dataset, H: int, seed: int = 0, params: Optional[dict] = None) -> float:
    """Taqqoslash uchun: odatiy TASODIFIY train/test boʻlinishi (leakage bor) — AUC sunʼiy yuqori chiqadi."""
    from sklearn.model_selection import train_test_split
    S = ds.S
    ycol = f"y{H}"
    V = S.loc[S[ycol].notna(), MODEL_COLS + [ycol]]
    Xtr, Xte, ytr, yte = train_test_split(V[MODEL_COLS].values, V[ycol].values, test_size=0.25,
                                          shuffle=True, random_state=seed)
    m = _make_model(params).fit(Xtr, ytr)
    return _auc(yte, m.predict_proba(Xte)[:, 1])


# --------------------------------------------------------------------------------------
# Backtest: AI top-N  vs  momentum top-N  vs  teng ogʻirlikdagi universum  vs  SPY
# --------------------------------------------------------------------------------------
@dataclass
class BtResult:
    H: int
    top_n: int
    cost_bps: float
    periods: pd.DataFrame
    curves: Dict[str, pd.Series]
    stats: Dict[str, Dict[str, float]]
    n_periods: int
    t_vs_ew: float
    t_vs_mom: float
    avg_turnover: float
    alpha: float = float("nan")        # AI qaytimining universumga nisbatan regressiya alfasi (davr boshiga)
    beta: float = float("nan")
    t_alpha: float = float("nan")
    alpha_mom: float = float("nan")
    t_alpha_mom: float = float("nan")


def _alpha_beta(y: pd.Series, x: pd.Series) -> Tuple[float, float, float]:
    """y = alfa + beta*x + xato. Qaytaradi: alfa, beta, alfa t-statistikasi."""
    d = pd.concat([y, x], axis=1).dropna()
    if len(d) < 8:
        return float("nan"), float("nan"), float("nan")
    X = np.column_stack([np.ones(len(d)), d.iloc[:, 1].values])
    yy = d.iloc[:, 0].values
    coef = np.linalg.lstsq(X, yy, rcond=None)[0]
    resid = yy - X @ coef
    s2 = float(resid @ resid) / (len(d) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    return float(coef[0]), float(coef[1]), float(coef[0] / math.sqrt(cov[0, 0])) if cov[0, 0] > 0 else float("nan")


def _perf(r: pd.Series, ppy: float) -> Dict[str, float]:
    r = r.dropna()
    n = len(r)
    if n == 0:
        return {}
    eq = (1 + r).cumprod()
    tot = float(eq.iloc[-1] - 1)
    yrs = n / ppy
    cagr = float((1 + tot) ** (1 / yrs) - 1) if yrs > 0 and 1 + tot > 0 else float("nan")
    sd = float(r.std(ddof=1)) if n > 1 else float("nan")
    sharpe = float(r.mean() / sd * math.sqrt(ppy)) if sd and sd > 0 else float("nan")
    mdd = float((eq / eq.cummax() - 1).min())
    return dict(total=tot, cagr=cagr, sharpe=sharpe, mdd=mdd, avg=float(r.mean()), n=n, win=float((r > 0).mean()))


def _t_mean(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 5 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def backtest_topn(ds: Dataset, ev: EvalResult, H: int = PRIMARY_H, top_n: int = TOP_N,
                  cost_bps: float = COST_BPS) -> Optional[BtResult]:
    S = ds.S
    fwd, mf = f"fwd{H}", f"mfwd{H}"
    m = ev.oos.notna() & S[fwd].notna()
    if m.sum() < 200:
        return None
    D = S.loc[m, ["di", "date", "ticker", fwd, "mom_12_1", mf]].copy()
    D["score"] = ev.oos[m].values
    dis = np.sort(D["di"].unique())
    if len(dis) < 8:
        return None
    k = max(1, int(math.ceil(H / ds.stride)))
    prev: Dict[str, set] = {"ai": set(), "mom": set()}
    rows = []
    for di in dis[::k]:
        g = D[D["di"] == di]
        if len(g) < max(2 * top_n, 10):
            continue
        picks = {"ai": g.nlargest(top_n, "score"), "mom": g.nlargest(top_n, "mom_12_1")}
        row = {"date": g["date"].iloc[0], "ew": float(g[fwd].mean()), "spy": float(g[mf].iloc[0])}
        for key, sel in picks.items():
            cur = set(sel["ticker"])
            turn = 1.0 - len(cur & prev[key]) / top_n
            cost = turn * (cost_bps / 1e4) * (2.0 if prev[key] else 1.0)
            row[key] = float(sel[fwd].mean()) - cost
            row[key + "_turn"] = turn
            prev[key] = cur
        rows.append(row)
    if len(rows) < 6:
        return None
    P = pd.DataFrame(rows).set_index("date")
    ppy = 252.0 / H
    names = {"ai": "AI (top-%d)" % top_n, "mom": "Momentum (top-%d)" % top_n,
             "ew": "Universum (teng ulush)", "spy": "SPY"}
    curves: Dict[str, pd.Series] = {}
    stats: Dict[str, Dict[str, float]] = {}
    start = P.index[0] - pd.Timedelta(days=1)
    for key, nm in names.items():
        eq = (1 + P[key]).cumprod()
        curves[nm] = pd.concat([pd.Series([1.0], index=[start]), eq])
        stats[nm] = _perf(P[key], ppy)
        stats[nm]["hit_ew"] = float((P[key] > P["ew"]).mean()) if key != "ew" else float("nan")
    a, b, ta = _alpha_beta(P["ai"], P["ew"])
    am, _, tam = _alpha_beta(P["mom"], P["ew"])
    return BtResult(H=H, top_n=top_n, cost_bps=cost_bps, periods=P, curves=curves, stats=stats,
                    n_periods=len(P), t_vs_ew=_t_mean(P["ai"] - P["ew"]), t_vs_mom=_t_mean(P["ai"] - P["mom"]),
                    avg_turnover=float(P["ai_turn"].mean()), alpha=a, beta=b, t_alpha=ta,
                    alpha_mom=am, t_alpha_mom=tam)


# --------------------------------------------------------------------------------------
# Yakuniy modellar va joriy ballar
# --------------------------------------------------------------------------------------
def fit_final_models(ds: Dataset, horizons: List[int], params: Optional[dict] = None) -> Dict[int, Any]:
    S = ds.S
    out = {}
    for H in horizons:
        V = S.loc[S[f"y{H}"].notna(), MODEL_COLS + [f"y{H}"]]      # faqat kerakli ustunlar (xotira tejash)
        if len(V) < 1000:
            continue
        out[H] = _make_model(params).fit(V[MODEL_COLS].values, V[f"y{H}"].values)
        del V
    return out


def score_latest(ds: Dataset, models: Dict[int, Any]) -> pd.DataFrame:
    X = model_matrix_from_features(ds.latest, ds.ref)
    sc = pd.DataFrame(index=ds.latest.index)
    for H, m in models.items():
        sc[H] = m.predict_proba(X)[:, 1]
    return sc


# --------------------------------------------------------------------------------------
# Monte-Karlo yadrosi (bitta standart tasodifiy matritsa; koʻp joyda qayta ishlatiladi)
# --------------------------------------------------------------------------------------
_STD_CACHE: Dict[tuple, np.ndarray] = {}


def std_cumsums(n: int = MC_PATHS, hmax: Optional[int] = None, seed: int = 12345, nu: int = MC_NU) -> np.ndarray:
    hmax = hmax or max(HORIZONS)
    key = (n, hmax, seed, nu)
    if key not in _STD_CACHE:
        rng = np.random.default_rng(seed)
        z = rng.standard_t(nu, size=(n, hmax)) / math.sqrt(nu / (nu - 2.0))     # birlik dispersiyali t-zarbalar
        _STD_CACHE[key] = np.cumsum(z, axis=1).astype(np.float32)
    return _STD_CACHE[key]


def std_quantiles(horizons, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> Dict[int, np.ndarray]:
    cs = std_cumsums(hmax=max(max(horizons), max(HORIZONS)))
    return {h: np.quantile(cs[:, h - 1] / math.sqrt(h), qs) for h in horizons}


def range_calibration(ds: Dataset, horizons=(5, 20, 60, 125)) -> Dict[int, Dict[str, float]]:
    """Narx diapazoni prognozi (Monte-Karlo) tarixda qanchalik toʻgʻri chiqqanini tekshiradi."""
    S = ds.S
    res: Dict[int, Dict[str, float]] = {}
    hs = [h for h in horizons if f"fwd{h}" in S.columns]
    if not hs:
        return res
    sq = std_quantiles(hs)
    for H in hs:
        d = S[["sig_mc", f"fwd{H}"]].dropna()
        d = d[d["sig_mc"] > 0]
        if len(d) < 300:
            continue
        sig = d["sig_mc"].values
        lr = np.log1p(d[f"fwd{H}"].values)
        z = (lr - (MC_DRIFT - 0.5 * sig ** 2) * H) / (sig * math.sqrt(H))
        q05, q25, q50, q75, q95 = sq[H]
        res[H] = dict(n=int(len(z)), cov90=float(((z >= q05) & (z <= q95)).mean()),
                      cov50=float(((z >= q25) & (z <= q75)).mean()), below_med=float((z < q50).mean()),
                      below5=float((z < q05).mean()), above95=float((z > q95).mean()))
    return res


# ======================================================================================
# 5. TIZIM HOLATI (build, kesh, fon yangilanishi)
# ======================================================================================
@dataclass
class State:
    asof: pd.Timestamp
    built_at: dt.datetime
    data: Dict[str, pd.DataFrame]
    ds: Dataset
    evals: Dict[int, EvalResult]
    bt: Optional[BtResult]
    models: Dict[int, Any]
    scores: pd.DataFrame
    naive_auc: float
    calib: Dict[int, Dict[str, float]]
    primary_h: int
    build_seconds: float
    coefs: Optional[pd.Series] = None                         # asosiy ufq uchun logistik koeffitsientlar
    cache: Dict[str, Any] = field(default_factory=dict)       # UI uchun tayyor HTML boʻlaklari


_STATE: Optional[State] = None
_ERR: Optional[str] = None
_LOCK = threading.Lock()
_BG_LOCK = threading.Lock()
_BG_STARTED = False
_PROG: Dict[str, Any] = {"frac": 0.0, "msg": "Boshlanmagan", "building": False}


def build_state(cb: Optional[Callable[[float, str], None]] = None, data: Optional[Dict[str, pd.DataFrame]] = None,
                horizons: Optional[List[int]] = None, n_folds: int = N_FOLDS) -> State:
    t0 = time.time()

    def step(frac: float, msg: str):
        _PROG["frac"], _PROG["msg"] = frac, msg
        if cb:
            try:
                cb(frac, msg)
            except Exception:
                pass
        log(f"[{frac * 100:3.0f}%] {msg}")

    step(0.02, "Narx maʼlumotlari yuklanmoqda...")
    if data is None:
        data = load_universe_data(cb=lambda f, m: step(0.02 + 0.28 * f, m))
    step(0.32, "Belgilar va etiketlar hisoblanmoqda...")
    ds = build_dataset(data, horizons=horizons or HORIZONS, cb=lambda f, m: step(0.32 + 0.08 * f, m))
    trim_memory()
    hz = ds.horizons
    evals: Dict[int, EvalResult] = {}
    for i, H in enumerate(hz):
        step(0.42 + 0.38 * i / len(hz), f"Model halol sinovdan oʻtkazilmoqda (purged walk-forward): {HORIZON_UZ.get(H, str(H) + ' kun')}")
        evals[H] = walk_forward_eval(ds, H, n_folds=n_folds)
        trim_memory()
    ph = PRIMARY_H if PRIMARY_H in evals else hz[0]
    step(0.82, "Backtest hisoblanmoqda...")
    bt = backtest_topn(ds, evals[ph], H=ph)
    step(0.86, "Yakuniy modellar oʻqitilmoqda...")
    models = fit_final_models(ds, hz)
    scores = score_latest(ds, models)
    trim_memory()
    step(0.94, "Qoʻshimcha tekshiruvlar (leakage va diapazon kalibrovkasi)...")
    naive = naive_split_auc(ds, ph)
    calib = range_calibration(ds)
    coefs = None
    try:
        coefs = pd.Series(models[ph].lr.coef_[0], index=MODEL_COLS)
    except Exception:
        pass
    st = State(asof=ds.last_date, built_at=dt.datetime.now(), data=data, ds=ds, evals=evals, bt=bt,
               models=models, scores=scores, naive_auc=naive, calib=calib, primary_h=ph,
               build_seconds=time.time() - t0, coefs=coefs)
    ds.S = ds.S.iloc[0:0].copy()          # modellar oʻqitib boʻlindi: katta namuna jadvali endi kerak emas (xotira tejaladi)
    trim_memory()
    step(1.0, "Tayyor")
    return st


def get_state() -> Optional[State]:
    return _STATE


def ensure_ready(cb: Optional[Callable[[float, str], None]] = None, force: bool = False,
                 max_age_h: float = REFRESH_HOURS) -> State:
    global _STATE, _ERR
    with _LOCK:
        st = _STATE
        fresh = st is not None and (dt.datetime.now() - st.built_at) < dt.timedelta(hours=max_age_h)
        if st is not None and fresh and not force:
            return st
        _PROG["building"] = True
        try:
            new = build_state(cb=cb)
            for fn in _ON_BUILT:
                try:
                    fn(new)
                except Exception as e:
                    log("on_built xatosi:", e)
            _STATE, _ERR = new, None
            trim_memory()
            return new
        except Exception as e:
            _ERR = str(e) if isinstance(e, DataError) else f"{type(e).__name__}: {e}"
            log("build_state xatosi:", _ERR)
            if st is not None:
                return st
            raise
        finally:
            _PROG["building"] = False


_ON_BUILT: List[Callable[[State], None]] = []       # UI qatlami tayyor HTML ni shu yerda keshlaydi


def start_background() -> None:
    """Tizimni fon oqimida tayyorlaydi va har REFRESH_HOURS soatda yangilab turadi."""
    global _BG_STARTED
    with _BG_LOCK:
        if _BG_STARTED:
            return
        _BG_STARTED = True

    def loop():
        while True:
            try:
                ensure_ready()
                wait = 1800
            except Exception:
                wait = 300
            time.sleep(wait)

    threading.Thread(target=loop, daemon=True, name="aksiya-bg").start()


def retry_now() -> None:
    """Xatodan keyin darhol qayta urinish (UI tugmasi)."""
    if _PROG.get("building"):
        return
    threading.Thread(target=lambda: _safe(ensure_ready, force=True), daemon=True).start()


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        return None


def wait_ready(progress=None, timeout: float = 1500.0) -> State:
    start_background()
    t0 = time.time()
    while True:
        st = _STATE
        if st is not None:
            return st
        if _ERR and not _PROG.get("building"):
            raise DataError(_ERR)
        if progress is not None:
            try:
                progress(float(_PROG["frac"]), desc=str(_PROG["msg"]))
            except Exception:
                pass
        if time.time() - t0 > timeout:
            raise DataError("Tizimni tayyorlash vaqti oshib ketdi. Sahifani yangilab qayta urinib koʻring.")
        time.sleep(0.8)


# ======================================================================================
# 6. BITTA AKSIYANI CHUQUR TAHLIL QILISH
# ======================================================================================
MC_TOUCH_LEVELS = (0.05, 0.10, 0.20, 0.30, 0.50)
MC_DROP_LEVELS = (0.05, 0.10, 0.20, 0.30)
MC_QS = (0.05, 0.25, 0.50, 0.75, 0.95)


def _mc_paths(df: pd.DataFrame, hmax: int) -> Tuple[float, float, np.ndarray]:
    """Monte-Karlo yoʻllari: (joriy narx, kunlik sigma, log(narx/P0) yoʻllari). mc_forecast va mc_barrier_grid umumiy manbasi."""
    c = df["Close"].astype(float)
    P0 = float(c.iloc[-1])
    lr = np.log(c).diff().dropna()
    ew = float(lr.ewm(alpha=0.06, min_periods=40).std().iloc[-1])
    lt = float(lr.tail(252).std())
    sig = math.sqrt(0.5 * ew ** 2 + 0.5 * lt ** 2)
    cs = std_cumsums(hmax=hmax)
    ks = np.arange(1, cs.shape[1] + 1, dtype=np.float32)
    L = (MC_DRIFT - 0.5 * sig ** 2) * ks[None, :] + sig * cs
    return P0, sig, L


def mc_barrier_grid(df: pd.DataFrame, horizons: List[int] = HORIZONS, step: float = 0.01,
                    up_max: float = 1.61, dn_max: float = 1.39) -> Dict[str, Any]:
    """Statik sayt uchun: maqsad narxga yetish ehtimolini BRAUZERDA (JavaScript) hisoblash uchun ehtimollar jadvali.
    x = ln(maqsad / joriy narx).  x >= 0: tu = P(davr ichida x ga yetadi), eu = P(ufq oxirida >= x);
    x < 0: td = P(davr ichida x ga tushadi), ed = P(ufq oxirida <= x).  Qiymatlar 10000 ga koʻpaytirilgan butun sonlar;
    indeks i  ->  |x| = i * step.  mc_forecast(target=...) bilan bir xil yoʻllardan hisoblanadi."""
    P0, sig, L = _mc_paths(df, max(horizons))
    n = L.shape[0]
    xu = np.arange(int(round(up_max / step)) + 1) * step
    xd = -np.arange(int(round(dn_max / step)) + 1) * step
    out: Dict[str, Any] = dict(P0=P0, step=step, hs=[int(h) for h in horizons], tu=[], eu=[], td=[], ed=[])
    q = lambda p: np.round(np.asarray(p) * 10000).astype(int).tolist()
    for h in horizons:
        mx = np.sort(L[:, :h].max(axis=1)).astype(np.float64)
        mn = np.sort(L[:, :h].min(axis=1)).astype(np.float64)
        en = np.sort(L[:, h - 1]).astype(np.float64)
        out["tu"].append(q(1.0 - np.searchsorted(mx, xu, side="left") / n))
        out["eu"].append(q(1.0 - np.searchsorted(en, xu, side="left") / n))
        out["td"].append(q(np.searchsorted(mn, xd, side="right") / n))
        out["ed"].append(q(np.searchsorted(en, xd, side="right") / n))
    return out


def mc_forecast(df: pd.DataFrame, horizons: List[int] = HORIZONS, target: Optional[float] = None,
                levels: Optional[List[Tuple[str, float]]] = None) -> Dict[str, Any]:
    """Monte-Karlo: narx diapazoni va maqsadga yetish ehtimoli.
    Drift = bozor-prior (aksiyaning oʻz oʻtmish daromadi EMAS), volatillik = EWMA va 1 yillik oʻrtachasi aralashmasi,
    zarbalar = t-taqsimot (ogʻir dumlar). Bu AI signali emas, balki statistik "volatillik konusi"."""
    P0, sig, L = _mc_paths(df, max(horizons))                                # L: log(narx/P0) yoʻllari
    rows = []
    for h in horizons:
        Lh, mx, mn = L[:, h - 1], L[:, :h].max(axis=1), L[:, :h].min(axis=1)
        q = np.quantile(Lh, MC_QS)
        rows.append(dict(
            h=h, p_up=float((Lh > 0).mean()),
            q=[float(P0 * math.exp(x)) for x in q],
            touch={x: float((mx >= math.log1p(x)).mean()) for x in MC_TOUCH_LEVELS},
            drop={x: float((mn <= math.log1p(-x)).mean()) for x in MC_DROP_LEVELS},
            end_above={x: float((Lh >= math.log1p(x)).mean()) for x in MC_TOUCH_LEVELS}))
    out: Dict[str, Any] = dict(P0=P0, sigma_d=sig, sigma_ann=sig * math.sqrt(252), rows=rows)
    bands = np.quantile(L, MC_QS, axis=0)                                    # (5, hmax)
    out["bands"] = P0 * np.exp(bands)
    if target and target > 0:
        b = math.log(target / P0)
        t_touch, t_end = {}, {}
        for h in horizons:
            if target >= P0:
                t_touch[h] = float((L[:, :h].max(axis=1) >= b).mean())
                t_end[h] = float((L[:, h - 1] >= b).mean())
            else:
                t_touch[h] = float((L[:, :h].min(axis=1) <= b).mean())
                t_end[h] = float((L[:, h - 1] <= b).mean())
        out["target"] = dict(price=float(target), pct=float(target / P0 - 1), touch=t_touch, end=t_end,
                             up=bool(target >= P0))
    lv = []
    for name, price in (levels or []):
        b = math.log(price / P0)
        d = {}
        for h in (20, 60, 125):
            if h not in horizons:
                continue
            d[h] = float((L[:, :h].max(axis=1) >= b).mean()) if price >= P0 else float((L[:, :h].min(axis=1) <= b).mean())
        lv.append(dict(name=name, price=float(price), touch=d))
    out["levels"] = lv
    return out


def find_levels(df: pd.DataFrame, lookback: int = 250, win: int = 5) -> Dict[str, Any]:
    """Support/resistance: pivot (mahalliy ekstremum) nuqtalarini narx boʻyicha klasterlaydi; teginishlar soni va yangilikka koʻra baholaydi."""
    d = df.tail(lookback)
    h, l, c = d["High"].values, d["Low"].values, d["Close"].values
    n = len(d)
    price = float(c[-1])
    atr = float(atr_series(df).iloc[-1])
    tol = max(0.6 * atr, 0.006 * price)
    piv = []
    for i in range(win, n - win):
        if h[i] >= h[i - win:i + win + 1].max():
            piv.append((float(h[i]), i))
        if l[i] <= l[i - win:i + win + 1].min():
            piv.append((float(l[i]), i))
    piv.sort(key=lambda x: x[0])
    clusters: List[Dict[str, Any]] = []
    for p, i in piv:
        if clusters and p - clusters[-1]["first"] <= tol:
            clusters[-1]["pts"].append((p, i))
        else:
            clusters.append(dict(first=p, pts=[(p, i)]))
    lv = []
    for cl in clusters:
        ps = np.array([x[0] for x in cl["pts"]])
        idx = np.array([x[1] for x in cl["pts"]])
        lv.append(dict(price=float(ps.mean()), touches=len(ps), last=d.index[int(idx.max())],
                       score=float(len(ps) + (idx / n).sum())))
    sup = sorted([x for x in lv if x["price"] < price - 0.25 * tol], key=lambda x: price - x["price"])
    res = sorted([x for x in lv if x["price"] > price + 0.25 * tol], key=lambda x: x["price"] - price)

    def pick(arr):
        strong = [x for x in arr if x["touches"] >= 2]
        weak = [x for x in arr if x["touches"] < 2]
        sel = (strong + weak)[:3] if len(strong) < 2 else strong[:3]
        return sorted(sel, key=lambda x: abs(x["price"] - price))
    sup_sel, res_sel = pick(sup), pick(res)
    for x in sup_sel + res_sel:
        x["dist"] = x["price"] / price - 1.0
    return dict(support=sup_sel, resistance=res_sel, tol=tol, atr=atr,
                hi52=float(df["High"].tail(252).max()), lo52=float(df["Low"].tail(252).min()))


def wyckoff_events(df: pd.DataFrame, rng_bars: int = 60, recent: int = 10) -> Dict[str, Any]:
    """Wyckoff Spring / Test / Upthrust: oxirgi `recent` kunda diapazon (oldingi `rng_bars` kun) chegarasidan
    "yolgʻon" chiqish va qaytish (undercut/overshoot + recovery) qidiriladi."""
    n = len(df)
    out: Dict[str, Any] = dict(range_ok=False, spring=None, test=None, upthrust=None, support=None, resistance=None)
    if n < rng_bars + recent + 5:
        return out
    rng = df.iloc[-(rng_bars + recent):-recent]
    sup, res = float(rng["Low"].min()), float(rng["High"].max())
    out.update(support=sup, resistance=res)
    price = float(df["Close"].iloc[-1])
    height = max(res - sup, 1e-9)
    travel = abs(float(rng["Close"].iloc[-1]) - float(rng["Close"].iloc[0]))
    # haqiqiy savdo diapazoni: juda keng emas va narx diapazon orqali "yurib oʻtmagan" (trend emas)
    out["range_ok"] = bool(height / price <= 0.40 and travel <= 0.6 * height)
    if not out["range_ok"]:
        return out
    vavg = float(rng["Volume"].mean()) or 1.0
    rec = df.iloc[-recent:]
    sp_c: List[Tuple[float, int]] = []
    ut_c: List[Tuple[float, int]] = []
    for i in range(len(rec)):                                # nomzodlar: chegaradan "yolgʻon" chiqib, shu kunning oʻzida qaytganlar
        row = rec.iloc[i]
        if row["Low"] < sup * 0.999 and row["Close"] > sup:
            sp_c.append((float(sup / row["Low"] - 1), -i))
        if row["High"] > res * 1.001 and row["Close"] < res:
            ut_c.append((float(row["High"] / res - 1), -i))
    if sp_c:                                                 # Spring = eng chuqur "silkitish" (teng boʻlsa — eng erta); keyingi sayoz tushishlar — Test nomzodi
        i = -max(sp_c)[1]
        row = rec.iloc[i]
        out["spring"] = dict(date=rec.index[i], low=float(row["Low"]), close=float(row["Close"]),
                             pierce=float(sup / row["Low"] - 1), vol_ratio=float(row["Volume"] / vavg), pos=i)
    if ut_c:
        i = -max(ut_c)[1]
        row = rec.iloc[i]
        out["upthrust"] = dict(date=rec.index[i], high=float(row["High"]), close=float(row["Close"]),
                               pierce=float(row["High"] / res - 1), vol_ratio=float(row["Volume"] / vavg), pos=i)
    sp = out["spring"]
    if sp is not None:
        after = rec.iloc[sp["pos"] + 1:]
        for j in range(len(after)):
            r = after.iloc[j]
            if r["Low"] <= sp["low"] * 1.02 and r["Close"] >= sup and (r["Volume"] / vavg) < 0.8:
                out["test"] = dict(date=after.index[j], low=float(r["Low"]), vol_ratio=float(r["Volume"] / vavg))
                break
    return out


def find_base(df: pd.DataFrame, lookback: int = 320, comp_max: float = 1.3, min_touch: int = 3,
              widths: Tuple[int, ...] = (160, 120, 90, 70, 55)) -> Dict[str, Any]:
    """Baza (konsolidatsiya) va breakout. Eng soʻnggi TOR savdo diapazonini ("baza") topadi: chuqurligi 4–35%, ichida trend yoʻq,
    yuqori va past chegarasi kamida 2 martadan sinalgan. Narx undan yuqoriga chiqqan boʻlsa — breakout (hajm tasdigʻi va qayta sinov bilan),
    boʻlmasa — baza davom etmoqda. Maqsadlar: baza balandligining 50% va 100% ini breakout darajasidan yuqoriga proyeksiya qilish
    (klassik "measured move" qoidasi). Bu QOIDA — bashorat emas; AI modeliga kirmaydi va tarixiy aniqligi tekshirilmagan."""
    out: Dict[str, Any] = dict(ok=False)
    tail_n = min(lookback, len(df))
    d = df.tail(tail_n)
    n = len(d)
    if n < 140:
        return out
    off = len(df) - n
    H, L, C, V = (d[k].values.astype(float) for k in ("High", "Low", "Close", "Volume"))
    price = float(C[-1])
    sig_ref = float(np.std(np.diff(np.log(C)))) or 1e-9                  # tarixiy kunlik volatillik (butun oyna)

    def touches(mask: np.ndarray) -> int:
        cnt, last = 0, -99
        for i in np.flatnonzero(mask):
            if i - last >= 5:
                cnt += 1
            last = i
        return cnt

    def valid(e: int, W: int) -> Optional[Dict[str, Any]]:
        s = e - W + 1
        if s < 0:
            return None
        h_, l_, c_ = H[s:e + 1], L[s:e + 1], C[s:e + 1]
        hi, lo = float(np.percentile(h_, 97)), float(np.percentile(l_, 3))
        height = hi - lo
        if height <= 0 or hi <= 0 or not (0.04 <= height / hi <= 0.35):
            return None
        if math.log(hi / lo) > comp_max * sig_ref * math.sqrt(W):           # siqilish: diapazon tasodifiy yurishdagidan tordir
            return None
        if abs(c_[-1] - c_[0]) > 0.6 * height:
            return None
        if abs(np.polyfit(np.arange(W), c_, 1)[0] * (W - 1)) > 0.5 * height:
            return None
        if np.mean((c_ >= lo * 0.99) & (c_ <= hi * 1.01)) < 0.88:
            return None
        tt, bt = touches(h_ >= hi - 0.12 * height), touches(l_ <= lo + 0.12 * height)
        if tt < min_touch or bt < min_touch:
            return None
        return dict(s=s, e=e, W=W, hi=hi, lo=lo, height=height, depth=height / hi, top_t=tt, bot_t=bt)

    def refine(b0: Dict[str, Any]) -> Dict[str, Any]:
        """Oyna oxirida breakout kunlari qolib ketgan boʻlsa (baza chegarasi sun'iy koʻtarilib ketadi) — oynani breakout'dan oldingi
        kunga qisqartirib, qayta tekshiradi. Baza yuqorisi: oynaning DASTLABKI 70% idagi eng yuqori narxlardan baholanadi."""
        s, e0, W0 = b0["s"], b0["e"], b0["W"]
        k = max(int(0.7 * W0), 20)
        core_top = float(np.percentile(H[s:s + k], 95))
        for i in range(s + W0 // 2, e0 + 1):
            if C[i] > core_top * 1.01:
                e2 = i - 1
                if e2 - s + 1 >= 40:
                    b2 = valid(e2, e2 - s + 1)
                    if b2:
                        return b2
                break
        return b0

    b = None
    for lag in (0, 2, 4, 6, 8, 11, 15, 20, 26, 33, 41, 50, 60, 75, 90, 105, 120):   # eng yangi bazani qidiramiz
        for W in widths:
            b = valid(n - 1 - lag, W)
            if b:
                b = refine(b)
                b["lag"] = n - 1 - b["e"]
                break
        if b:
            break
    if not b:
        return out
    hi, lo, height, e = b["hi"], b["lo"], b["height"], b["e"]
    aC, aH, aL, aV = C[e + 1:], H[e + 1:], L[e + 1:], V[e + 1:]
    broke = next((j for j, c_ in enumerate(aC) if c_ > hi * 1.01), None)
    if broke is not None:
        state = "breakout" if price >= hi else ("failed" if price >= lo * 0.99 else "breakdown")
    else:
        state = "breakdown" if price < lo * 0.99 else "inside"
    v50 = float(np.mean(V[max(0, e - 49):e + 1])) or 1.0
    res = dict(ok=True, state=state, hi=hi, lo=lo, height=height, depth=b["depth"], W=b["W"], lag=b["lag"],
               top_t=b["top_t"], bot_t=b["bot_t"], start=d.index[b["s"]], end=d.index[e], s_abs=off + b["s"], e_abs=off + e,
               pos=float((price - lo) / height), trigger=float(hi * 1.01), price=price)
    if broke is not None:
        vr = float(np.max(aV[broke:broke + 3]) / v50)
        res.update(broke=d.index[e + 1 + broke], days_since=int(len(aC) - broke), gain=float(price / hi - 1), vol_ratio=vr,
                   confirmed=bool(vr >= 1.5),
                   retest=bool(len(aL) > broke + 1 and float(np.min(aL[broke + 1:])) <= hi * 1.02 and price >= hi))
    tg = []
    for k, frac in ((1, 0.5), (2, 1.0), (3, 1.618)):
        tp = float(hi + frac * height)
        reached = bool(len(aH) and float(np.max(aH)) >= tp)
        tg.append(dict(name=f"T{k}", price=tp, frac=frac, pct=float(tp / price - 1), reached=reached))
    res["targets"] = tg
    return res


def gann_fan(df: pd.DataFrame, ahead: int = 40) -> Dict[str, Any]:
    g = gann_anchor(df).iloc[-1]
    n = len(df)
    pos = int(g["anchor_pos"])
    if pos < 0:
        return dict(ok=False)
    atr = float(atr_series(df).iloc[-1])
    unit = GANN_UNIT_ATR * atr
    low0 = float(g["anchor_low"])
    bars_now = n - 1 - pos
    price = float(df["Close"].iloc[-1])
    lines = []
    for name, ratio in GANN_RATIOS:
        s = ratio * unit
        lines.append(dict(name=name, ratio=ratio, slope=s, now=low0 + s * bars_now, proj=low0 + s * (bars_now + ahead)))
    below = [x for x in lines if x["now"] <= price]
    above = [x for x in lines if x["now"] > price]
    sup = max(below, key=lambda x: x["now"]) if below else None
    res = min(above, key=lambda x: x["now"]) if above else None
    slope_u = (price - low0) / (max(bars_now, 5) * unit) if unit > 0 else float("nan")
    return dict(ok=True, anchor_pos=pos, anchor_date=df.index[pos], anchor_low=low0, bars=bars_now, unit=unit,
                lines=lines, support=sup, resistance=res, slope=float(slope_u), price=price, ahead=ahead)


def candles_last(df: pd.DataFrame, n: int = 5) -> List[Dict[str, Any]]:
    nm = candle_names_series(df).tail(n)
    out = []
    for d, name in nm.items():
        out.append(dict(date=d, name=name, dir=_CANDLE_DIR.get(name, 0.0), text=CANDLE_UZ.get(name, "")))
    return out


# --------------------------------------------------------------------------------------
# Fundamental va yangiliklar (Yahoo Finance; xato boʻlsa tahlil davom etadi)
# --------------------------------------------------------------------------------------
_FUND_KEYS = ["shortName", "sector", "industry", "marketCap", "trailingPE", "forwardPE", "priceToBook",
              "profitMargins", "operatingMargins", "returnOnEquity", "revenueGrowth", "earningsGrowth",
              "debtToEquity", "currentRatio", "dividendYield", "dividendRate", "beta", "targetMeanPrice", "targetHighPrice",
              "targetLowPrice", "numberOfAnalystOpinions", "recommendationKey", "shortPercentOfFloat",
              "heldPercentInsiders", "heldPercentInstitutions"]


def fetch_fundamentals(ticker: str) -> Dict[str, Any]:
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        d = {k: info.get(k) for k in _FUND_KEYS}
        if not any(v is not None for k, v in d.items() if k != "shortName"):
            return dict(ok=False, error="Yahoo fundamental maʼlumot bermadi")
        d["ok"] = True
        return d
    except Exception as e:
        return dict(ok=False, error=f"{type(e).__name__}: {str(e)[:100]}")


def score_fundamentals(f: Dict[str, Any], price: float) -> Tuple[int, List[str]]:
    """Klassik qoidalar boʻyicha oddiy fundamental ball (tarixiy sinovdan oʻtmagan — faqat kontekst)."""
    if not f.get("ok"):
        return 0, []
    s, why = 0, []

    def num(k):
        v = f.get(k)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    g = num("revenueGrowth")
    if g is not None:
        if g > 0.10:
            s += 1; why.append(f"Daromad oʻsishi {g * 100:.0f}% (yuqori)")
        elif g < 0:
            s -= 1; why.append(f"Daromad kamaymoqda ({g * 100:.0f}%)")
    pm = num("profitMargins")
    if pm is not None:
        if pm > 0.15:
            s += 1; why.append(f"Sof foyda marjasi {pm * 100:.0f}% (yaxshi)")
        elif pm < 0:
            s -= 1; why.append("Kompaniya zarar bilan ishlayapti")
    pe = num("forwardPE") or num("trailingPE")
    if pe is not None:
        if 0 < pe <= 25:
            s += 1; why.append(f"P/E {pe:.1f} (moʻtadil)")
        elif pe > 45:
            s -= 1; why.append(f"P/E {pe:.0f} (qimmat)")
    roe = num("returnOnEquity")
    if roe is not None:
        if roe > 0.15:
            s += 1; why.append(f"ROE {roe * 100:.0f}% (yuqori)")
        elif roe < 0:
            s -= 1; why.append("ROE manfiy")
    de = num("debtToEquity")
    if de is not None:
        if de > 200:
            s -= 1; why.append(f"Qarz/kapital {de:.0f}% (yuqori qarz)")
        elif de < 80:
            s += 1; why.append(f"Qarz/kapital {de:.0f}% (past qarz)")
    tg = num("targetMeanPrice")
    if tg and price:
        up = tg / price - 1
        if up > 0.15:
            s += 1; why.append(f"Analitiklar oʻrtacha maqsadi narxdan {up * 100:.0f}% yuqori")
        elif up < -0.05:
            s -= 1; why.append(f"Analitiklar maqsadi narxdan past ({up * 100:.0f}%)")
    return s, why


def fetch_news(ticker: str, n: int = 8) -> Dict[str, Any]:
    try:
        import yfinance as yf
        items = yf.Ticker(ticker).news or []
    except Exception as e:
        return dict(ok=False, error=f"{type(e).__name__}: {str(e)[:100]}", items=[])
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        an = SentimentIntensityAnalyzer()
    except Exception:
        an = None
    out = []
    for it in items[:n]:
        if not isinstance(it, dict):
            continue
        c = it.get("content") if isinstance(it.get("content"), dict) else it
        title = c.get("title") or it.get("title") or ""
        if not title:
            continue
        summ = (c.get("summary") or "")[:300]
        pub = c.get("pubDate") or c.get("displayTime") or it.get("providerPublishTime")
        try:
            pub = pd.to_datetime(pub, unit="s") if isinstance(pub, (int, float)) else pd.to_datetime(pub)
            pub = pub.tz_localize(None) if getattr(pub, "tzinfo", None) is not None else pub
        except Exception:
            pub = None
        prov = c.get("provider") if isinstance(c.get("provider"), dict) else {}
        url = ((c.get("canonicalUrl") or {}) if isinstance(c.get("canonicalUrl"), dict) else {}).get("url") \
            or ((c.get("clickThroughUrl") or {}) if isinstance(c.get("clickThroughUrl"), dict) else {}).get("url") \
            or it.get("link") or ""
        comp = float(an.polarity_scores((title + ". " + summ).strip())["compound"]) if an else 0.0
        out.append(dict(title=title, publisher=prov.get("displayName") or it.get("publisher") or "", date=pub,
                        url=url, sent=comp))
    avg = float(np.mean([x["sent"] for x in out])) if out else float("nan")
    return dict(ok=True, items=out, avg=avg)


# --------------------------------------------------------------------------------------
# AI bloki va yakuniy xulosa
# --------------------------------------------------------------------------------------
def ai_block(st: State, last: pd.Series) -> Optional[Dict[str, Any]]:
    if not last[FEATURES].notna().all():
        return None
    X = model_matrix_from_features(pd.DataFrame([last[FEATURES]]), st.ds.ref)
    probs, pct = {}, {}
    for H, m in st.models.items():
        p = float(m.predict_proba(X)[0, 1])
        probs[H] = p
        pct[H] = float((st.scores[H] < p).mean()) if H in st.scores.columns else float("nan")
    ood = [f_ for f_ in FEATURES_CONT
           if not (st.ds.ref[f_][0] <= float(last[f_]) <= st.ds.ref[f_][-1])]
    return dict(probs=probs, pct=pct, ood=ood)


def build_conclusion(R: Dict[str, Any], st: State) -> Dict[str, Any]:
    votes: List[Dict[str, Any]] = []

    def add(name, sig, w, text, counted=True, kind="klassik"):
        votes.append(dict(name=name, sig=float(sig), w=float(w), text=text, counted=bool(counted), kind=kind))
    last = R["last"]
    ai = R.get("ai")
    reliable = [H for H, e in st.evals.items() if e.verdict in ("weak", "strong")]
    if ai is not None:
        if reliable:
            avg = float(np.nanmean([ai["pct"][H] for H in reliable if H in ai["pct"]]))
            strong = any(st.evals[H].verdict == "strong" for H in reliable)
            sig = max(-1.0, min(1.0, (avg - 0.5) * 2.0))
            add("AI (mashinaviy oʻqitish)", sig, 3.0 if strong else 1.5,
                f"{len(reliable)} ta ufqda tarixiy sinovdan oʻtgan; aksiya universum ichida oʻrtacha {avg * 100:.0f}-persentilda.",
                kind="ai")
        else:
            h0 = st.primary_h
            pc = ai["pct"].get(h0, float("nan"))
            add("AI (mashinaviy oʻqitish)", 0.0, 0.0,
                f"Model tarixiy sinovda ishonchli ustunlik koʻrsatmadi, shu sababli xulosaga QOʻSHILMADI "
                f"(maʼlumot uchun: {HORIZON_UZ.get(h0, h0)} ufqida universumda {pc * 100:.0f}-persentil).",
                counted=False, kind="ai")
    else:
        add("AI (mashinaviy oʻqitish)", 0.0, 0.0, "Bu aksiya uchun AI ball hisoblanmadi (tarix yetarli emas yoki maʼlumot toʻliq emas).",
            counted=False, kind="ai")
    c = float(R["df"]["Close"].iloc[-1])
    s50, s200 = float(last.get("dist_sma50", np.nan)), float(last.get("dist_sma200", np.nan))
    sma50 = float(R["df"]["Close"].rolling(50).mean().iloc[-1])
    sma200 = float(R["df"]["Close"].rolling(200).mean().iloc[-1]) if len(R["df"]) >= 200 else float("nan")
    if np.isfinite(sma200):
        if c > sma50 > sma200:
            add("Trend", 1, 1, "Narx SMA50 va SMA200 dan yuqorida, SMA50 > SMA200 (oʻsish trendi).")
        elif c < sma50 < sma200:
            add("Trend", -1, 1, "Narx SMA50 va SMA200 dan pastda, SMA50 < SMA200 (pasayish trendi).")
        else:
            add("Trend", 0, 1, "Trend aralash: narx va oʻrtacha chiziqlar bir yoʻnalishni koʻrsatmayapti.")
    r20, r60 = float(last.get("ret20", np.nan)), float(last.get("ret60", np.nan))
    if np.isfinite(r60) and np.isfinite(r20):
        if r60 > 0.05 and r20 > 0:
            add("Momentum", 1, 1, f"Oxirgi 3 oyda {r60 * 100:+.1f}%, 1 oyda {r20 * 100:+.1f}% — oʻsish davom etmoqda.")
        elif r60 < -0.05 and r20 < 0:
            add("Momentum", -1, 1, f"Oxirgi 3 oyda {r60 * 100:+.1f}%, 1 oyda {r20 * 100:+.1f}% — pasayish davom etmoqda.")
        else:
            add("Momentum", 0, 1, f"3 oy: {r60 * 100:+.1f}%, 1 oy: {r20 * 100:+.1f}% — aniq yoʻnalish yoʻq.")
    wy = R["wy_code"]
    ev = R["wy_events"]
    if wy == 2:
        add("Wyckoff", 1, 1, "Markup fazasi: trend yuqoriga, talab ustun.")
    elif wy == 4:
        add("Wyckoff", -1, 1, "Markdown fazasi: trend pastga, taklif ustun.")
    elif wy == 1:
        if ev.get("spring") and ev.get("test"):
            add("Wyckoff", 1, 1, "Accumulation + Spring va muvaffaqiyatli Test: koʻtarilish boshlanishi ehtimoli oshgan.")
        elif ev.get("spring"):
            add("Wyckoff", 0.6, 1, "Accumulation + Spring bor, lekin Test hali tasdiqlanmagan.")
        else:
            add("Wyckoff", 0.2, 1, "Accumulation (toʻplanish) diapazoni: hali tasdiq (SOS) yoʻq.")
    elif wy == 3:
        if ev.get("upthrust"):
            add("Wyckoff", -1, 1, "Distribution + Upthrust: yuqoridan yolgʻon chiqish — pasayish xavfi.")
        else:
            add("Wyckoff", -0.3, 1, "Distribution (taqsimlanish) diapazoni: ehtiyotkorlik.")
    bs = R.get("base") or {}
    if bs.get("ok"):
        st_ = bs["state"]
        if st_ == "breakout" and bs["days_since"] <= 60:
            if bs["confirmed"]:
                add("Baza / breakout", 1, 0.5, f"Narx {bs['days_since']} savdo kuni oldin bazadan hajm bilan chiqqan (hajm {bs['vol_ratio']:.1f}×), hozir bazadan {bs['gain'] * 100:+.0f}% yuqorida.")
            else:
                add("Baza / breakout", 0.5, 0.5, f"Narx {bs['days_since']} savdo kuni oldin bazadan chiqqan, lekin hajm tasdigʻi zaif ({bs['vol_ratio']:.1f}×).")
        elif st_ == "inside" and bs["pos"] >= 0.8:
            add("Baza / breakout", 0.2, 0.5, f"Narx bazaning yuqori qismida ({bs['pos'] * 100:.0f}%): breakout {usd(bs['trigger'])} dan yuqorida tasdiqlanadi.")
        elif st_ in ("breakdown", "failed"):
            add("Baza / breakout", -1 if st_ == "breakdown" else -0.5, 0.5,
                "Narx bazaning pastidan chiqdi — baza buzildi." if st_ == "breakdown" else "Breakout muvaffaqiyatsiz: narx bazaga qaytdi.")
    rsi = float(last.get("rsi14", np.nan))
    if np.isfinite(rsi):
        if rsi >= 75:
            add("RSI", -1, 0.5, f"RSI {rsi:.0f}: haddan tashqari xarid — qisqa muddatli tuzatish xavfi.")
        elif rsi <= 25:
            add("RSI", 1, 0.5, f"RSI {rsi:.0f}: haddan tashqari sotilgan — qisqa muddatli tiklanish ehtimoli.")
        else:
            add("RSI", 0, 0.5, f"RSI {rsi:.0f}: neytral zona.")
    cs = R["candles"]
    cd = sum(x["dir"] for x in cs[-3:]) if cs else 0
    names = ", ".join(x["name"] for x in cs[-3:] if x["name"]) or "aniq model yoʻq"
    add("Sham modellari", 1 if cd > 0 else -1 if cd < 0 else 0, 0.5, f"Oxirgi 3 sham: {names}.")
    gf = R["gann"]
    if gf.get("ok") and gf["bars"] >= 8:
        sl = gf["slope"]
        if sl >= 1.0:
            add("Gann burchagi", 1, 0.5, "Narx Gann 1x1 chizigʻidan yuqorida — kuchli trend.")
        elif sl < 0.5:
            add("Gann burchagi", -1, 0.5, "Narx Gann 1x2 chizigʻidan pastda — trend zaif.")
        else:
            add("Gann burchagi", 0, 0.5, "Narx 1x2 va 1x1 chiziqlari orasida — oʻrtacha trend.")
    fu = R["fund"]
    if fu.get("ok"):
        fs, why = R["fund_score"]
        add("Fundamental", 1 if fs >= 2 else -1 if fs <= -2 else 0, 1, "; ".join(why[:3]) or "Maʼlumot yetarli emas.")
    nw = R["news"]
    if nw.get("ok") and len(nw["items"]) >= 3:
        a = nw["avg"]
        add("Yangiliklar", 1 if a >= 0.25 else -1 if a <= -0.25 else 0, 0.5,
            f"Soʻnggi {len(nw['items'])} sarlavha kayfiyati: {a:+.2f} (VADER).")
    cnt = [v for v in votes if v["counted"]]
    wsum = sum(v["w"] for v in cnt) or 1.0
    score = sum(v["sig"] * v["w"] for v in cnt) / wsum
    label = "Ijobiy" if score > 0.25 else "Salbiy" if score < -0.25 else "Neytral / aralash"
    ai_counted = any(v["kind"] == "ai" and v["counted"] for v in votes)
    conf = "oʻrta" if ai_counted else "past"
    return dict(votes=votes, score=float(score), label=label, confidence=conf, ai_counted=ai_counted)


# --------------------------------------------------------------------------------------
# Asosiy tahlil funksiyasi
# --------------------------------------------------------------------------------------
_ANALYSIS_CACHE: Dict[Tuple[str, Optional[float]], Tuple[float, Dict[str, Any]]] = {}


def analyze_ticker(ticker: str, st: State, target: Optional[float] = None, use_cache: bool = True) -> Dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor
    t = normalize_ticker(ticker)
    key = (t, round(target, 4) if target else None)
    if use_cache and key in _ANALYSIS_CACHE and time.time() - _ANALYSIS_CACHE[key][0] < 1800 \
            and _ANALYSIS_CACHE[key][1].get("built_at") == st.built_at:
        return _ANALYSIS_CACHE[key][1]
    in_uni = t in st.data and t != BENCH
    df = st.data[t] if t in st.data else fetch_one(t)
    if len(df) < 120:
        raise DataError(f"'{t}' uchun tarix juda qisqa ({len(df)} kun) — kamida ~120 savdo kuni kerak.")
    market = st.data[BENCH]["Close"]
    pool = ThreadPoolExecutor(max_workers=2)
    f_fund = pool.submit(fetch_fundamentals, t)
    f_news = pool.submit(fetch_news, t)
    F = compute_features(df, market)
    last = F.iloc[-1]
    price = float(df["Close"].iloc[-1])
    ai = ai_block(st, last) if len(df) >= WARMUP + 5 else None
    wy_code_series = wyckoff_code(df)
    wy_code = int(wy_code_series.iloc[-1]) if pd.notna(wy_code_series.iloc[-1]) else 0
    lv = find_levels(df)
    ev = wyckoff_events(df)
    if wy_code in (1, 3) and ev.get("range_ok"):          # trendsiz diapazonda Spring/Upthrust hodisasi faza yorligʻini aniqlashtiradi
        if ev.get("spring") and not ev.get("upthrust"):
            wy_code = 1
        elif ev.get("upthrust") and not ev.get("spring"):
            wy_code = 3
    gf = gann_fan(df)
    bs = find_base(df)
    lvl_list = [(f"R{i + 1}", x["price"]) for i, x in enumerate(lv["resistance"][:2])] + \
               [(f"S{i + 1}", x["price"]) for i, x in enumerate(lv["support"][:2])]
    if bs.get("ok") and bs["state"] in ("inside", "breakout"):           # baza maqsadlari uchun Monte-Karlo ehtimoli
        lvl_list += [(x["name"], x["price"]) for x in bs["targets"] if x["price"] > price]
    mc = mc_forecast(df, target=target, levels=lvl_list)
    try:
        fund = f_fund.result(timeout=15)
    except Exception as e:
        fund = dict(ok=False, error=f"{type(e).__name__}")
    try:
        news = f_news.result(timeout=15)
    except Exception as e:
        news = dict(ok=False, error=f"{type(e).__name__}", items=[])
    pool.shutdown(wait=False)
    R: Dict[str, Any] = dict(ticker=t, in_universe=in_uni, df=df, last=last, price=price, asof=df.index[-1], ai=ai,
                             wy_code=wy_code, wy_events=ev, levels=lv, gann=gf, base=bs, candles=candles_last(df), mc=mc,
                             fund=fund, fund_score=score_fundamentals(fund, price), news=news, target=target,
                             built_at=st.built_at)
    chg = {}
    c = df["Close"]
    for name, n in (("1 kun", 1), ("1 hafta", 5), ("1 oy", 21), ("3 oy", 63), ("6 oy", 126), ("1 yil", 252)):
        if len(c) > n:
            chg[name] = float(c.iloc[-1] / c.iloc[-1 - n] - 1)
    R["chg"] = chg
    R["conclusion"] = build_conclusion(R, st)
    _ANALYSIS_CACHE[key] = (time.time(), R)
    if len(_ANALYSIS_CACHE) > 60:
        for k in sorted(_ANALYSIS_CACHE, key=lambda k: _ANALYSIS_CACHE[k][0])[:20]:
            _ANALYSIS_CACHE.pop(k, None)
    return R


# ======================================================================================
# 7. GRAFIKLAR (matplotlib -> PNG/base64) VA HTML HISOBOT
# ======================================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})

# Rang tokenlari (dataviz: tasdiqlangan kategorik palitra, yorugʻ sirt)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_RED = "#e34948"
BLUE_150, BLUE_300, BLUE_600 = "#b7d3f6", "#6da7ec", "#184f95"
C_GOOD, C_WARN, C_CRIT = "#0ca30c", "#fab219", "#d03b3b"


def _fs(narrow: bool) -> float:
    """Shrift oʻlchami: telefon varianti (4.6 dyuym) ekranda ~1.1 px/pt bilan chiqadi, shuning uchun kattaroq."""
    return 9.5 if narrow else 8.5


def _money(v: float) -> str:
    return f"${v:,.0f}" if abs(v) >= 100 else f"${v:,.2f}"


def _style(ax, grid_y: bool = True, fs: float = 8.5):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", colors=MUTED, labelsize=fs, length=0)
    ax.grid(axis="y", visible=grid_y, color=GRID, linewidth=0.8, linestyle="-")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURF, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img(b64: str, alt: str) -> str:
    return f'<img alt="{_html.escape(alt)}" src="data:image/png;base64,{b64}"/>'


def _nudge(ys: List[float], gap: float) -> List[float]:
    order = np.argsort(ys)
    out = list(ys)
    prev = -np.inf
    for i in order:
        out[i] = max(ys[i], prev + gap)
        prev = out[i]
    return out


def _usd_axis(ax, axis="y"):
    fmt = FuncFormatter(lambda v, _: f"${v:,.0f}" if abs(v) >= 100 else f"${v:,.2f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def chart_price(R: Dict[str, Any], narrow: bool = False) -> str:
    bars, ahead, zone = (100, 22, 30) if narrow else (170, 45, 17)
    fs = _fs(narrow)
    full = R["df"]
    df = full.tail(bars)
    n, off = len(df), len(full) - len(df)
    x = np.arange(n)
    o, h, l, c = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    up = c >= o
    fig = plt.figure(figsize=(4.6, 6.0) if narrow else (9.6, 5.9), dpi=170 if narrow else 110, facecolor=SURF)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.4, 1.0], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=ax)
    _style(ax, fs=fs)
    _style(axv, grid_y=False, fs=fs)
    col = np.where(up, C_BLUE, C_RED)
    ax.vlines(x, l, h, colors=col, linewidth=1.0, zorder=2)
    body = np.maximum(np.abs(c - o), (h - l) * 0.012 + 1e-9)
    ax.bar(x, body, bottom=np.minimum(o, c), width=0.64, color=col, linewidth=0, zorder=3)
    s50 = full["Close"].rolling(50).mean().iloc[off:].values
    s200 = full["Close"].rolling(200).mean().iloc[off:].values
    ax.plot(x, s50, color=C_ORANGE, lw=1.6, zorder=4)
    ax.plot(x, s200, color=C_AQUA, lw=1.6, zorder=4)
    lo_y, hi_y = float(l.min()), float(h.max())
    span0 = hi_y - lo_y
    ylo, yhi = lo_y - 0.05 * span0, hi_y + 0.12 * span0
    labels: List[Tuple[float, str]] = []
    gf = R["gann"]
    if gf.get("ok"):
        a = gf["anchor_pos"] - off
        xs = np.arange(max(a, 0), n + ahead)
        for ln in gf["lines"]:
            if ln["name"] in ("4x1", "2x1", "1x1", "1x2", "1x4"):
                ys = gf["anchor_low"] + ln["slope"] * (xs - a)
                is11 = ln["name"] == "1x1"
                ax.plot(xs, ys, color=INK2 if is11 else AXIS, lw=1.2 if is11 else 0.9, zorder=1,
                        ls="-" if is11 else (0, (1, 2)))
                if is11 or ln["name"] in ("2x1", "1x2", "1x4"):
                    yend = float(ys[-1])
                    if ylo <= yend <= yhi:
                        labels.append((yend, "Gann " + ln["name"]))
        if a >= 0:
            ax.plot([a], [gf["anchor_low"]], marker="o", ms=7, color=INK2, mec=SURF, mew=1.5, zorder=6, ls="none")
    lv = R["levels"]
    for k, (items, tag) in enumerate(((lv["resistance"][:2], "R"), (lv["support"][:2], "S"))):
        for i, it in enumerate(items):
            ax.hlines(it["price"], 0, n + ahead, colors=MUTED, linestyles=(0, (4, 3)), linewidth=0.9, zorder=1)
            pr = float(it["price"])
            labels.append((pr, f"{tag}{i + 1} {_money(pr)}" if narrow else f"{tag}{i + 1}  ${pr:,.2f}"))
    bs = R.get("base") or {}
    has_base = False
    if bs.get("ok") and bs["e_abs"] - off >= 0:                            # baza: soyali toʻgʻri toʻrtburchak + breakout daraja + maqsadlar
        from matplotlib.patches import Rectangle
        xs0, xe0 = max(bs["s_abs"] - off, 0), bs["e_abs"] - off
        ax.add_patch(Rectangle((xs0 - 0.5, bs["lo"]), xe0 - xs0 + 1, bs["height"], facecolor=C_BLUE, alpha=0.08, edgecolor="none", zorder=0))
        ax.hlines(bs["hi"], xs0, n + ahead, colors=C_BLUE, linestyles=(0, (1, 1.6)), linewidth=1.1, zorder=1)
        labels.append((bs["hi"], "Baza usti" if narrow else f"Baza usti  ${bs['hi']:,.2f}"))
        has_base = True
        if bs["state"] in ("inside", "breakout"):
            for tg_ in bs["targets"]:
                if tg_["price"] > float(c[-1]):
                    ax.hlines(tg_["price"], max(n - 30, 0), n + ahead, colors=C_GOOD, linestyles=(0, (5, 3)), linewidth=1.0, zorder=1)
                    labels.append((tg_["price"], (f"M{tg_['name'][1]} " + _money(tg_["price"])) if narrow else f"Maqsad {tg_['name'][1]}  ${tg_['price']:,.2f}"))
                    yhi = max(yhi, min(tg_["price"] + 0.04 * span0, hi_y + 0.55 * span0))
    ev = R["wy_events"]
    for key, yk, colr, txt in (("spring", "low", C_BLUE, "Spring"), ("upthrust", "high", C_RED, "Upthrust")):
        e = ev.get(key)
        if e and e["date"] in df.index:
            ix = df.index.get_loc(e["date"])
            ax.plot([ix], [e[yk]], marker="o", ms=8, color=colr, mec=SURF, mew=1.8, zorder=7, ls="none")
            ax.annotate(txt, (ix, e[yk]), xytext=(0, -14 if key == "spring" else 12), textcoords="offset points",
                        ha="center", fontsize=fs - 0.5, color=INK2)
    if ev.get("test") and ev["test"]["date"] in df.index:
        ix = df.index.get_loc(ev["test"]["date"])
        ax.plot([ix], [ev["test"]["low"]], marker="o", ms=8, color=C_BLUE, mec=SURF, mew=1.8, zorder=7, ls="none")
        ax.annotate("Test", (ix, ev["test"]["low"]), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=fs - 0.5, color=INK2)
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(-1, n + ahead + zone)
    if labels:
        gap = (ax.get_ylim()[1] - ax.get_ylim()[0]) * (0.045 if narrow else 0.035)
        ys = _nudge([y for y, _ in labels], gap)
        for (y0, t), y1 in zip(labels, ys):
            ax.text(n + ahead + 1.0, y1, t, fontsize=fs - 0.5, color=INK2, va="center", ha="left")
    _usd_axis(ax)
    ax.tick_params(labelbottom=False)
    vol = df["Volume"].values.astype(float)
    if vol.max() > 0:
        axv.bar(x, vol, width=0.64, color=AXIS, linewidth=0)
        axv.yaxis.set_major_locator(MaxNLocator(2, prune="lower"))      # "0" yorligʻi sana yorligʻi bilan toʻqnashmasin
        axv.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M" if v >= 1e6 else f"{v / 1e3:.0f}K"))
    else:
        axv.set_yticks([])
    ticks = np.linspace(0, n - 1, 3 if narrow else 6).round().astype(int)
    axv.set_xticks(ticks)
    axv.set_xticklabels([df.index[i].strftime("%d %b '%y") for i in ticks], fontsize=fs, color=MUTED)
    handles = [Patch(color=C_BLUE, label="Oʻsgan" if narrow else "Oʻsgan sham"),
               Patch(color=C_RED, label="Tushgan" if narrow else "Tushgan sham"),
               Line2D([0], [0], color=C_ORANGE, lw=1.6, label="SMA50"),
               Line2D([0], [0], color=C_AQUA, lw=1.6, label="SMA200"),
               Line2D([0], [0], color=MUTED, lw=0.9, ls=(0, (4, 3)), label="S/R" if narrow else "Support / resistance"),
               Line2D([0], [0], color=INK2, lw=1.2, label="Gann 1x1")]
    if has_base:
        handles.append(Patch(facecolor=C_BLUE, alpha=0.25, label="Baza"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3 if narrow else len(handles), frameon=False,
              fontsize=fs - (0 if narrow else 0.5), labelcolor=INK2, handlelength=1.4, columnspacing=1.1, borderaxespad=0.2)
    return _b64(fig)


def chart_cone(R: Dict[str, Any], narrow: bool = False) -> str:
    mc = R["mc"]
    bands, P0 = mc["bands"], mc["P0"]
    hmax = bands.shape[1]
    x = np.arange(0, hmax + 1)

    def full(a):
        return np.concatenate([[P0], a])
    fs = _fs(narrow)
    mny = _money if narrow else (lambda v: f"${v:,.2f}")
    fig, ax = plt.subplots(figsize=(4.6, 3.9) if narrow else (9.2, 4.0), dpi=170 if narrow else 110, facecolor=SURF)
    _style(ax, fs=fs)
    ax.fill_between(x, full(bands[0]), full(bands[4]), color=BLUE_150, linewidth=0)
    ax.fill_between(x, full(bands[1]), full(bands[3]), color=BLUE_300, linewidth=0)
    ax.plot(x, full(bands[2]), color=C_BLUE, lw=2.0, solid_capstyle="round")
    ax.hlines(P0, 0, hmax, colors=AXIS, linewidth=0.8, zorder=0)
    labs = [(float(bands[4][-1]), f"95%: {mny(bands[4][-1])}"), (float(bands[2][-1]), f"Median: {mny(bands[2][-1])}"),
            (float(bands[0][-1]), f"5%: {mny(bands[0][-1])}")]
    tg = mc.get("target")
    if tg:
        ax.hlines(tg["price"], 0, hmax, colors=INK2, linewidth=1.0, linestyles=(0, (4, 3)), zorder=1)
        labs.append((tg["price"], f"Maqsad: {mny(tg['price'])}"))
    ys = ax.get_ylim()
    gap = (max(ys[1], max(l[0] for l in labs)) - min(ys[0], min(l[0] for l in labs))) * (0.06 if narrow else 0.045)
    adj = _nudge([y for y, _ in labs], gap)
    for (y0, t), y1 in zip(labs, adj):
        ax.text(hmax + 2, y1, t, fontsize=fs, color=INK2, va="center", ha="left")
    hs = [h for h in HORIZONS if 20 <= h <= hmax and (not narrow or h in (20, 60, 125))]
    ax.set_xticks(hs)
    ax.set_xticklabels([f"{HORIZON_UZ.get(h, h)}\n({h} kun)" for h in hs], fontsize=fs - 0.5, color=MUTED)
    ax.set_xlim(0, hmax * (1.55 if narrow else 1.22))
    _usd_axis(ax)
    handles = [Patch(color=BLUE_150, label="90% diapazon (5–95%)"), Patch(color=BLUE_300, label="50% diapazon (25–75%)"),
               Line2D([0], [0], color=C_BLUE, lw=2.0, label="Median")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1 if narrow else 3, frameon=False,
              fontsize=fs, labelcolor=INK2, borderaxespad=0.2)
    return _b64(fig)


def chart_tstat(st: State, narrow: bool = False) -> str:
    hs = sorted(st.evals)
    vals = [st.evals[h].ic_t if np.isfinite(st.evals[h].ic_t) else 0.0 for h in hs]
    cols = [BLUE_600 if st.evals[h].verdict == "strong" else C_BLUE if st.evals[h].verdict == "weak" else AXIS for h in hs]
    fs = _fs(narrow)
    fig, ax = plt.subplots(figsize=(4.6, 3.7) if narrow else (8.6, 3.5), dpi=170 if narrow else 110, facecolor=SURF)
    _style(ax, fs=fs)
    xs = np.arange(len(hs))
    ax.bar(xs, vals, width=0.58 if narrow else 0.52, color=cols, linewidth=0)
    ax.axhline(0, color=AXIS, lw=0.8)
    for yv, txt in ((T_WEAK, f"t = {T_WEAK:.1f}\nzaif dalil"), (T_STRONG, f"t = {T_STRONG:.1f}\nkuchli dalil")):
        ax.hlines(yv, -0.6, len(hs) - 0.4, colors=MUTED, linewidth=0.9, linestyles=(0, (4, 3)), zorder=1)
        ax.text(len(hs) - 0.3, yv, txt, fontsize=fs - 0.5, color=INK2, ha="left", va="center")
    ax.set_xlim(-0.6, len(hs) + (1.5 if narrow else 0.9))
    for i, v in enumerate(vals):
        ax.text(i, v + (0.08 if v >= 0 else -0.08), f"{v:+.1f}" if narrow else f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=fs, color=INK, bbox=dict(facecolor=SURF, edgecolor="none", pad=1.2), zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels([HORIZON_UZ.get(h, str(h)).replace(" ", "\n") if narrow else HORIZON_UZ.get(h, str(h)) for h in hs],
                       fontsize=fs, color=MUTED)
    top = max(max(vals), T_STRONG) + 0.9
    ax.set_ylim(min(min(vals) - 0.6, -0.8), top)
    handles = [Patch(color=BLUE_600, label="Kuchli dalil"), Patch(color=C_BLUE, label="Zaif dalil"),
               Patch(color=AXIS, label="Dalil yoʻq")]
    ax.legend(handles=handles, loc="upper left", ncol=3, frameon=False, fontsize=fs - (1.0 if narrow else 0),
              labelcolor=INK2, columnspacing=1.0, handlelength=1.2)
    return _b64(fig)


def chart_backtest(bt: BtResult, narrow: bool = False) -> str:
    fs = _fs(narrow)
    fig, ax = plt.subplots(figsize=(4.6, 3.9) if narrow else (9.2, 4.0), dpi=170 if narrow else 110, facecolor=SURF)
    _style(ax, fs=fs)
    style = {"AI (top-%d)" % bt.top_n: (C_BLUE, 2.0, "-"), "Momentum (top-%d)" % bt.top_n: (C_ORANGE, 1.6, "-"),
             "Universum (teng ulush)": (C_AQUA, 1.6, "-"), "SPY": (MUTED, 1.4, (0, (4, 3)))}
    for name, eq in bt.curves.items():
        col, lw, ls = style.get(name, (INK2, 1.4, "-"))
        ax.plot(eq.index, eq.values, color=col, lw=lw, ls=ls, label=f"{name}   {(eq.iloc[-1] - 1) * 100:+.0f}%",
                solid_capstyle="round")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}x"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=4 if narrow else 7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper left", frameon=False, fontsize=fs, labelcolor=INK2)
    return _b64(fig)


def chart_importance(ev: EvalResult, narrow: bool = False) -> str:
    import textwrap
    items = sorted(ev.importance.items(), key=lambda kv: kv[1])
    fs = _fs(narrow)
    names = [("\n".join(textwrap.wrap(k, 17)) if narrow else k) for k, _ in items]
    vals = [v * 100 for _, v in items]
    fig, ax = plt.subplots(figsize=(4.6, 4.7) if narrow else (8.6, 3.9), dpi=170 if narrow else 110, facecolor=SURF)
    _style(ax, grid_y=False, fs=fs)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ys = np.arange(len(vals))
    ax.barh(ys, vals, height=0.5, color=C_BLUE, linewidth=0)
    ax.axvline(0, color=AXIS, lw=0.8)
    lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
    span = max(hi - lo, 0.1)
    off = 0.02 * span
    for y, v in zip(ys, vals):
        v = 0.0 if abs(v) < 0.005 else v
        ax.text(v + (off if v >= 0 else -off), y, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=fs, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=fs, color=INK2)
    # chap tomonda manfiy qiymat yozuvlari uchun joy (yozuv kengligi ~ 14-22% oraliq)
    ax.set_xlim(lo - (0.24 if narrow else 0.14) * span if lo < 0 else lo - 0.03 * span, hi + (0.2 if narrow else 0.14) * span)
    ax.set_xlabel("Guruh aralashtirilganda AUC qancha tushadi (p.p.)" if narrow else
                  "Guruh aralashtirilganda AUC qancha tushadi (foiz punkti)", fontsize=fs, color=MUTED)
    return _b64(fig)


def _pic(fn, alt: str) -> str:
    """Keng (kompyuter) va tor (telefon) variantli rasm: <picture> brauzer oʻlchamiga qarab tanlaydi."""
    wide, nar = fn(False), fn(True)
    return (f'<picture><source media="(max-width: 640px)" srcset="data:image/png;base64,{nar}"/>'
            f'<img alt="{_html.escape(alt)}" src="data:image/png;base64,{wide}"/></picture>')


# --------------------------------------------------------------------------------------
# HTML bloklar
# --------------------------------------------------------------------------------------
CSS = """
<style>
.ai{--ink:var(--body-text-color,#0b0b0b);--ink2:var(--body-text-color-subdued,#52514e);--line:var(--border-color-primary,#d9d8d0);
 --card:var(--background-fill-secondary,#f6f6f3);font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:var(--ink);
 line-height:1.55;font-size:15px;max-width:1080px;margin:0 auto}
.ai *{box-sizing:border-box}
.ai h2{font-size:1.18rem;margin:1.7em 0 .5em;font-weight:650}
.ai h3{font-size:1.02rem;margin:1.3em 0 .4em;font-weight:650}
.ai p{margin:.5em 0}
.ai .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.ai .good{border-left:5px solid #0ca30c}.ai .warn{border-left:5px solid #fab219}.ai .bad{border-left:5px solid #d03b3b}.ai .info{border-left:5px solid #2a78d6}
.ai .row{display:flex;flex-wrap:wrap;gap:18px;justify-content:space-between;align-items:flex-start}
.ai .big{font-size:2rem;font-weight:650;line-height:1.15}
.ai .lbl{font-size:.82rem;color:var(--ink2)}
.ai .muted{color:var(--ink2)}.ai .small{font-size:.84rem}
.ai .grid{display:grid;gap:10px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));margin:10px 0}
.ai .kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.ai .kpi .v{font-size:1.35rem;font-weight:650}
.ai .wrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
.ai table{border-collapse:collapse;width:100%;font-size:.9rem}
.ai th,.ai td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
.ai th{font-weight:600;color:var(--ink2);font-size:.82rem;vertical-align:bottom;line-height:1.3}
.ai td:first-child{white-space:nowrap}
.ai td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.ai th.n{text-align:right}
.ai tr.dim td{color:var(--ink2)}
.ai .tag{display:inline-block;padding:1px 9px;border-radius:999px;font-size:.78rem;border:1px solid var(--line);white-space:nowrap}
.ai .tag.good{border-color:#0ca30c;border-left-width:1px}.ai .tag.warn{border-color:#c98500;border-left-width:1px}.ai .tag.bad{border-color:#d03b3b;border-left-width:1px}
.ai .bar{display:inline-block;width:54px;height:6px;background:var(--line);border-radius:3px;vertical-align:middle;margin-left:7px;overflow:hidden}
.ai .bar i{display:block;height:100%;background:#2a78d6;border-radius:3px}
.ai picture{display:block}.ai img{max-width:100%;height:auto;border-radius:8px;display:block;margin:8px auto}
.ai a{color:#2a78d6}
.ai ul{margin:.4em 0 .4em 1.2em;padding:0}
.ai .chips{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0}
.ai .note{font-size:.85rem;color:var(--ink2);border-top:1px solid var(--line);margin-top:14px;padding-top:10px}
.ai .pick{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px 12px;margin:10px 0}
.ai .pick .ph{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 12px}
.ai .pick .rk{display:inline-block;min-width:1.7em;text-align:center;border-radius:999px;background:var(--line);font-size:.8rem;padding:0 7px}
.ai .pick .tk{font-size:1.2rem;font-weight:700}
.ai .pick .px{font-variant-numeric:tabular-nums;font-weight:600}
.ai .pick .ctx{font-size:.84rem;color:var(--ink2)}
.ai .pg{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:8px 0 2px}
.ai .pc{border:1px solid var(--line);border-radius:8px;padding:6px 9px}
.ai .pc .lbl{display:block}
.ai .pc b{font-variant-numeric:tabular-nums;font-size:1.05rem}
.ai .pc .bar{display:block;width:100%;margin:4px 0 0}
.ai .pick .pf{font-size:.84rem;color:var(--ink2);margin-top:6px}
.ai .mo{display:none}
.ai .vote{display:grid;grid-template-columns:190px 84px 64px 1fr;gap:2px 12px;padding:8px 4px;border-bottom:1px solid var(--line);align-items:baseline}
.ai .vote.vh{font-weight:600;color:var(--ink2);font-size:.82rem}
.ai .vote.dim{color:var(--ink2)}
.ai .vote .vs{font-variant-numeric:tabular-nums;font-weight:600;white-space:nowrap}
.ai .vote .vw{color:var(--ink2);font-size:.84rem}
.ai .vote .vt{font-size:.84rem}
@media (max-width:640px){
 .ai .mo{display:inline}.ai .dm{display:none}
 .ai{font-size:14px}.ai .big{font-size:1.6rem}
 .ai table{font-size:.8rem}
 .ai th,.ai td{padding:6px 4px}
 .ai th{font-size:.72rem}
 .ai table.wf td:first-child{white-space:normal}
 .ai .hm{display:none}
 .ai .bar{width:32px;margin-left:4px}
 .ai .pg{grid-template-columns:repeat(2,1fr)}
 .ai .grid{grid-template-columns:repeat(2,1fr)}
 .ai .kpi .v{font-size:1.15rem}
 .ai h2{font-size:1.1rem}
 .ai .vote{grid-template-columns:1fr auto;padding:8px 2px}
 .ai .vote.vh,.ai .vote .vw{display:none}
 .ai .vote .vt{grid-column:1/-1}
}
</style>
"""

FEATURE_UZ = {
    "ret1": "1 kunlik daromad", "ret5": "1 haftalik daromad", "ret10": "2 haftalik daromad", "ret20": "1 oylik daromad",
    "ret60": "3 oylik daromad", "ret120": "6 oylik daromad", "mom_12_1": "12−1 oylik momentum",
    "vol20": "1 oylik volatillik", "vol60": "3 oylik volatillik", "vol_ratio_20_60": "volatillik oʻzgarishi",
    "atr_pct": "kunlik tebranish (ATR)", "beta60": "beta", "dist_sma10": "SMA10 dan masofa", "dist_sma30": "SMA30 dan masofa",
    "dist_sma50": "SMA50 dan masofa", "dist_sma200": "SMA200 dan masofa", "pos60": "60 kunlik diapazondagi oʻrni",
    "dist_52w_high": "52 hafta yuqorisidan masofa", "dist_52w_low": "52 hafta pastidan masofa", "rsi14": "RSI",
    "vol_ratio10": "hajm (10 kunlik nisbat)", "vol_trend": "hajm trendi", "gann_slope": "Gann burchagi",
    "candle_dir": "sham yoʻnalishi", "wy_accum": "Wyckoff: accumulation", "wy_markup": "Wyckoff: markup",
    "wy_distrib": "Wyckoff: distribution", "wy_markdown": "Wyckoff: markdown",
}
VERDICT_UZ = {"strong": ("Kuchli dalil", "good", "✔"), "weak": ("Zaif dalil", "warn", "~"),
              "none": ("Dalil yoʻq", "bad", "✖"), "inverse": ("Teskari", "bad", "✖")}
esc = _html.escape


def usd(x, d: int = 2) -> str:
    return "—" if x is None or not np.isfinite(x) else f"${x:,.{d}f}"


def usd_r(x) -> str:
    """Moslashuvchan narx: kompyuterda tiyinlar bilan, telefonda butun dollar (joyni tejaydi)."""
    if x is None or not np.isfinite(x):
        return "—"
    if abs(x) < 100:
        return usd(x)
    return f'<span class="dm">{usd(x)}</span><span class="mo">{usd(x, 0)}</span>'


def pc(x, d: int = 1, sign: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def verdict_chip(v: str) -> str:
    lab, cls, ic = VERDICT_UZ.get(v, VERDICT_UZ["none"])
    return f'<span class="tag {cls}">{ic} {lab}</span>'


def prob_cell(p: float) -> str:
    if not np.isfinite(p):
        return '<td class="n">—</td>'
    w = max(0.0, min(100.0, (p - 0.40) / 0.25 * 100.0))
    return f'<td class="n">{p * 100:.1f}%<span class="bar"><i style="width:{w:.0f}%"></i></span></td>'


DASH_CELL = '<td class="n">—</td>'


def prob_card(label: str, p: float) -> str:
    """Tavsiya kartasidagi bitta ufq: yorliq + foiz + kichik chiziq."""
    if not np.isfinite(p):
        return f'<div class="pc"><span class="lbl">{esc(label)}</span><b>—</b></div>'
    w = max(0.0, min(100.0, (p - 0.40) / 0.25 * 100.0))
    return (f'<div class="pc"><span class="lbl">{esc(label)}</span><b>{p * 100:.1f}%</b>'
            f'<span class="bar"><i style="width:{w:.0f}%"></i></span></div>')


def sig_cell(s: float) -> str:
    if s > 0.15:
        return f'<td class="n">▲ {s:+.1f}</td>'
    if s < -0.15:
        return f'<td class="n">▼ {s:+.1f}</td>'
    return f'<td class="n">● {s:+.1f}</td>'


_TD_RE = re.compile(r"<td((?:\s[^>]*)?)>")


def _hide_cells(row: str, hide: set) -> str:
    """Qatordagi berilgan indeksli <td> larga 'hm' (telefonda yashirish) sinfini qoʻshadi."""
    if not hide:
        return row
    i = -1

    def rep(m):
        nonlocal i
        i += 1
        attrs = m.group(1) or ""
        if i in hide:
            attrs = attrs.replace('class="', 'class="hm ', 1) if 'class="' in attrs else attrs + ' class="hm"'
        return f"<td{attrs}>"
    return _TD_RE.sub(rep, row)


def _table(head: List[str], rows: List[str], num_cols: Optional[List[int]] = None,
           hide_mobile: Optional[List[int]] = None, wrap_first: bool = False) -> str:
    """HTML jadval. num_cols — oʻngga tekislanadigan ustunlar; hide_mobile — telefonda (≤640px) yashiriladigan ustunlar;
    wrap_first — telefonda birinchi ustun matni qatorlarga boʻlinishi mumkin."""
    nc, hm = set(num_cols or []), set(hide_mobile or [])

    def th(i: int, x: str) -> str:
        cls = " ".join(c for c in (("n" if i in nc else ""), ("hm" if i in hm else "")) if c)
        return f'<th class="{cls}">{esc(x)}</th>' if cls else f"<th>{esc(x)}</th>"
    h = "".join(th(i, x) for i, x in enumerate(head))
    body = "".join(_hide_cells(r, hm) for r in rows)
    tcls = ' class="wf"' if wrap_first else ""
    return f'<div class="wrap"><table{tcls}><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def error_html(msg: str, hint: str = "") -> str:
    return (CSS + f'<div class="ai"><div class="card bad"><b>✖ Xatolik</b><p>{esc(msg)}</p>'
            + (f'<p class="muted small">{esc(hint)}</p>' if hint else "") + "</div></div>")


def evidence_banner(st: State) -> str:
    ev = st.evals[st.primary_h]
    lab, cls, ic = VERDICT_UZ[ev.verdict]
    hl = HORIZON_UZ.get(st.primary_h, str(st.primary_h))
    others = "".join(f'<span class="tag {VERDICT_UZ[e.verdict][1]}">{VERDICT_UZ[e.verdict][2]} {HORIZON_UZ.get(h, h)}: '
                     f'{VERDICT_UZ[e.verdict][0]}</span>' for h, e in sorted(st.evals.items()))
    if ev.verdict == "strong":
        head = f"{ic} Modelning tarixiy ustunligi: KUCHLI dalil ({hl} ufqi)"
        body = (f"Model out-of-sample sinovda tasodifdan statistik ahamiyatli yaxshi chiqdi: IC t-statistikasi {ev.ic_t:+.2f} "
                f"(qatʼiy chegara {T_STRONG}), AUC {ev.auc:.3f}, {ev.pos_folds}/{len(ev.folds)} test davrida ijobiy.")
    elif ev.verdict == "weak":
        head = f"{ic} Modelning tarixiy ustunligi: ZAIF dalil ({hl} ufqi)"
        body = (f"Out-of-sample sinovda ustunlik koʻrinadi (IC t={ev.ic_t:+.2f}, AUC {ev.auc:.3f}), lekin u qatʼiy chegaradan "
                f"({T_STRONG}) past. Yetti ufq sinalgani uchun bu tasodif boʻlishi ham mumkin — roʻyxatga ehtiyotkorlik bilan qarang.")
    elif ev.verdict == "inverse":
        head = f"{ic} Model teskari ishlamoqda ({hl} ufqi)"
        body = f"IC t-statistikasi {ev.ic_t:+.2f}: yuqori ball olgan aksiyalar tarixda oʻrtachadan ORTDA qolgan. Roʻyxatga ishonmang."
    else:
        head = f"{ic} Modelning tarixiy ustunligi: DALIL YOʻQ ({hl} ufqi)"
        body = (f"Out-of-sample sinovda model tasodifdan ishonchli darajada yaxshi emas (IC t={ev.ic_t:+.2f}, AUC {ev.auc:.3f}). "
                f"Quyidagi roʻyxat — modelning eng yuqori ball bergan nomzodlari, lekin ular TAVSIYA EMAS: ularning ortida isbotlangan ustunlik yoʻq.")
    return (f'<div class="card {cls}"><b>{head}</b><p>{body}</p><div class="chips">{others}</div>'
            f'<p class="small muted">Bu — universum ({len(st.ds.latest)} ta yirik aksiya) boʻyicha, oʻtmishda hech qachon koʻrilmagan '
            f'davrlarda tekshirilgan natija. Kelajak kafolati emas.</p></div>')


def bt_summary_html(st: State) -> str:
    bt = st.bt
    if bt is None:
        return '<p class="muted">Backtest uchun yetarli davr yoʻq.</p>'
    rows = []
    for name, s in bt.stats.items():
        rows.append(f'<tr><td>{esc(name)}</td><td class="n">{pc(s["cagr"], 1, True)}</td><td class="n">{s["sharpe"]:.2f}</td>'
                    f'<td class="n">{pc(s["mdd"], 1)}</td><td class="n">{pc(s["win"], 0)}</td></tr>')
    tbl = _table(["Strategiya", "Yillik daromad", "Sharpe", "Eng katta tushish", "Foydali davrlar"], rows, [1, 2, 3, 4],
                 hide_mobile=[4], wrap_first=True)
    return (tbl + f'<p class="small muted">{bt.n_periods} ta oʻzaro kesishmaydigan {HORIZON_UZ.get(bt.H, bt.H)}lik davr; har tomonga '
            f'{bt.cost_bps:.0f} bp xarajat hisobga olingan. AI alfasi (universumga nisbatan, beta tuzatilgan): '
            f'{pc(bt.alpha, 2, True)}/davr, t={bt.t_alpha:+.2f}; AI − universum farqi t={bt.t_vs_ew:+.2f}.</p>')


def reco_picks(st: State, top_n: int = TOP_N) -> Tuple[pd.DataFrame, List[int]]:
    sc = st.scores
    reliable = [H for H, e in st.evals.items() if e.verdict in ("weak", "strong") and H in sc.columns]
    use = reliable if reliable else [st.primary_h]
    comp = sc[use].rank(pct=True).mean(axis=1)
    top = comp.sort_values(ascending=False).head(top_n).index
    return sc.loc[top], use


def render_reco(st: State, top_n: int = TOP_N, deep: bool = True) -> str:
    picks, use = reco_picks(st, top_n)
    lat = st.ds.latest
    shown = [h for h in (5, 20, 60, 125) if h in st.scores.columns] or sorted(st.scores.columns)[:4]
    cards = []
    for i, (t, r) in enumerate(picks.iterrows(), 1):
        f = lat.loc[t]
        wy = int(1 * f["wy_accum"] + 2 * f["wy_markup"] + 3 * f["wy_distrib"] + 4 * f["wy_markdown"])
        flags = []
        if f["rsi14"] >= 75:
            flags.append("RSI yuqori (haddan tashqari xarid)")
        if f["dist_52w_high"] > -0.02:
            flags.append("52 hafta yuqorisi yaqinida")
        if f["vol60"] * math.sqrt(252) > 0.6:
            flags.append("juda volatil")
        if f["ret20"] > 0.25:
            flags.append("1 oyda +25% dan koʻp oʻsgan")
        pcs = "".join(prob_card(HORIZON_UZ.get(h, str(h)), float(r[h])) for h in shown)
        flag_html = f'<div class="pf">⚠ {esc("; ".join(flags))}</div>' if flags else ""
        cards.append(f'<div class="pick"><div class="ph"><span class="rk">{i}</span><span class="tk">{esc(t)}</span>'
                     f'<span class="px">{usd(st.ds.latest_close.get(t, np.nan))}</span>'
                     f'<span class="ctx">{esc(WYCKOFF_UZ.get(wy, ""))} · RSI {f["rsi14"]:.0f}</span></div>'
                     f'<div class="pg">{pcs}</div>{flag_html}</div>')
    tbl = "".join(cards)
    rank_txt = ("Reyting: tarixiy sinovdan oʻtgan ufqlar ("
                + ", ".join(HORIZON_UZ.get(h, str(h)) for h in use) + ") ballarining oʻrtacha persentili boʻyicha."
                if any(st.evals[h].verdict in ("weak", "strong") for h in use) else
                f"Reyting: sinovdan oʻtgan ufq yoʻq, shuning uchun faqat asosiy ({HORIZON_UZ.get(st.primary_h, st.primary_h)}) ball boʻyicha.")
    title = "AI tanlagan nomzodlar" if st.evals[st.primary_h].verdict in ("weak", "strong") else "Modelning eng yuqori ball bergan nomzodlari (isbotlanmagan signal)"
    out = [CSS, '<div class="ai">',
           f'<div class="card info"><div class="row"><div><div class="lbl">Maʼlumot sanasi</div><b>{st.asof.date()}</b></div>'
           f'<div><div class="lbl">Universum</div><b>{len(st.ds.latest)} ta aksiya</b></div>'
           f'<div><div class="lbl">Tahlil vaqti</div><b>{st.built_at:%Y-%m-%d %H:%M}</b></div>'
           f'<div><div class="lbl">Keyingi yangilanish</div><b>~{REFRESH_HOURS} soatdan keyin</b></div></div></div>',
           evidence_banner(st),
           f"<h2>{esc(title)}</h2>",
           f'<p class="muted small">Ball = aksiyaning oʻrtacha aksiyadan (bozor yoʻnalishi tozalangan holda) yaxshiroq boʻlish ehtimoli. '
           f'50% — neytral; “narx koʻtariladi” degani EMAS. {esc(rank_txt)}</p>', tbl,
           "<h2>Tarixiy sinov natijasi (backtest)</h2>", bt_summary_html(st)]
    if st.bt is not None:
        out.append(_pic(lambda n: chart_backtest(st.bt, n), "Backtest: AI, momentum, universum va SPY"))
    out.append('<p class="small muted">Diqqat: universum — BUGUNGI yirik aksiyalar. Oʻtmishda yirik boʻlib, keyin tushib ketgan yoki '
               'roʻyxatdan chiqqan kompaniyalar yoʻq (survivorship bias) — bu barcha strategiyalar natijasini odatda yuqoriroq koʻrsatadi.</p>')
    if deep and len(picks):
        t1 = str(picks.index[0])
        out.append(f"<h2>Birinchi nomzodning chuqur tahlili: {esc(t1)}</h2>")
        try:
            R = analyze_ticker(t1, st)
            out.append(render_analysis_body(R, st))
        except Exception as e:
            out.append(f'<div class="card warn">Chuqur tahlilni tayyorlab boʻlmadi: {esc(str(e))}. '
                       f'“Istalgan aksiya” boʻlimida {esc(t1)} ni qoʻlda tahlil qilib koʻring.</div>')
    out.append(DISCLAIMER)
    out.append("</div>")
    return "".join(out)


DISCLAIMER = ('<div class="note">Bu dastur taʼlim va tadqiqot maqsadida; moliyaviy maslahat EMAS. Hech bir model kelajakni kafolatlamaydi — '
              'aksiyalar narxi tushishi, siz kiritgan mablagʻning bir qismi yoki hammasini yoʻqotishingiz mumkin. Qaror qabul qilishdan oldin '
              'maʼlumotni mustaqil tekshiring va zarur boʻlsa litsenziyalangan maslahatchiga murojaat qiling.</div>')


def render_quality(st: State) -> str:
    out = [CSS, '<div class="ai">', evidence_banner(st), "<h2>Har bir ufq boʻyicha halol sinov natijalari</h2>",
           '<p class="small muted">Har bir test davri uchun model faqat undan OLDINGI maʼlumotda oʻqitilgan, oraliqdagi (ustma-ust tushgan) namunalar olib tashlangan. '
           'IC — model bali bilan keyingi daromad orasidagi reyting korrelyatsiyasi (kuniga hisoblanib oʻrtalanadi); t-statistika faqat '
           'oʻzaro kesishmaydigan kuzatuvlardan olinadi.</p>']
    rows = []
    for h, e in sorted(st.evals.items()):
        rows.append(f'<tr><td>{esc(HORIZON_UZ.get(h, str(h)))}</td><td class="n">{e.auc:.3f}</td><td class="n">{e.ic_mean:+.4f}</td>'
                    f'<td class="n">{e.ic_t:+.2f}</td><td class="n">{e.n_eff:.0f}</td><td class="n">{pc(e.hit_top, 1)}</td>'
                    f'<td class="n">{pc(e.spread, 2, True)}</td><td class="n">{e.pos_folds}/{len(e.folds)}</td>'
                    f'<td>{verdict_chip(e.verdict)}{"<span class=small> (kuzatuv kam)</span>" if e.n_eff < 8 else ""}</td></tr>')
    out.append(_table(["Ufq", "AUC", "IC", "IC t-stat", "Mustaqil kuzatuvlar", "Top-20% aniqligi", "Top−pastki 20% farqi", "Ijobiy davrlar", "Hukm"],
                      rows, [1, 2, 3, 4, 5, 6, 7], hide_mobile=[4, 6, 7]))
    out.append(f'<p class="small muted">AUC 50% — tasodif. Top-20% aniqligi: eng yuqori ball olgan 20% aksiyaning necha foizi haqiqatan oʻrtachadan yaxshi chiqdi (tasodif — 50%). '
               f'Hukm chegaralari: t ≥ {T_WEAK} zaif, t ≥ {T_STRONG} kuchli (7 ufq sinalgani uchun qatʼiyroq); mustaqil kuzatuvlar 8 tadan kam boʻlsa hukm chiqarilmaydi.</p>')
    out.append(_pic(lambda n: chart_tstat(st, n), "IC t-statistikasi ufqlar boʻyicha"))
    ev = st.evals[st.primary_h]
    out.append(f'<h2>Nima uchun oddiy "train/test" bo‘linishiga ishonib boʻlmaydi</h2>'
               f'<div class="card info"><div class="grid"><div class="kpi"><div class="lbl">Tasodifiy train/test (leakage bor)</div><div class="v">AUC {st.naive_auc:.3f}</div></div>'
               f'<div class="kpi"><div class="lbl">Purged walk-forward (halol)</div><div class="v">AUC {ev.auc:.3f}</div></div>'
               f'<div class="kpi"><div class="lbl">Sunʼiy oshish</div><div class="v">{(st.naive_auc - ev.auc) * 100:+.1f} p.p.</div></div></div>'
               f'<p class="small">Narx maʼlumotida qoʻshni kunlarning belgilari va etiketlari deyarli bir xil. Namunalarni tasodifiy boʻlsangiz, model test qatoriga '
               f'oʻxshash "qoʻshni" qatorni oʻqitishda koʻradi va natija yolgʻon yaxshi chiqadi. Shu sababli har bir sinov faqat KELAJAK davrida va oraliq tashlab oʻtilgan holda oʻtkazildi.</p></div>')
    out.append("<h2>Portfel backtesti</h2>")
    out.append(bt_summary_html(st))
    if st.bt is not None:
        out.append(_pic(lambda n: chart_backtest(st.bt, n), "Backtest"))
    if ev.importance:
        out.append(f"<h2>Qaysi belgilar guruhi modelga yordam beradi? ({esc(HORIZON_UZ.get(st.primary_h, st.primary_h))})</h2>")
        out.append('<p class="small muted">Har bir guruh test maʼlumotida har bir sana ichida aralashtirilib, AUC qanchaga tushishi oʻlchandi. '
                   'Nolga yaqin yoki manfiy — guruh foyda bermayapti (u shovqin).</p>')
        out.append(_pic(lambda n: chart_importance(ev, n), "Belgilar guruhlari ahamiyati"))
    if st.coefs is not None:
        cs = st.coefs.sort_values()
        lab = lambda k: FEATURE_UZ.get(k[2:] if k.startswith("r_") else k, k)
        neg = ", ".join(f"{lab(k)} ({v:+.2f})" for k, v in cs.head(4).items())
        posv = ", ".join(f"{lab(k)} ({v:+.2f})" for k, v in cs.tail(4)[::-1].items())
        out.append(f'<h2>Model nimalarga eʼtibor beradi</h2><div class="card"><p class="small"><b>Ball oshiradigan belgilar</b> '
                   f'(logistik regressiya qismi): {esc(posv)}</p><p class="small"><b>Ball pasaytiradigan belgilar:</b> {esc(neg)}</p>'
                   f'<p class="small muted">Eslatma: belgilar bir-biri bilan bogʻliq, shuning uchun alohida koeffitsientlarni sabab sifatida talqin qilmang.</p></div>')
    if st.calib:
        crows = [f'<tr><td>{esc(HORIZON_UZ.get(h, h))}</td><td class="n">{pc(c["cov90"], 1)}</td><td class="n">{pc(c["cov50"], 1)}</td>'
                 f'<td class="n">{pc(c["below5"], 1)}</td><td class="n">{pc(c["above95"], 1)}</td><td class="n">{c["n"]}</td></tr>'
                 for h, c in sorted(st.calib.items())]
        out.append("<h2>Narx diapazoni prognozi tarixda qanchalik toʻgʻri chiqqan?</h2>")
        out.append(_table(["Ufq", "90% diapazon qamrovi (maqsad 90%)", "50% diapazon qamrovi (maqsad 50%)", "Pastki 5% dan pastga (maqsad 5%)",
                           "Yuqori 5% dan yuqoriga (maqsad 5%)", "Kuzatuvlar"], crows, [1, 2, 3, 4, 5], hide_mobile=[5]))
        out.append('<p class="small muted">Monte-Karlo diapazoni faqat oʻtmish volatilligidan hisoblanadi, shuning uchun uni istalgan sanada tarixga qarshi tekshirish mumkin. '
                   'Qamrov maqsadga yaqin boʻlsa — diapazon yaxshi kalibrovka qilingan.</p>')
    if ev.folds:
        frows = [f'<tr><td class="n">{f["fold"]}</td><td>{f["start"]:%Y-%m-%d} → {f["end"]:%Y-%m-%d}</td><td class="n">{f["n_train"]}</td>'
                 f'<td class="n">{f["n_test"]}</td><td class="n">{f["auc"]:.3f}</td><td class="n">{f["ic"]:+.4f}</td></tr>' for f in ev.folds]
        out.append(f"<h2>Test davrlari ({esc(HORIZON_UZ.get(st.primary_h, st.primary_h))})</h2>")
        out.append(_table(["#", "Test davri", "Oʻqitish qatorlari", "Test qatorlari", "AUC", "IC"], frows, [0, 2, 3, 4, 5],
                          hide_mobile=[2, 3]))
    out.append('<h2>Cheklovlar (halol roʻyxat)</h2><ul class="small">'
               '<li><b>Survivorship bias:</b> universum — bugungi yirik kompaniyalar; oʻtmishda tushib ketganlari yoʻq.</li>'
               '<li><b>Kichik kompaniyalar:</b> model yirik aksiyalar uchun oʻqitilgan; kichik/illikvid aksiyalar (masalan, $12 lik kichik kompaniya) uchun ishonchsizroq.</li>'
               '<li><b>Yangilik va hodisalar:</b> hisobotlar (earnings), FDA qarorlari, M&A kabi voqealar modelga kirmaydi.</li>'
               '<li><b>Tasodif va koʻp sinov:</b> 7 ufq sinalgan; bitta “zaif dalil” tasodif boʻlishi mumkin.</li>'
               '<li><b>Faqat tarixiy daromad:</b> backtest xarajatlarni (10 bp) hisobga oladi, lekin slippage, soliq va likvidlikni emas.</li></ul>')
    out.append(DISCLAIMER)
    out.append("</div>")
    return "".join(out)


# --------------------------------------------------------------------------------------
# Bitta aksiya hisoboti
# --------------------------------------------------------------------------------------
def _mc_tables(R: Dict[str, Any]) -> str:
    mc = R["mc"]
    rows = []
    for r in mc["rows"]:
        q = r["q"]
        rows.append(f'<tr><td>{esc(HORIZON_UZ.get(r["h"], r["h"]))}</td><td class="n">{usd_r(q[0])}</td><td class="n">{usd_r(q[1])}</td>'
                    f'<td class="n"><b>{usd_r(q[2])}</b></td><td class="n">{usd_r(q[3])}</td><td class="n">{usd_r(q[4])}</td>'
                    f'<td class="n">{pc(r["p_up"], 0)}</td></tr>')
    t1 = _table(["Ufq", "Pastki (5%)", "25%", "Median", "75%", "Yuqori (95%)", "Yuqori boʻlish ehtimoli"], rows, [1, 2, 3, 4, 5, 6],
                hide_mobile=[2, 4])
    hs = [r["h"] for r in mc["rows"]]
    head = ["Narx harakati"] + [HORIZON_UZ.get(h, str(h)) for h in hs]
    rows2 = []
    for x in MC_TOUCH_LEVELS:
        cells = "".join(f'<td class="n">{pc(r["touch"][x], 1) if r["touch"][x] >= 0.001 else "<0.1%"}</td>' for r in mc["rows"])
        p_up = mc["P0"] * (1 + x)
        rows2.append(f'<tr><td><span class="dm">+{x * 100:.0f}% ({usd(p_up)}) ga YETISH</span>'
                     f'<span class="mo">▲ +{x * 100:.0f}% {usd(p_up, 0)}</span></td>{cells}</tr>')
    for x in MC_DROP_LEVELS:
        cells = "".join(f'<td class="n">{pc(r["drop"][x], 1) if r["drop"][x] >= 0.001 else "<0.1%"}</td>' for r in mc["rows"])
        p_dn = mc["P0"] * (1 - x)
        rows2.append(f'<tr class="dim"><td><span class="dm">−{x * 100:.0f}% ({usd(p_dn)}) ga TUSHISH</span>'
                     f'<span class="mo">▼ −{x * 100:.0f}% {usd(p_dn, 0)}</span></td>{cells}</tr>')
    # telefonda faqat 1 hafta / 1 oy / 3 oy / 6 oy ustunlari (5, 20, 60, 125 kun) qoladi
    hide2 = [j + 1 for j, h in enumerate(hs) if h not in (5, 20, 60, 125)]
    t2 = _table(head, rows2, list(range(1, len(hs) + 1)), hide_mobile=hide2, wrap_first=True)
    return t1, t2


def render_analysis_body(R: Dict[str, Any], st: State) -> str:
    t, df, last, price = R["ticker"], R["df"], R["last"], R["price"]
    con = R["conclusion"]
    fund = R["fund"]
    name = fund.get("shortName") if fund.get("ok") else None
    chips = "".join(f'<span class="tag">{esc(k)}: {pc(v, 1, True)}</span>' for k, v in R["chg"].items())
    lab = con["label"]
    arrow = "▲" if lab == "Ijobiy" else "▼" if lab == "Salbiy" else "●"
    cls = "good" if lab == "Ijobiy" else "bad" if lab == "Salbiy" else "warn"
    out = [f'<div class="card info"><div class="row"><div><div class="lbl">{esc(t)}{" · " + esc(name) if name else ""}</div>'
           f'<div class="big">{usd(price)}</div><div class="lbl">{R["asof"]:%Y-%m-%d} dagi yopilish narxi</div><div class="chips">{chips}</div></div>'
           f'<div class="card {cls}" style="margin:0;min-width:230px"><div class="lbl">Umumiy xulosa</div><div class="big">{arrow} {esc(lab)}</div>'
           f'<div class="lbl">ball {con["score"]:+.2f} · ishonch: {esc(con["confidence"])}</div></div></div></div>']
    if not R["in_universe"]:
        out.append('<div class="card warn small">Bu aksiya AI universumiga kirmaydi: model unga ham qoʻllanadi, lekin u asosan yirik, likvid aksiyalarda '
                   'oʻqitilgan — kichik yoki juda volatil kompaniyalar uchun natija ishonchsizroq.</div>')
    # --- xulosa: omillar roʻyxati (kompyuterda 4 ustun, telefonda 2 qatorli)
    def sig_txt(sv: float) -> str:
        return f"▲ {sv:+.1f}" if sv > 0.15 else f"▼ {sv:+.1f}" if sv < -0.15 else f"● {sv:+.1f}"
    vrows = ['<div class="vote vh"><div>Omil</div><div>Signal</div><div>Ogʻirlik</div><div>Izoh</div></div>']
    for v in con["votes"]:
        cls_r = "vote" if v["counted"] else "vote dim"
        sg = sig_txt(v["sig"]) if v["counted"] else "—"
        vrows.append(f'<div class="{cls_r}"><div class="vn">{esc(v["name"])}</div>'
                     f'<div class="vs">{sg}<span class="mo"> ×{v["w"]:.1f}</span></div>'
                     f'<div class="vw">{v["w"]:.1f}</div><div class="vt">{esc(v["text"])}</div></div>')
    out.append("<h2>Umumiy xulosa nimaga asoslangan</h2>")
    out.append('<div class="votes">' + "".join(vrows) + "</div>")
    out.append('<p class="small muted">Xulosa = ogʻirlikli oʻrtacha signal (−1…+1). Faqat AI bloki out-of-sample sinovdan oʻtgan; qolgan qoidalar '
               '(trend, momentum, Wyckoff, RSI, sham, Gann, fundamental, yangilik) klassik qoidalar boʻlib, ularning bashorat kuchi bu yerda tekshirilmagan — '
               'shuning uchun ularga kichik ogʻirlik berilgan.</p>')
    # --- grafik
    out.append("<h2>Narx grafigi, support/resistance, Gann yelpigʻichi va Wyckoff hodisalari</h2>")
    out.append(_pic(lambda n: chart_price(R, n), f"{t} narx grafigi"))
    # --- AI
    out.append("<h2>AI prognozi (mashinaviy oʻqitish)</h2>")
    ai = R["ai"]
    if ai is None:
        out.append('<div class="card warn small">Bu aksiya uchun AI ball hisoblanmadi (tarix yetarli emas yoki hajm/narx maʼlumoti toʻliq emas).</div>')
    else:
        rows = []
        for H in sorted(ai["probs"]):
            e = st.evals.get(H)
            rows.append(f'<tr><td>{esc(HORIZON_UZ.get(H, H))}</td>{prob_cell(ai["probs"][H])}<td class="n">{ai["pct"][H] * 100:.0f}%</td>'
                        f'<td>{verdict_chip(e.verdict) if e else ""}</td><td class="n">{e.ic_t:+.2f}</td></tr>')
        out.append(_table(["Ufq", "Oʻrtacha aksiyadan yaxshiroq boʻlish ehtimoli", "Universumdagi persentil", "Modelning tarixiy ishonchliligi", "IC t"], rows, [1, 2, 4],
                          hide_mobile=[2, 4]))
        out.append('<p class="small muted">50% — neytral. Bu ehtimol “narx koʻtariladi” degani emas: bozor yoʻnalishi tozalangan holda, aksiyaning oʻrtacha '
                   'aksiyadan yaxshiroq (yoki yomonroq) chiqish ehtimoli. Ishonchlilik ustuniga qarang: “Dalil yoʻq” boʻlgan ufqdagi ehtimolga tayanmang.</p>')
        if ai["ood"]:
            out.append(f'<div class="card warn small">Diqqat: {esc(", ".join(ai["ood"][:6]))} belgilari universum diapazonidan tashqarida — '
                       f'model bunday aksiyalar uchun oʻqitilmagan, natija ishonchsizroq.</div>')
    # --- Monte-Karlo
    mc = R["mc"]
    t1, t2 = _mc_tables(R)
    out.append("<h2>Narx qaysi diapazonda boʻlishi mumkin (Monte-Karlo)</h2>")
    out.append(f'<p class="small muted">Kunlik volatillik {mc["sigma_d"] * 100:.2f}% (yillik {mc["sigma_ann"] * 100:.0f}%). '
               f'Bu AI emas — statistik “volatillik konusi”: drift bozor-prior ({MC_DRIFT * 252 * 100:.0f}% yillik), aksiyaning oʻz oʻtmish daromadi emas; '
               f'zarbalar ogʻir dumli (t-taqsimot).</p>')
    out.append(_pic(lambda n: chart_cone(R, n), "Monte-Karlo narx konusi"))
    out.append(t1)
    cal = st.calib
    if cal:
        bits = [f'{HORIZON_UZ.get(h, h)}: 90% diapazon {c["cov90"] * 100:.0f}% hollarda qamragan' for h, c in sorted(cal.items()) if h in (20, 60)]
        if bits:
            out.append(f'<p class="small muted">Tarixiy tekshiruv (butun universum): {esc("; ".join(bits))}. Qamrov 90% ga yaqin boʻlsa, diapazon yaxshi kalibrovka qilingan.</p>')
    out.append("<h3>Narx hozirgidan qancha yuqoriga/pastga borishi ehtimoli (davr ichida kamida bir marta tegishi)</h3>")
    out.append(t2)
    tg = mc.get("target")
    if tg:
        trows = [f'<tr><td>{esc(HORIZON_UZ.get(h, h))}</td><td class="n"><b>{pc(tg["touch"][h], 1) if tg["touch"][h] >= 0.001 else "<0.1%"}</b></td>'
                 f'<td class="n">{pc(tg["end"][h], 1) if tg["end"][h] >= 0.001 else "<0.1%"}</td></tr>' for h in sorted(tg["touch"])]
        out.append(f'<h3>Maqsad narx: {usd(tg["price"])} ({pc(tg["pct"], 1, True)})</h3>')
        out.append(_table(["Ufq", "Davr ichida kamida bir marta yetish ehtimoli",
                           "Ufq oxirida shu darajada yoki undan " + ("yuqorida" if tg["up"] else "pastda") + " boʻlish ehtimoli"], trows, [1, 2]))
    # --- darajalar
    lv = R["levels"]
    out.append("<h2>Support / Resistance darajalari</h2>")
    lvl_touch = {x["name"]: x["touch"] for x in mc.get("levels", [])}
    rows = []
    for tag, items in (("R", lv["resistance"]), ("S", lv["support"])):
        for i, it in enumerate(items):
            nm = f"{tag}{i + 1}"
            tp = lvl_touch.get(nm, {}).get(60)
            rows.append(f'<tr><td>{"Qarshilik" if tag == "R" else "Qoʻllab-quvvatlash"} {nm}</td><td class="n">{usd_r(it["price"])}</td>'
                        f'<td class="n">{pc(it["dist"], 1, True)}</td><td class="n">{it["touches"]}</td><td class="n">{it["last"]:%Y-%m-%d}</td>'
                        f'<td class="n">{pc(tp, 0) if tp is not None else "—"}</td></tr>')
    if rows:
        out.append(_table(["Daraja", "Narx", "Hozirgi narxdan", "Teginishlar", "Oxirgi teginish", "3 oyda yetish ehtimoli (MC)"], rows, [1, 2, 3, 4, 5],
                          hide_mobile=[3, 4], wrap_first=True))
    sc = []
    r1 = lv["resistance"][0] if lv["resistance"] else None
    r2 = lv["resistance"][1] if len(lv["resistance"]) > 1 else None
    s1 = lv["support"][0] if lv["support"] else None
    s2 = lv["support"][1] if len(lv["support"]) > 1 else None
    if r1:
        sc.append(f'<li><b>Yuqoriga ssenariy:</b> narx {usd(r1["price"])} dan yuqorida yopilsa, '
                  + (f'keyingi qarshilik {usd(r2["price"])}; ' if r2 else "yuqorida yaqin tarixiy qarshilik yoʻq; ")
                  + (f'ssenariy {usd(s1["price"])} dan pastga yopilganda bekor boʻladi.' if s1 else "") + "</li>")
    else:
        sc.append(f'<li>Narx 52 haftalik yuqori ({usd(lv["hi52"])}) atrofida: yuqorida tarixiy qarshilik yoʻq (narx “kashf etilmagan” hududda).</li>')
    if s1:
        sc.append(f'<li><b>Pastga ssenariy:</b> narx {usd(s1["price"])} dan pastga yopilsa, ' + (f'keyingi qoʻllab-quvvatlash {usd(s2["price"])}.' if s2 else "past tomonda yaqin tarixiy tayanch yoʻq.") + "</li>")
    out.append(f'<ul class="small">{"".join(sc)}</ul>')
    out.append(f'<p class="small muted">52 haftalik diapazon: {usd(lv["lo52"])} – {usd(lv["hi52"])}. Daraja kengligi (ATR asosida) ≈ {usd(lv["tol"])}. '
               f'Darajalar oʻtmish pivot nuqtalaridan olingan; ular “ish beradi” degan kafolat yoʻq.</p>')
    # --- Wyckoff
    ev = R["wy_events"]
    out.append("<h2>Wyckoff tahlili</h2>")
    out.append(f'<p><b>Joriy faza:</b> {esc(WYCKOFF_UZ.get(R["wy_code"], "Aniqlanmadi"))}.</p>')
    bs0 = R.get("base") or {}
    if R["wy_code"] in (1, 3) and bs0.get("ok") and bs0.get("state") == "inside":
        out.append('<p class="small muted">Eslatma: narx baza (konsolidatsiya) ichida. Trend boʻlmaganda Accumulation/Distribution farqi diapazon yarmiga qarab '
                   'MEXANIK belgilanadi — haqiqiy yoʻnalishni breakout (yuqoriga) yoki breakdown (pastga) hamda Spring/Upthrust hodisalari koʻrsatadi.</p>')
    wl = []
    if not ev.get("range_ok"):
        wl.append("Soʻnggi ~70 kunda aniq savdo diapazoni yoʻq (narx trendda yoki juda keng) — Spring/Upthrust qidirilmadi.")
    else:
        wl.append(f'Savdo diapazoni: {usd(ev["support"])} – {usd(ev["resistance"])}.')
        sp, ts, up_ = ev.get("spring"), ev.get("test"), ev.get("upthrust")
        if sp:
            wl.append(f'<b>Spring:</b> {sp["date"]:%Y-%m-%d} kuni narx diapazon pastki chegarasidan {sp["pierce"] * 100:.1f}% pastga tushib '
                      f'({usd(sp["low"])}), {usd(sp["close"])} da yopildi (hajm oʻrtachaning {sp["vol_ratio"]:.1f}×) — yolgʻon pasayish.')
            if ts:
                wl.append(f'<b>Test:</b> {ts["date"]:%Y-%m-%d} kuni past hajmda ({ts["vol_ratio"]:.1f}×) qayta sinab koʻrildi va ushlab turdi — Spring tasdiqlandi.')
            else:
                wl.append(f'Test hali yoʻq: Spring pastidan ({usd(sp["low"])}) yuqorida, past hajmli qayta sinov kutiladi. '
                          f'{usd(sp["low"])} dan pastga yopilish — signalni bekor qiladi.')
        if up_:
            wl.append(f'<b>Upthrust:</b> {up_["date"]:%Y-%m-%d} kuni narx diapazon yuqori chegarasidan {up_["pierce"] * 100:.1f}% yuqoriga chiqib '
                      f'({usd(up_["high"])}), {usd(up_["close"])} da yopildi — yolgʻon yuqoriga chiqish (taqsimlanish xavfi).')
        if not sp and not up_:
            wl.append("Soʻnggi 10 kunda Spring yoki Upthrust hodisasi aniqlanmadi.")
    out.append(f'<ul class="small">{"".join("<li>" + x + "</li>" for x in wl)}</ul>')
    out.append('<p class="small muted">Wyckoff fazasi va hodisalari qoidalarga asoslangan mexanik talqin (tajribali trader koʻzi bilan emas); '
               'faza belgisi AI modelida ham ishlatiladi (uning foydasini “Model sifati” boʻlimidan koʻring).</p>')
    # --- baza va breakout
    bs = R.get("base") or {}
    out.append("<h2>Baza (konsolidatsiya) va breakout: maqsadlar</h2>")
    if not bs.get("ok"):
        out.append('<p class="muted">Soʻnggi ~1.3 yilda aniq baza (tor diapazonda uzoq turish, chegaralari kamida 2 martadan sinalgan) topilmadi: '
                   'narx yoki trendda boʻlgan, yoki diapazon juda keng/notekis.</p>')
    else:
        head = (f'Baza: <b>{bs["start"]:%Y-%m-%d} – {bs["end"]:%Y-%m-%d}</b> ({bs["W"]} savdo kuni), diapazon <b>{usd(bs["lo"])} – {usd(bs["hi"])}</b> '
                f'(chuqurligi {bs["depth"] * 100:.0f}%; yuqori chegara {bs["top_t"]} marta, past chegara {bs["bot_t"]} marta sinalgan).')
        stt = bs["state"]
        if stt == "breakout":
            vtxt = (f"breakout kuni hajm oʻrtachaning {bs['vol_ratio']:.1f}× — <b>hajm bilan tasdiqlangan</b>" if bs["confirmed"]
                    else f"breakout kuni hajm oʻrtachaning atigi {bs['vol_ratio']:.1f}× — <b>hajm tasdigʻi zaif</b>")
            rtxt = (" Narx breakout darajasini qayta sinab, uni ushlab turdi (retest)." if bs["retest"] else " Qayta sinov (retest) hali boʻlmagan.")
            msg = (f'{head} Narx {bs["broke"]:%Y-%m-%d} kuni bazadan yuqoriga chiqqan ({bs["days_since"]} savdo kuni oldin) va hozir bazadan '
                   f'<b>{bs["gain"] * 100:+.1f}%</b> yuqorida; {vtxt}.{rtxt} Breakout bekor boʻladi: narx {usd(bs["hi"])} dan pastga qaytib yopilsa.')
        elif stt == "inside":
            msg = (f'{head} Narx hozir bazaning ichida (diapazonning {bs["pos"] * 100:.0f}% qismida): breakout hali yoʻq. '
                   f'Breakout sharti — {usd(bs["trigger"])} dan yuqorida yopilish (ideal holda hajm oshishi bilan); past chegara {usd(bs["lo"])} dan pastga yopilish — baza buziladi.')
        elif stt == "failed":
            msg = f'{head} Narx bazadan yuqoriga chiqishga urindi, lekin qaytib diapazon ichiga tushdi: <b>breakout muvaffaqiyatsiz</b> (ehtiyotkorlik).'
        else:
            msg = f'{head} Narx bazaning past chegarasidan pastga chiqdi: <b>baza buzildi</b> (pasayish signali).'
        out.append(f"<p>{msg}</p>")
        if stt in ("inside", "breakout"):
            lt_ = {x["name"]: x["touch"] for x in R["mc"].get("levels", [])}
            rows = []
            for tg_ in bs["targets"]:
                pr_ = lt_.get(tg_["name"], {})
                cells = "".join(f'<td class="n">{pc(pr_[h], 0) if h in pr_ else "—"}</td>' for h in (20, 60, 125))
                note = ("allaqachon yetib borilgan" + (" (hozir narx undan pastda)" if tg_["price"] > price else "") if tg_["reached"] else
                        f'baza balandligining {tg_["frac"] * 100:.0f}% i breakout darajasidan yuqoriga' + (" (kengaytma)" if tg_["frac"] > 1 else ""))
                if tg_["price"] <= price:
                    cells = '<td class="n">—</td>' * 3
                rows.append(f'<tr><td>Maqsad {tg_["name"][1]}</td><td class="n">{usd_r(tg_["price"])}</td><td class="n">{pc(tg_["price"] / price - 1, 1, True)}</td>'
                            f'<td>{esc(note)}</td>{cells}</tr>')
            out.append(_table(["Maqsad", "Narx", "Hozirgidan", "Qoida", "1 oyda yetish (MC)", "3 oyda (MC)", "6 oyda (MC)"], rows, [1, 2, 4, 5, 6],
                              hide_mobile=[3], wrap_first=True))
        out.append('<p class="small muted">Maqsadlar — klassik “measured move” qoidasi (bazaning balandligi breakout darajasiga qoʻshiladi). Bu bashorat EMAS: '
                   'bu qoidaning tarixiy aniqligi bu yerda tekshirilmagan; “Monte-Karlo” ustunlari esa faqat volatillikdan hisoblangan statistik ehtimol. '
                   'Baza va breakout mexanik qoidalar bilan topiladi (tajribali trader koʻzi bilan farq qilishi mumkin).</p>')
    # --- sham
    out.append("<h2>Sham (candlestick) modellari — oxirgi 5 sham</h2>")
    rows = []
    for x in R["candles"]:
        sub = f'<div class="small muted">{esc(x["text"])}</div>' if x["name"] and x["text"] else ""
        rows.append(f'<tr><td>{x["date"]:%Y-%m-%d}</td><td><b>{esc(x["name"]) or "Aniq model yoʻq"}</b>{sub}</td></tr>')
    out.append(_table(["Sana", "Model va maʼnosi"], rows))
    # --- Gann
    gf = R["gann"]
    out.append("<h2>Gann yelpigʻichi</h2>")
    if gf.get("ok"):
        if gf["slope"] >= 1.0:
            gtxt = "Narx 1x1 chizigʻidan yuqorida — kuchli oʻsish trendi."
        elif gf["slope"] >= 0.5:
            gtxt = "Narx 1x2 va 1x1 chiziqlari orasida — oʻrtacha trend."
        else:
            gtxt = "Narx 1x2 chizigʻidan pastda — trend zaif."
        out.append(f'<p>Tayanch (soʻnggi 6 oydagi eng past nuqta): {gf["anchor_date"]:%Y-%m-%d}, {usd(gf["anchor_low"])}; undan beri {gf["bars"]} bar. {gtxt}</p>')
        rows, placed = [], False
        for ln in sorted(gf["lines"], key=lambda x: -x["now"]):
            if not placed and ln["now"] <= gf["price"]:
                rows.append(f'<tr><td><b>NARX</b></td><td class="n"><b>{usd(gf["price"])}</b></td><td class="n">—</td></tr>')
                placed = True
            rows.append(f'<tr><td>{ln["name"]}</td><td class="n">{usd_r(ln["now"])}</td><td class="n">{usd_r(ln["proj"])}</td></tr>')
        if not placed:
            rows.append(f'<tr><td><b>NARX</b></td><td class="n"><b>{usd(gf["price"])}</b></td><td class="n">—</td></tr>')
        out.append(_table(["Chiziq", "Hozirgi daraja", f"{gf['ahead']} bardan keyin"], rows, [1, 2]))
        out.append('<p class="small muted">Gann masshtabi (1x1 = har bar uchun 0.25 ATR) ixtiyoriy tanlov: Gann uni qatʼiy belgilamagan. '
                   'Gann yelpigʻichining bashorat kuchi isbotlanmagan; bu yerda u tahlil kontekstini toʻldirish uchun va AI uchun belgi sifatida ishlatiladi.</p>')
    else:
        out.append('<p class="muted">Gann chiziqlari uchun yetarli tarix yoʻq.</p>')
    # --- fundamental
    out.append("<h2>Fundamental koʻrsatkichlar</h2>")
    if fund.get("ok"):
        def g(k, f="{:.2f}", mult=1.0, suffix=""):
            v = fund.get(k)
            return "—" if not isinstance(v, (int, float)) or isinstance(v, bool) else f.format(v * mult) + suffix
        mcap = fund.get("marketCap")
        mc_txt = "—" if not mcap else (f"${mcap / 1e9:,.1f} mlrd" if mcap >= 1e9 else f"${mcap / 1e6:,.0f} mln")
        kp = [("Bozor qiymati", mc_txt), ("P/E (joriy / kelgusi)", f'{g("trailingPE", "{:.1f}")} / {g("forwardPE", "{:.1f}")}'),
              ("P/B", g("priceToBook", "{:.1f}")), ("Sof foyda marjasi", g("profitMargins", "{:.0f}", 100, "%")),
              ("ROE", g("returnOnEquity", "{:.0f}", 100, "%")), ("Daromad oʻsishi", g("revenueGrowth", "{:+.0f}", 100, "%")),
              ("Foyda oʻsishi", g("earningsGrowth", "{:+.0f}", 100, "%")), ("Qarz/kapital", g("debtToEquity", "{:.0f}", 1, "%")),
              ("Beta", g("beta", "{:.2f}")), ("Dividend daromadi", (f"{fund['dividendRate'] / price * 100:.2f}%" if isinstance(fund.get("dividendRate"), (int, float)) and price else g("dividendYield", "{:.2f}", 1, "%"))),
              ("Analitiklar maqsadi (oʻrtacha)", usd(fund.get("targetMeanPrice")) if fund.get("targetMeanPrice") else "—"),
              ("Analitiklar tavsiyasi", esc(str(fund.get("recommendationKey") or "—")))]
        out.append(f'<p class="muted small">{esc(str(fund.get("sector") or ""))} · {esc(str(fund.get("industry") or ""))}</p>')
        out.append('<div class="grid">' + "".join(f'<div class="kpi"><div class="lbl">{esc(a)}</div><div class="v" style="font-size:1.08rem">{b}</div></div>' for a, b in kp) + "</div>")
        fs, why = R["fund_score"]
        if why:
            out.append(f'<ul class="small">{"".join("<li>" + esc(w) + "</li>" for w in why)}</ul>')
    else:
        out.append(f'<p class="muted">Fundamental maʼlumot olinmadi ({esc(str(fund.get("error", "")))}). Texnik va AI tahlil bunga bogʻliq emas.</p>')
    # --- yangiliklar
    out.append("<h2>Soʻnggi yangiliklar va kayfiyat (VADER)</h2>")
    nw = R["news"]
    if nw.get("ok") and nw["items"]:
        rows = []
        for it in nw["items"]:
            s = it["sent"]
            tag = f'<span class="tag good">▲ ijobiy {s:+.2f}</span>' if s >= 0.25 else f'<span class="tag bad">▼ salbiy {s:+.2f}</span>' if s <= -0.25 else f'<span class="tag">● neytral {s:+.2f}</span>'
            title = esc(it["title"])
            if it["url"].startswith("http"):
                title = f'<a href="{esc(it["url"])}" target="_blank" rel="noopener noreferrer">{title}</a>'
            dtx = f'{it["date"]:%Y-%m-%d}' if it["date"] is not None and pd.notna(it["date"]) else ""
            rows.append(f'<tr><td class="n">{dtx}</td><td>{title}<div class="small muted">{esc(it["publisher"])}<span class="mo"> · {dtx}</span></div></td><td>{tag}</td></tr>')
        out.append(_table(["Sana", "Sarlavha", "Kayfiyat"], rows, hide_mobile=[0]))
        out.append(f'<p class="small muted">Oʻrtacha kayfiyat: {nw["avg"]:+.2f}. Sarlavha kayfiyati inglizcha qoidaga asoslangan VADER bilan hisoblanadi — '
                   f'u kontekstni (masalan, kinoya yoki kutilganidan yaxshi/yomon natija) tushunmaydi; bashorat kuchi tekshirilmagan.</p>')
    else:
        out.append(f'<p class="muted">Yangiliklar olinmadi{(" (" + esc(str(nw.get("error"))) + ")") if nw.get("error") else ""}.</p>')
    return "".join(out)


def render_analysis(R: Dict[str, Any], st: State) -> str:
    return CSS + '<div class="ai">' + render_analysis_body(R, st) + DISCLAIMER + "</div>"


def standalone_html(body: str, title: str) -> str:
    return ('<!doctype html><html lang="uz"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)}</title><style>body{{margin:0;padding:16px;background:#fff;color:#0b0b0b}}</style></head><body>{body}</body></html>')


# ======================================================================================
# 9. GRADIO ILOVASI (telefon va kompyuter brauzerida ishlaydi)
# ======================================================================================
METHODOLOGY_MD = """
## Bu tizim qanday ishlaydi

**1. Maʼlumot.** ~170 ta yirik AQSh aksiyasi (NYSE/NASDAQ) va SPY indeksining 8 yillik kunlik narx tarixi (Yahoo Finance; zaxira — Stooq).
Tizim har ~12 soatda maʼlumotni yangilab, modelni qaytadan oʻqitadi va tekshiradi.

**2. Belgilar (features).** Har bir aksiya va har bir sana uchun faqat OʻTMISH maʼlumotidan hisoblanadi (kelajakka qarash dasturiy testdan oʻtgan):
momentum (1 kundan 12 oygacha), volatillik va beta, SMA masofalari (10/30/50/200), 60 kunlik va 52 haftalik diapazondagi oʻrni, RSI, hajm oʻzgarishi,
**Wyckoff fazasi** (accumulation / markup / distribution / markdown), **sham modeli** (hammer, engulfing, marubozu, doji ...) va **Gann burchagi**
(narx soʻnggi 6 oylik eng past nuqtadan Gann 1×1 chizigʻiga nisbatan qanchalik tez oʻsgan). Barcha belgilar har bir sana ichida aksiyalar orasida
reytingga (0–1) aylantiriladi — shunda bozor rejimi oʻzgarsa ham belgilar solishtirma boʻlib qoladi.

**3. Nimani bashorat qiladi.** “Keyingi *H* savdo kunida bu aksiya oʻrtacha aksiyadan yaxshiroq chiqadimi?” (H = 5, 10, 20, 40, 60, 90, 125 kun).
Daromad bozor yoʻnalishidan (beta orqali) tozalanadi, shuning uchun model “bozor oʻssa yuqori-beta aksiya yutadi” kabi tasodifiy foydadan dalil yasay olmaydi.
Tasodifiy taxmin = 50%.

**4. Model.** Ikki modelning oʻrtachasi: juda qatʼiy regullarizatsiyalangan sayoz **gradient boosting** va **logistik regressiya**.
Moliyaviy maʼlumotda signal/shovqin nisbati juda past, shuning uchun murakkab modellar odatda shovqinni yodlab oladi (sintetik sinovda aynan shu kuzatildi).
Giperparametrlar natijaga qarab “sozlanmagan”.

**5. Halol sinov (purged walk-forward).** Test davrlari vaqt boʻyicha ketma-ket: har bir davr uchun model faqat undan OLDINGI maʼlumotda oʻqitiladi.
Etiketi test davriga tutashadigan oʻqitish namunalari olib tashlanadi (*embargo*). Natijalar:
**AUC**, **IC** (model bali bilan keyingi daromad orasidagi reyting korrelyatsiyasi), **IC t-statistikasi** (faqat oʻzaro kesishmaydigan kuzatuvlardan),
eng yuqori 20% aniqligi va **backtest** (AI top-5 / momentum top-5 / universum / SPY, 10 bp xarajat bilan).

**6. Hukm qoidasi.** IC t ≥ 2.7 va barqaror → *kuchli dalil*; t ≥ 2.0 → *zaif dalil*; aks holda → *dalil yoʻq*. 7 ufq sinalgani uchun chegara qatʼiy.
Sintetik sinovda (ustunlik yoʻq) ~32 sinovdan 0 tasi “dalil” deb chiqdi; ekilgan kuchli signal esa 100% topildi.
Dalil yoʻq boʻlsa, tizim buni roʻyxat tepasida ochiq yozadi va AI bali umumiy xulosaga QOʻSHILMAYDI.

**7. Narx diapazoni (Monte-Karlo).** Volatillik (EWMA va 1 yillik aralashmasi) va ogʻir dumli (t) zarbalar bilan 20 000 ta yoʻl; drift — bozor-prior,
aksiyaning oʻz oʻtmish daromadi emas. Bu AI emas, statistik “volatillik konusi”; uning kalibrovkasi tarixda tekshiriladi (“Model sifati” boʻlimi).

**8. Klassik tahlil bloklari.** Support/resistance (pivotlar klasteri), Wyckoff Spring/Test/Upthrust, Gann yelpigʻichi, sham modellari, fundamental
koʻrsatkichlar (Yahoo), yangilik kayfiyati (VADER). Ular kontekst beradi, lekin ularning bashorat kuchi bu yerda **tekshirilmagan** — faqat AI bloki sinovdan oʻtadi.

**9. Baza va breakout.** Soʻnggi ~1.3 yildagi eng yangi **tor savdo diapazoni (“baza”)** mexanik qoidalar bilan topiladi: chuqurligi 4–35%, ichida trend yoʻq,
yuqori va past chegarasi kamida 3 martadan sinalgan. Narx undan yuqoriga chiqsa — breakout (hajm tasdigʻi va qayta sinov bilan); maqsadlar — baza balandligining
50%, 100% va 161.8% ini breakout darajasiga qoʻshish (klassik “measured move”), ularga yetish ehtimoli esa Monte-Karlo bilan hisoblanadi. **Halol eslatma:** signalsiz
sintetik tasodifiy bozorda ham bu qoida oynalarning ~49% ida “baza”, ~5% ida “hajm bilan tasdiqlangan breakout” topadi — shuning uchun baza topilishi oʻz-oʻzidan
signal emas; qoidaning real bozordagi bashorat kuchi bu yerda **tekshirilmagan** va umumiy xulosaga faqat kichik ogʻirlik (0.5) bilan kiradi.

## Cheklovlar (halol)

* **Mukammal AI yoʻq.** Bozor raqobatli: oddiy, ochiq maʼlumotdagi ustunliklar odatda juda kichik yoki yoʻq.
* **Survivorship bias:** universum — bugungi yirik kompaniyalar; tushib ketganlari yoʻq. Backtest natijalari odatda yuqoriroq koʻrinadi.
* **Kichik va illikvid aksiyalar** (masalan, $12 lik kichik kompaniya) uchun model oʻqitilmagan — ishonchsizroq.
* **Real maʼlumotda tekshiruv:** metodologiya (kelajakka qarash yoʻqligi, halol validatsiya, baza/Spring aniqlash mexanikasi) sintetik maʼlumotda dasturiy testdan oʻtgan; haqiqiy bozor natijalari esa har kuni “Model sifati” boʻlimida yangilanadi.
* **Voqealar:** hisobotlar, FDA qarorlari, qoʻshilishlar kabi voqealar modelga kirmaydi.
* **Insider (Form 4) maʼlumotlari** bu versiyada yoʻq (barqaror bepul API yoʻq).
* Bu dastur taʼlim/tadqiqot uchun — **moliyaviy maslahat emas**.
"""

# Telefonda 4 ta bo'lim tugmasi bitta qatorga sig'ishi uchun (Gradio ortiqcha tugmalarni "…" menyusiga yashiradi)
APP_CSS = ("@media (max-width:640px){.tab-container button{padding:6px 6px !important;font-size:.84rem !important}"
           ".main.fillable{padding-left:8px !important;padding-right:8px !important}"
           ".html-container{padding-left:2px !important;padding-right:2px !important}}")
TAB_CSS = f"<style>{APP_CSS}</style>"

HEADER_HTML = (CSS + TAB_CSS + '<div class="ai"><div class="card info"><div class="row"><div>'
               '<div class="big" style="font-size:1.6rem">Aksiya AI</div>'
               '<div class="muted small">AQSh aksiyalari: mashinaviy oʻqitish, halol tarixiy sinov va chuqur texnik/fundamental tahlil</div></div>'
               f'<div class="muted small">v{APP_VERSION} · maʼlumot: Yahoo Finance · moliyaviy maslahat emas</div></div></div></div>')


def _build_cache(st: State) -> None:
    try:
        st.cache["reco"] = render_reco(st)
    except Exception as e:  # pragma: no cover
        log("render_reco xatosi:", e)
        st.cache["reco"] = error_html(f"Tavsiyalar sahifasini tayyorlashda xatolik: {e}")
    try:
        st.cache["quality"] = render_quality(st)
    except Exception as e:  # pragma: no cover
        log("render_quality xatosi:", e)
        st.cache["quality"] = error_html(f"Model sifati sahifasini tayyorlashda xatolik: {e}")


if _build_cache not in _ON_BUILT:
    _ON_BUILT.append(_build_cache)


def _loading_html(msg: str) -> str:
    return CSS + f'<div class="ai"><div class="card info"><b>⏳ {esc(msg)}</b><p class="muted small">Tizim ilk ishga tushganda ~2–4 daqiqa oladi ' \
                 f'(maʼlumot yuklash va modelni halol sinash). Keyin hamma narsa darhol ishlaydi.</p></div></div>'


def _save_report(R: Dict[str, Any], body_html: str) -> Optional[str]:
    try:
        d = os.path.join(tempfile.gettempdir(), "aksiya_reports")
        os.makedirs(d, exist_ok=True)
        fp = os.path.join(d, f"aksiya_{R['ticker']}_{dt.date.today().isoformat()}.html")
        with open(fp, "w", encoding="utf-8") as f:
            f.write(standalone_html(body_html, f"{R['ticker']} tahlili"))
        return fp
    except Exception:
        return None


def _parse_price(x) -> Optional[float]:
    """Maqsad narx matnini songa aylantiradi ("250", "250.5", "$1,250.50", "250,5")."""
    if x is None:
        return None
    t = str(x).strip().replace("$", "").replace(" ", "")
    if not t:
        return None
    if "," in t and "." in t:
        t = t.replace(",", "")
    else:
        t = t.replace(",", ".")
    try:
        v = float(t)
    except ValueError:
        return None
    return v if np.isfinite(v) and v > 0 else None


_HEAVY = threading.BoundedSemaphore(3)      # bir vaqtda koʻpi bilan 3 ta tahlil (xotira va protsessorni asrash uchun)


def build_demo():
    import gradio as gr

    def ui_load(progress=gr.Progress()):
        try:
            st = wait_ready(progress)
        except DataError as e:
            msg = error_html(str(e), "“Yangilash / qayta urinish” tugmasini bosing yoki bir necha daqiqadan keyin sahifani yangilang.")
            return msg, msg
        if "reco" not in st.cache:
            _build_cache(st)
        return st.cache["reco"], st.cache["quality"]

    def ui_retry(progress=gr.Progress()):
        retry_now()
        time.sleep(1.0)
        return ui_load(progress)

    def ui_analyze(ticker, target, progress=gr.Progress()):
        hide = gr.DownloadButton(visible=False)
        try:
            t = normalize_ticker(ticker)
        except DataError as e:
            return error_html(str(e)), hide
        try:
            st = wait_ready(progress)
            tgt = _parse_price(target)
            warn = ""
            if tgt is None and str(target or "").strip():
                warn = '<div class="ai"><div class="card warn small">Maqsad narx tushunarsiz yozilgan — hisobga olinmadi (masalan: 250 yoki 250.5).</div></div>'
            progress(0.35, desc=f"{t} tahlil qilinmoqda...")
            with _HEAVY:
                R = analyze_ticker(t, st, tgt)
                progress(0.8, desc="Hisobot tayyorlanmoqda...")
                body = render_analysis(R, st)
                fp = _save_report(R, body)
            trim_memory()
            return warn + body, (gr.DownloadButton(value=fp, visible=True) if fp else hide)
        except DataError as e:
            return error_html(str(e), "Tikerni tekshiring (masalan AAPL, NVDA, BRK-B) va internet aloqasini koʻring."), hide
        except Exception as e:
            import traceback
            traceback.print_exc()
            return error_html(f"Kutilmagan xatolik: {type(e).__name__}: {e}"), hide

    with gr.Blocks(title="Aksiya AI") as demo:
        gr.HTML(HEADER_HTML)
        with gr.Tabs():
            with gr.Tab("🏆 Tavsiyalar"):
                reco_out = gr.HTML(value=_loading_html("Tizim tayyorlanmoqda..."))
                retry_btn = gr.Button("🔄 Yangilash / qayta urinish", size="sm")
            with gr.Tab("🔍 Aksiya"):
                with gr.Row():
                    ticker = gr.Textbox(label="Aksiya tikeri (masalan: AAPL, NVDA, TSLA, BRK-B)", placeholder="AAPL", scale=3,
                                        max_lines=1, min_width=200)
                    target = gr.Textbox(label="Maqsad narx, $ (ixtiyoriy)", placeholder="masalan 250", scale=2,
                                        max_lines=1, min_width=140)
                    go = gr.Button("Tahlil qilish", variant="primary", scale=1, min_width=140)
                with gr.Row():
                    quick = [gr.Button(tk, size="sm", min_width=64) for tk in ("AAPL", "NVDA", "MSFT", "TSLA", "AMD")]
                dl = gr.DownloadButton("⬇️ Hisobotni yuklab olish (HTML)", visible=False, size="sm")
                one_out = gr.HTML(value="")
            with gr.Tab("📊 Sifat"):
                quality_out = gr.HTML(value=_loading_html("Tizim tayyorlanmoqda..."))
            with gr.Tab("ℹ️ Usul"):
                gr.Markdown(METHODOLOGY_MD)
        demo.load(ui_load, None, [reco_out, quality_out])
        retry_btn.click(ui_retry, None, [reco_out, quality_out])
        for qb in quick:
            qb.click(lambda t: t, qb, ticker).then(ui_analyze, [ticker, target], [one_out, dl])
        go.click(ui_analyze, [ticker, target], [one_out, dl])
        ticker.submit(ui_analyze, [ticker, target], [one_out, dl])
    try:
        demo.queue(default_concurrency_limit=8)
    except TypeError:
        demo.queue()
    return demo


def _launch_kwargs(demo, host: str, port: int, share: bool) -> Dict[str, Any]:
    import inspect
    kw: Dict[str, Any] = dict(server_name=host, server_port=port, share=share, show_error=True)
    params = inspect.signature(demo.launch).parameters
    if "footer_links" in params:          # Gradio 6: "API" havolasini yashiramiz
        kw["footer_links"] = ["settings", "gradio"]
    if "css" in params:                   # Gradio 6: CSS launch() da beriladi (birinchi chizishdan oldin qoʻllanadi)
        kw["css"] = APP_CSS
    return kw


def main(share: bool = False) -> None:
    start_background()
    demo = build_demo()
    port = int(os.environ.get("PORT", "7860"))
    host = "0.0.0.0" if (os.environ.get("PORT") or share) else "127.0.0.1"
    log(f"Ilova ishga tushmoqda: http://{host}:{port}  (tizim fon rejimida tayyorlanmoqda)")
    demo.launch(**_launch_kwargs(demo, host, port, share))


# ======================================================================================
# 8. SINTETIK BOZOR VA OʻZ-OʻZINI TEKSHIRISH (metodologiya toʻgʻriligini isbotlash)
# ======================================================================================
def base_breakout_series(days: int = 1700, seed: int = 11, cut: Optional[str] = None) -> pd.DataFrame:
    """SPSC-tipidagi sintetik sinov seriyasi: uzoq oʻsish -> ~6 oylik "baza" (diapazon, Spring va past hajmli Test bilan)
    -> bazadan yuqoriga chiqish (breakout, hajm bilan) va davom etish. Dvigatel mexanikasini tekshirish uchun; haqiqiy aksiya EMAS.
    cut: None — toʻliq; "base" — breakout'dan oldin (baza oxiri); "test" — Spring va Test'dan keyin darhol."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=days)
    n_base, n_brk = 120, 75
    n_pre = days - n_base - n_brk
    lp = np.cumsum(rng.normal(0.0006, 0.014, n_pre))
    lp = lp - lp[-1]                                          # bazaga kirish nuqtasi = 0 (log-daraja)
    top, bot = 0.0, -0.15
    t_ = np.arange(n_base)
    base = (top + bot) / 2 + (top - bot) / 2 * 0.96 * np.cos(2 * np.pi * t_ / 30) + rng.normal(0, 0.005, n_base)   # diapazon ichida tebranish (davri 30 kun)
    base = np.clip(base, bot + 0.012, top - 0.004)
    base[75] = bot + 0.012                                    # Spring kuni (diapazon pastida): kun ichida past, yopilish diapazon ichida
    base[81] = bot + 0.016                                    # Test kuni
    brk = np.empty(n_brk)
    lvl = top - 0.01
    for i in range(n_brk):                                    # breakout: qarshilikdan yuqoriga va davom etish
        lvl += (0.0075 if i < 40 else 0.0030) + rng.normal(0, 0.013)
        brk[i] = lvl
    path = np.concatenate([lp, base, brk]) + math.log(60.0)
    close = np.exp(path)
    opn = np.empty(days)
    opn[0] = close[0]
    opn[1:] = close[:-1] * np.exp(rng.normal(0, 0.004, days - 1))
    hi = np.maximum(opn, close) * np.exp(np.abs(rng.normal(0, 0.008, days)))
    lo = np.minimum(opn, close) * np.exp(-np.abs(rng.normal(0, 0.008, days)))
    vol = np.exp(rng.normal(13.2, 0.25, days))
    sp, ts = n_pre + 75, n_pre + 81
    opn[sp] = close[sp - 1]
    lo[sp] = close[sp] * math.exp(-0.05)                      # Spring: diapazon pastidan ~5% pastga sanchilib, qaytib yopilgan
    vol[sp] *= 2.6                                            # yuqori hajm
    lo[ts] = lo[sp] * 1.012                                   # Test: Spring pastiga yaqin, lekin past hajmda
    vol[ts] *= 0.5
    vol[n_pre + n_base:n_pre + n_base + 6] *= 2.2             # breakout hajmi
    hi = np.maximum(hi, np.maximum(opn, close))
    lo = np.minimum(lo, np.minimum(opn, close))
    df = pd.DataFrame({"Open": opn, "High": hi, "Low": lo, "Close": close, "Volume": vol}, index=idx)
    if cut == "base":
        return df.iloc[:n_pre + n_base]
    if cut == "test":
        return df.iloc[:n_pre + 83]
    return df


def make_synthetic_universe(n_stocks: int = 60, days: int = 1700, seed: int = 0, alpha: float = 0.0,
                            start: str = "2016-01-04") -> Dict[str, pd.DataFrame]:
    """Sintetik bozor.  alpha=0 -> toza tasodifiy yurish (bashorat qilinadigan signal YOʻQ).
    alpha>0 -> har bir aksiyada sekin oʻzgaruvchi yashirin drift (momentum) ekilgan: model uni topishi kerak."""
    from scipy.signal import lfilter
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=days)
    mkt = rng.standard_t(5, days) / math.sqrt(5 / 3) * 0.010 + 0.0003

    def ohlc(ret: np.ndarray, sig: float, base: float = 100.0) -> pd.DataFrame:
        close = base * np.exp(np.cumsum(ret))
        opn = np.empty(days)
        opn[0] = base
        opn[1:] = close[:-1] * np.exp(rng.normal(0, 0.15 * sig, days - 1))
        hi = np.maximum(opn, close) * np.exp(np.abs(rng.normal(0, 0.45 * sig, days)))
        lo = np.minimum(opn, close) * np.exp(-np.abs(rng.normal(0, 0.45 * sig, days)))
        vol = np.exp(rng.normal(14, 0.3, days)) * (1 + 5 * np.abs(ret) / sig)
        return pd.DataFrame({"Open": opn, "High": hi, "Low": lo, "Close": close, "Volume": vol}, index=idx)

    data = {BENCH: ohlc(mkt, 0.010)}
    rho = 0.998
    for i in range(n_stocks):
        beta = rng.uniform(0.6, 1.5)
        sig = rng.uniform(0.009, 0.025)
        idio = rng.standard_t(5, days) / math.sqrt(5 / 3) * sig
        z = lfilter([math.sqrt(1 - rho ** 2)], [1.0, -rho], rng.standard_normal(days))
        ret = beta * mkt + alpha * z + idio
        data[f"S{i + 1:03d}"] = ohlc(ret, sig)
    return data


def _check_no_lookahead(data: Dict[str, pd.DataFrame], tick: str) -> Tuple[bool, str]:
    df, market = data[tick], data[BENCH]["Close"]
    full = compute_features(df, market)
    cols = FEATURES + EXTRA
    bad = []
    for cut in (400, 700, 1000, len(df) - 3):
        part = compute_features(df.iloc[:cut], market)
        a = full.iloc[cut - 1][cols].astype(float).values
        b = part.iloc[-1][cols].astype(float).values
        if not np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True):
            diff = [cols[j] for j in range(len(cols))
                    if not np.isclose(a[j], b[j], rtol=1e-9, atol=1e-12, equal_nan=True)]
            bad.append((cut, diff))
    return (not bad), ("belgilar kelajak maʼlumotiga bogʻliq emas" if not bad else f"KELAJAKKA QARASH: {bad}")


def _check_labels(ds: Dataset, data: Dict[str, pd.DataFrame], H: int = 20, n: int = 300) -> Tuple[bool, str]:
    S = ds.S.sample(min(n, len(ds.S)), random_state=1)
    bad = 0
    for _, r in S.iterrows():
        df = data[r["ticker"]]
        loc = df.index.get_loc(r["date"])
        exp = df["Close"].iloc[loc + H] / df["Close"].iloc[loc] - 1 if loc + H < len(df) else np.nan
        got = r[f"fwd{H}"]
        if not ((np.isnan(exp) and np.isnan(got)) or np.isclose(exp, got, rtol=1e-9, atol=1e-12)):
            bad += 1
    return bad == 0, f"etiketlar {n} ta tasodifiy qatorda qayta hisoblab tekshirildi, nomuvofiqlik: {bad}"


def selftest(verbose: bool = True) -> bool:
    """Metodologiyani sintetik maʼlumotda tekshiradi:
       (1) belgilarda kelajakka qarash yoʻq; (2) etiketlar toʻgʻri; (3) toza tasodifiy bozorda model "ustunlik topmaydi";
       (4) ekilgan momentum signali topiladi; (5) oddiy tasodifiy train/test boʻlinishi sunʼiy yuqori natija beradi."""
    ok_all = True

    def rep(name: str, ok: bool, msg: str):
        nonlocal ok_all
        ok_all &= bool(ok)
        if verbose:
            print(("  [OK]   " if ok else "  [XATO] ") + name + " — " + msg, flush=True)

    hz = [20, 60]
    t0 = time.time()
    null = make_synthetic_universe(alpha=0.0, seed=1)
    ds0 = build_dataset(null, horizons=hz)
    ok, msg = _check_no_lookahead(null, "S007")
    rep("Kelajakka qarash testi", ok, msg)
    ok, msg = _check_labels(ds0, null)
    rep("Etiket toʻgʻriligi testi", ok, msg)
    ev0 = walk_forward_eval(ds0, 20)
    rep("Toza tasodifiy bozor: ustunlik topilmasligi kerak",
        ev0.verdict in ("none", "inverse") or ev0.ic_t < T_STRONG,
        f"AUC={ev0.auc:.3f}, IC={ev0.ic_mean:+.4f}, t={ev0.ic_t:+.2f} -> hukm: {ev0.verdict}")
    naive = naive_split_auc(ds0, 20)
    rep("Tasodifiy train/test (leakage) AUC sunʼiy yuqori chiqishi", naive > ev0.auc + 0.02,
        f"tasodifiy boʻlinish AUC={naive:.3f}  vs  purged walk-forward AUC={ev0.auc:.3f} (haqiqatda signal yoʻq!)")
    pl = make_synthetic_universe(alpha=0.0008, seed=2)
    ds1 = build_dataset(pl, horizons=hz)
    ev1 = walk_forward_eval(ds1, 20)
    rep("Ekilgan momentum signali topilishi kerak", ev1.verdict in ("weak", "strong") and ev1.ic_mean > 0,
        f"AUC={ev1.auc:.3f}, IC={ev1.ic_mean:+.4f}, t={ev1.ic_t:+.2f} -> hukm: {ev1.verdict}")
    bt = backtest_topn(ds1, ev1, H=20)
    rep("Backtest ishlaydi", bt is not None and bt.n_periods >= 8,
        "davrlar: %s" % (bt.n_periods if bt else 0))
    # (6) baza / breakout / Spring-Test detektorlari (mexanika tekshiruvi) va tasodifiy bozordagi "yolgʻon" ishlash chastotasi
    b_pre, b_full = find_base(base_breakout_series(cut="base")), find_base(base_breakout_series())
    rep("Baza aniqlanadi (breakout'dan oldin)", b_pre.get("ok") and b_pre["state"] == "inside" and b_pre["pos"] >= 0.6,
        (f"holat={b_pre['state']}, diapazon {b_pre['lo']:.1f}–{b_pre['hi']:.1f}, {b_pre['W']} kun" if b_pre.get("ok") else "topilmadi"))
    rep("Breakout aniqlanadi (hajm bilan, retest bilan)", b_full.get("ok") and b_full["state"] == "breakout" and b_full["confirmed"],
        (f"holat={b_full['state']}, {b_full['days_since']} kun oldin, hajm {b_full['vol_ratio']:.1f}×, maqsadlar "
         + ", ".join(f"{x['price']:.1f}" for x in b_full["targets"]) if b_full.get("ok") else "topilmadi"))
    ev_t = wyckoff_events(base_breakout_series(cut="test"))
    rep("Spring va Test aniqlanadi", ev_t["spring"] is not None and ev_t["test"] is not None,
        f"Spring {ev_t['spring']['date']:%Y-%m-%d} (hajm {ev_t['spring']['vol_ratio']:.1f}×), Test {ev_t['test']['date']:%Y-%m-%d}" if ev_t["spring"] and ev_t["test"] else "topilmadi")
    n_chk = n_base = n_brk = 0
    for t, d in null.items():
        if t == BENCH:
            continue
        for cut in (0, 150, 300, 450, 600, 750, 900):
            bb = find_base(d.iloc[:len(d) - cut])
            n_chk += 1
            n_base += int(bool(bb.get("ok")))
            n_brk += int(bool(bb.get("ok") and bb["state"] == "breakout" and bb["days_since"] <= 60 and bb["confirmed"]))
    rep("Tasodifiy bozorda baza/breakout 'topilishi' oʻz-o'zidan signal emas (halol chastota)", n_brk / n_chk < 0.12,
        f"signalsiz tasodifiy yurishda: 'baza' {n_base / n_chk * 100:.0f}% oynalarda, 'hajm bilan tasdiqlangan breakout (≤60 kun)' {n_brk / n_chk * 100:.1f}% da chiqadi")
    if verbose:
        print(f"  Jami vaqt: {time.time() - t0:.0f} s", flush=True)
    return ok_all


def check_data() -> bool:
    """Maʼlumot manbalarini HAQIQIY internetda tekshiradi (Yahoo, Stooq) va jadval chiqaradi.
    Narx manbalaridan kamida bittasi ishlasa True (tizim ishlay oladi). Fundamental va yangilik — ixtiyoriy."""
    rows: List[Tuple[str, str, bool, str, float]] = []

    def run(kind: str, name: str, fn: Callable[[], str]) -> None:
        t0 = time.time()
        try:
            rows.append((kind, name, True, fn(), time.time() - t0))
        except Exception as e:
            rows.append((kind, name, False, f"{type(e).__name__}: {str(e)[:150]}", time.time() - t0))

    def yf_bulk() -> str:
        d = _yf_bulk(["SPY", "AAPL", "AMD"], 1.0, batch=3)
        if not d:
            raise DataError("javob boʻsh (cheklangan boʻlishi mumkin)")
        return f"{len(d)}/3 ta tiker, oxirgi sana {max(x.index[-1] for x in d.values()).date()}"

    def yf_one() -> str:
        import yfinance as yf
        d = _clean_ohlcv(yf.Ticker("AAPL").history(period="3mo", interval="1d", auto_adjust=True))
        if d is None:
            raise DataError("javob boʻsh")
        return f"{len(d)} qator, oxirgi sana {d.index[-1].date()}"

    def stooq() -> str:
        d = _stooq_one("AAPL", 1.0)
        if d is None:
            raise DataError("javob boʻsh")
        return f"{len(d)} qator, oxirgi sana {d.index[-1].date()}"

    def fund() -> str:
        d = fetch_fundamentals("AAPL")
        if not d.get("ok"):
            raise DataError(d.get("error", "?"))
        return f"sektor: {d.get('sector')}, P/E: {d.get('trailingPE')}"

    def news() -> str:
        d = fetch_news("AAPL")
        if not d.get("ok") or not d["items"]:
            raise DataError(d.get("error") or "sarlavhalar yoʻq")
        return f"{len(d['items'])} ta sarlavha"

    run("NARX", "Yahoo (ommaviy yuklash — tizim shuni ishlatadi)", yf_bulk)
    run("NARX", "Yahoo (bitta tiker)", yf_one)
    run("NARX", "Stooq (zaxira)", stooq)
    run("IXTIYORIY", "Yahoo fundamental koʻrsatkichlar", fund)
    run("IXTIYORIY", "Yahoo yangiliklar", news)
    print("\nMAʼLUMOT MANBALARI TEKSHIRUVI")
    for kind, name, ok, msg, sec in rows:
        print(f"  [{'OK ' if ok else 'XATO'}] {kind:<9} {name:<48} {msg}  ({sec:.1f} s)")
    price_ok = any(ok for kind, _, ok, _, _ in rows if kind == "NARX")
    bulk_ok = rows[0][2]
    if price_ok and not bulk_ok:
        print("\n  Eslatma: Yahoo ommaviy yuklash ishlamayapti, lekin zaxira manba bor — tizim ishlaydi, faqat sekinroq "
              "(har bir tiker alohida yuklanadi).")
    print("\n  NATIJA: " + ("narx manbasi mavjud — tizim ishlay oladi." if price_ok else
                            "HECH QANDAY narx manbasi ishlamadi — tizim real maʼlumotsiz ishlamaydi. Keyinroq qayta urining."))
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        try:
            with open(summ, "a", encoding="utf-8") as f:
                f.write("### Maʼlumot manbalari\n\n| Manba | Holat | Izoh |\n|---|---|---|\n")
                for kind, name, ok, msg, sec in rows:
                    f.write(f"| {name} | {'✅' if ok else '❌'} | {msg.replace('|', '/')} |\n")
        except Exception:
            pass
    return price_ok


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description="Aksiya AI v%s" % APP_VERSION)
    _ap.add_argument("--share", action="store_true", help="vaqtinchalik ochiq havola (Colab)")
    _ap.add_argument("--selftest", action="store_true", help="sintetik maʼlumotda metodologiya tekshiruvi")
    _ap.add_argument("--check-data", action="store_true", help="Yahoo/Stooq maʼlumot manbalarini haqiqiy internetda tekshirish")
    _args = _ap.parse_args()
    if _args.selftest:
        sys.exit(0 if selftest() else 1)
    if _args.check_data:
        sys.exit(0 if check_data() else 1)
    main(share=_args.share)
