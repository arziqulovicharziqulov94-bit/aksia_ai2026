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
APP_VERSION = "4.2"
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
JPM BAC WFC GS MS C V MA AXP BLK SCHW BRK-B SPGI CB PGR MRSH USB PNC TFC COF ICE CME AON
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
    dropped = [t for t in tickers if t not in keep]
    if dropped:                                   # ro'yxat eskirsa (tiker o'zgargan, birjadan chiqqan) — jurnalda aniq ko'rinsin
        why = []
        for t in dropped:
            d = data.get(t)
            if d is None or len(d) == 0:
                why.append(f"{t} (maʼlumot yoʻq)")
            elif len(d) < WARMUP + 60:
                why.append(f"{t} (tarix qisqa: {len(d)} kun)")
            else:
                why.append(f"{t} (eskirgan: oxirgi sana {d.index[-1].date()})")
        log(f"Universumdan tushib qolgan tikerlar ({len(dropped)}): " + ", ".join(why[:60])
            + " — ro'yxat eskirgan bo'lishi mumkin (tiker nomi o'zgargan yoki aksiya birjadan chiqqan)")
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
    no_latest = [t for t in tick if t not in latest_rows]
    if no_latest:                                 # joriy kesimga (AI bali/to'liq sahifa) kirmagan tikerlar — sababini topish uchun
        log(f"Joriy kesimga kirmagan tikerlar ({len(no_latest)}): " + ", ".join(no_latest[:60]))
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
    if levels:
        mxs = {h: np.sort(L[:, :h].max(axis=1)) for h in (20, 60, 125) if h in horizons}
        mns = {h: np.sort(L[:, :h].min(axis=1)) for h in (20, 60, 125) if h in horizons}
        nP = L.shape[0]
    for name, price in (levels or []):
        b = math.log(price / P0)
        d = {}
        for h in mxs if levels else ():
            d[h] = float(1.0 - np.searchsorted(mxs[h], b, side="left") / nP) if price >= P0 else float(np.searchsorted(mns[h], b, side="right") / nP)
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
        half = max(0.5 * float(ps.max() - ps.min()), 0.4 * tol)                  # zona yarim kengligi: klaster tarqoqligi yoki 0.4·ATR dan kattasi
        lv.append(dict(price=float(ps.mean()), touches=len(ps), last=d.index[int(idx.max())],
                       score=float(len(ps) + (idx / n).sum()), lo=float(ps.mean() - half), hi=float(ps.mean() + half)))
    sup = sorted([x for x in lv if price / 3.0 <= x["price"] < price - 0.25 * tol], key=lambda x: price - x["price"])      # 3 baravardan uzoq darajalar
    res = sorted([x for x in lv if price + 0.25 * tol < x["price"] <= price * 3.0], key=lambda x: x["price"] - price)       # (qulagan aksiyalarning eski choʻqqilari) olinmaydi

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


# ======================================================================================
# 5b. TEXNIK CHIZMALAR (mexanik qoidalar): pivotlar, trend chiziqlari va kanal, zonalar, Fibonachchi, patternlar, Elliott
# ======================================================================================
# Hammasi faqat OʻTMISH maʼlumotidan, aniq qoidalar bilan hisoblanadi (kelajakka qarash yoʻq). Har bir blok oʻzining
# "tasodifiy bozordagi ishlash chastotasi" bilan birga hujjatlashtirilgan (selftest): detektor topishi oʻz-oʻzidan signal emas.
ZZ_K = 1.5                                    # zigzag chegarasi = k · σ_kunlik · √L  (log narxda)
ZZ_L = {"minor": 6, "mid": 20, "major": 60}   # "daraja": L kunlik oraliqdagi oʻrtacha tasodifiy harakat shkalasi
DEG_UZ = {"minor": "kichik", "mid": "oʻrta", "major": "katta"}
TA_BARS = 400                                 # detektorlar oxirgi shuncha barda ishlaydi
BRK_ATR = 0.5                                 # breakout/breakdown: chiziqdan kamida shuncha ATR ortiq YOPILISH
FLAT_SN = 0.30                                # normallashtirilgan qiyalik (σ-birlikda, 20 bar uchun) shundan kichik boʻlsa "yassi"


def _trap(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapetsiyasimon moslik funksiyasi: [b, c] da 1, a va d da 0, oraligʻida chiziqli."""
    if not np.isfinite(x) or x <= a or x >= d:
        return 0.0
    if x < b:
        return (x - a) / (b - a)
    if x <= c:
        return 1.0
    return (d - x) / (d - c)


def sigma_daily(df: pd.DataFrame) -> float:
    """Kunlik log-daromad volatilligi (Monte-Karlo bilan bir xil aralashma: EWMA va 1 yillik)."""
    lr = np.log(df["Close"].astype(float)).diff().dropna()
    if len(lr) < 30:
        return max(1e-4, float(lr.std() or 0.02))
    lt = float(lr.tail(252).std())
    ew = float(lr.ewm(alpha=0.06, min_periods=30).std().iloc[-1])
    s = math.sqrt(0.5 * ew ** 2 + 0.5 * lt ** 2) if np.isfinite(ew) else lt
    return max(1e-4, float(s))


def zigzag(H: np.ndarray, L: np.ndarray, thr: float) -> List[Dict[str, Any]]:
    """ZigZag pivotlari (log narxda, High/Low boʻyicha). Pivot oxirgi ekstremumdan kamida `thr` (log) qaytgandagina tasdiqlanadi;
    oxirgi pivot — joriy (tasdiqlanmagan, ok=False) ekstremum. Har bir pivot: i (indeks), y (ln narx), k (+1 yuqori / −1 past), ok."""
    n = len(H)
    if n < 3 or thr <= 0:
        return []
    yh, yl = np.log(H), np.log(L)
    piv: List[Dict[str, Any]] = []
    trend = 0
    hi_i, hi_v, lo_i, lo_v = 0, float(yh[0]), 0, float(yl[0])
    for i in range(1, n):
        if trend == 0:
            if yh[i] > hi_v:
                hi_i, hi_v = i, float(yh[i])
            if yl[i] < lo_v:
                lo_i, lo_v = i, float(yl[i])
            if hi_v - lo_v >= thr:
                if lo_i < hi_i:
                    piv.append(dict(i=lo_i, y=lo_v, k=-1, ok=True))
                    trend = 1
                else:
                    piv.append(dict(i=hi_i, y=hi_v, k=1, ok=True))
                    trend = -1
        elif trend == 1:
            if yh[i] > hi_v:
                hi_i, hi_v = i, float(yh[i])
            elif hi_v - yl[i] >= thr:
                piv.append(dict(i=hi_i, y=hi_v, k=1, ok=True))
                trend, lo_i, lo_v = -1, i, float(yl[i])
        else:
            if yl[i] < lo_v:
                lo_i, lo_v = i, float(yl[i])
            elif yh[i] - lo_v >= thr:
                piv.append(dict(i=lo_i, y=lo_v, k=-1, ok=True))
                trend, hi_i, hi_v = 1, i, float(yh[i])
    if trend == 1:
        piv.append(dict(i=hi_i, y=hi_v, k=1, ok=False))
    elif trend == -1:
        piv.append(dict(i=lo_i, y=lo_v, k=-1, ok=False))
    return piv


class _W:
    """Detektorlar uchun oyna konteksti: oxirgi TA_BARS bar, log-narxlar, ATR va hajm. Indekslar oyna ichida (0..n-1); off — mutlaq siljish."""

    def __init__(self, df: pd.DataFrame, T: Dict[str, Any]):
        d = df.tail(T["n"])
        self.d, self.off, self.n = d, T["off"], len(d)
        self.O, self.H, self.L, self.C, self.V = (d[k].values.astype(float) for k in ("Open", "High", "Low", "Close", "Volume"))
        self.yc, self.yh, self.yl = np.log(self.C), np.log(self.H), np.log(self.L)
        self.sig = T["sigma"]
        self.atr = atr_series(df).tail(self.n).values.astype(float)
        a = float(self.atr[-1]) if np.isfinite(self.atr[-1]) else float(np.nanmean(self.atr))
        self.atr_l = max(a / float(self.C[-1]), 0.004)
        self.idx = d.index

    def date(self, i: int):
        return self.idx[int(min(max(i, 0), self.n - 1))]


def compute_pivots(df: pd.DataFrame, bars: int = TA_BARS) -> Dict[str, Any]:
    """Uch darajali (kichik/oʻrta/katta) ZigZag pivotlari. Chegara = ZZ_K·σ·√L (σ — joriy kunlik volatillik)."""
    d = df.tail(bars)
    sig = sigma_daily(df)
    H, L = d["High"].values.astype(float), d["Low"].values.astype(float)
    T: Dict[str, Any] = dict(off=len(df) - len(d), n=len(d), sigma=sig, piv={}, thr={})
    for deg, Ld in ZZ_L.items():
        thr = ZZ_K * sig * math.sqrt(Ld)
        T["thr"][deg] = thr
        T["piv"][deg] = zigzag(H, L, thr)
    return T


def _lp(line: Dict[str, Any], i_abs: float) -> float:
    """Chiziq (i0, p0, s: log qiyalik/bar) ning mutlaq indeksdagi narxi."""
    return float(line["p0"] * math.exp(line["s"] * (i_abs - line["i0"])))


# --------------------------------------------------------------------------------------
# Trend chiziqlari va kanal
# --------------------------------------------------------------------------------------
def _best_trendline(W: _W, anchors: List[Dict[str, Any]], touch_pts: List[Dict[str, Any]], side: int) -> Optional[Dict[str, Any]]:
    """side=−1: qoʻllab-quvvatlash chizigʻi (pastlar orqali), +1: qarshilik (yuqorilar orqali).
    Qoida: chiziq ikkita pivotdan oʻtadi; ikkinchi tayanchgacha hech bir YOPILISH chiziqning narigi tomonida boʻlmasligi shart;
    teginishlar — chiziqqa 0.6·ATR dan yaqin, kamida 4 bar oraliqdagi pivotlar. Eng koʻp teginishli, eng yangi chiziq tanlanadi."""
    n, yc = W.n, W.yc
    tol_v, tol_t = 0.5 * W.atr_l, 0.6 * W.atr_l
    best: Optional[Dict[str, Any]] = None
    pts = anchors[-7:]
    for ai in range(len(pts) - 1):
        a = pts[ai]
        for bi in range(ai + 1, len(pts)):
            b = pts[bi]
            if b["i"] - a["i"] < 8:
                continue
            s = (b["y"] - a["y"]) / (b["i"] - a["i"])
            if abs(s) * 252 > 4.0:
                continue
            idx = np.arange(a["i"], n)
            line = a["y"] + s * (idx - a["i"])
            viol = (line - yc[a["i"]:]) * (-side) > tol_v
            k_b = b["i"] - a["i"]
            if viol[:k_b + 1].any():
                continue
            post = np.flatnonzero(viol[k_b + 1:])
            brk = int(b["i"] + 1 + post[0]) if len(post) else None
            if brk is not None and n - 1 - brk > 12:
                continue
            end = brk if brk is not None else n - 1
            if abs(float(yc[-1]) - (a["y"] + s * (n - 1 - a["i"]))) > 4.0 * W.sig * math.sqrt(20):     # chiziq narxdan juda uzoq — ahamiyatsiz
                continue
            cnt, last_i = 0, -99
            for p in sorted(touch_pts + [a, b], key=lambda q: q["i"]):
                if a["i"] <= p["i"] <= end and abs(p["y"] - (a["y"] + s * (p["i"] - a["i"]))) <= tol_t and p["i"] - last_i >= 4:
                    cnt += 1
                    last_i = p["i"]
            if n - 1 - last_i > 120:                                                                  # oxirgi teginish juda eski
                continue
            sc = cnt + 0.5 * (last_i >= n - 45) + 0.002 * (b["i"] - a["i"]) - (1.0 if brk is not None else 0.0)
            if best is None or sc > best["score"]:
                best = dict(side=side, a=a, b=b, s=s, touches=cnt, last=last_i, brk=brk, score=sc, end=end)
    return best


def _tl_out(W: _W, c: Dict[str, Any]) -> Dict[str, Any]:
    a, s, n = c["a"], c["s"], W.n
    v_now = math.exp(a["y"] + s * (n - 1 - a["i"]))
    price = float(W.C[-1])
    broke = c["brk"] is not None
    return dict(side=c["side"], i0=W.off + a["i"], p0=math.exp(a["y"]), i1=W.off + c["b"]["i"], p1=math.exp(c["b"]["y"]), s=s,
                slope_ann=math.exp(s * 252) - 1.0, touches=c["touches"], last=W.off + c["last"], broke=broke,
                broke_i=(W.off + c["brk"]) if broke else None, days_since=(n - 1 - c["brk"]) if broke else None,
                now=v_now, dist=price / v_now - 1.0, fut20=v_now * math.exp(s * 20), deg=c.get("deg", "oʻrta"),
                confirmed=bool(c["touches"] >= 3), date0=W.date(a["i"]), date1=W.date(c["b"]["i"]))


def regression_channel(W: _W, n_bars: int = 100) -> Dict[str, Any]:
    """Chiziqli regressiya kanali: oxirgi N bar log-yopilish narxiga toʻgʻri chiziq va ±2 standart ogʻish (qoldiq) bandlari."""
    N = int(min(n_bars, W.n))
    x = np.arange(N, dtype=float)
    y = W.yc[-N:]
    b, a = np.polyfit(x, y, 1)
    fit = a + b * x
    res = y - fit
    sd = float(res.std(ddof=2)) or 1e-9
    r2 = float(1.0 - res.var() / y.var()) if y.var() > 0 else 0.0
    i0 = W.off + W.n - N
    return dict(n=N, i0=i0, p0=math.exp(a), s=float(b), slope_ann=math.exp(b * 252) - 1.0, r2=r2, sd=sd, z=float(res[-1] / sd),
                mid=math.exp(fit[-1]), up=math.exp(fit[-1] + 2 * sd), lo=math.exp(fit[-1] - 2 * sd),
                width=math.exp(4 * sd) - 1.0, date0=W.date(W.n - N))


def find_trendlines(df: pd.DataFrame, T: Dict[str, Any]) -> Dict[str, Any]:
    """Qoʻllab-quvvatlash va qarshilik trend chiziqlari, kanal va regressiya kanali (hammasi pivotlardan, mexanik)."""
    W = _W(df, T)
    best: Dict[int, Optional[Dict[str, Any]]] = {}
    for side in (-1, 1):
        cand = None
        for deg in ("mid", "minor"):
            anchors = [p for p in T["piv"][deg] if p["k"] == side and p["ok"]]       # tayanch faqat tasdiqlangan pivot (joriy ekstremum emas)
            c = _best_trendline(W, anchors, [p for p in T["piv"]["minor"] if p["k"] == side], side)
            if c and (cand is None or (c["touches"], c["last"]) > (cand["touches"], cand["last"])):
                cand = c
                cand["deg"] = DEG_UZ[deg]
        best[side] = cand
    res: Dict[str, Any] = dict(ok=True, support=None, resistance=None, channel=None, trend="yon")
    sup = _tl_out(W, best[-1]) if best[-1] else None
    rs = _tl_out(W, best[1]) if best[1] else None
    res["support"], res["resistance"] = sup, rs
    # --- kanal: (a) ikki chiziq deyarli parallel boʻlsa — juftlik; (b) aks holda 3+ teginishli chiziqdan parallel
    n = W.n
    price = float(W.C[-1])
    ch = None
    sig = W.sig
    if sup and rs and not sup["broke"] and not rs["broke"]:
        sn_u, sn_l = rs["s"] * math.sqrt(20) / sig, sup["s"] * math.sqrt(20) / sig
        i_ref = max(sup["i0"], rs["i0"])
        w0 = math.log(_lp(rs, i_ref) / _lp(sup, i_ref))
        w1 = math.log(rs["now"] / sup["now"])
        if w0 > 0 and w1 > 0 and 0.75 <= w1 / w0 <= 1.3 and abs(sn_u - sn_l) < 0.35:
            ch = dict(kind="juftlik", top=dict(i0=rs["i0"], p0=rs["p0"], s=rs["s"]), bot=dict(i0=sup["i0"], p0=sup["p0"], s=sup["s"]),
                      touches_top=rs["touches"], touches_bot=sup["touches"])
    if ch is None:
        base = None
        for ln in (sup, rs):
            if ln and not ln["broke"] and ln["touches"] >= 3 and (base is None or ln["touches"] > base["touches"]):
                base = ln
        if base is not None:
            side_b = base["side"]
            opp = [p for p in T["piv"]["minor"] if p["k"] == -side_b and W.off + p["i"] >= base["i0"]]
            offs = [(p["y"] - (math.log(base["p0"]) + base["s"] * (W.off + p["i"] - base["i0"])), p) for p in opp]
            if offs:
                off_v = max(o for o, _ in offs) if side_b == -1 else min(o for o, _ in offs)
                if abs(off_v) >= 2.0 * W.atr_l and (off_v > 0) == (side_b == -1):
                    tc, last_i = 0, -99
                    for o, p in sorted(offs, key=lambda q: q[1]["i"]):
                        if abs(o - off_v) <= 0.8 * W.atr_l and p["i"] - last_i >= 4:
                            tc += 1
                            last_i = p["i"]
                    par = dict(i0=base["i0"], p0=base["p0"] * math.exp(off_v), s=base["s"])
                    own = dict(i0=base["i0"], p0=base["p0"], s=base["s"])
                    if tc >= 2:
                        ch = dict(kind="parallel", top=par if side_b == -1 else own, bot=own if side_b == -1 else par,
                                touches_top=(tc if side_b == -1 else base["touches"]), touches_bot=(base["touches"] if side_b == -1 else tc))
    if ch is not None:
        iN = W.off + n - 1
        tn, bn = _lp(ch["top"], iN), _lp(ch["bot"], iN)
        ch.update(top_now=tn, bot_now=bn, width=tn / bn - 1.0, pos=float((price - bn) / (tn - bn)) if tn > bn else float("nan"))
        res["channel"] = ch
    rg = regression_channel(W)
    res["regression"] = rg
    if rg["r2"] >= 0.25:                                  # regressiya maʼnoli boʻlsa — trend yoʻnalishi undan
        res["trend"] = "oʻsish" if rg["slope_ann"] > 0.10 else "pasayish" if rg["slope_ann"] < -0.10 else "yon"
    elif res["channel"] is not None:
        sa = math.exp(res["channel"]["top"]["s"] * 252) - 1.0
        res["trend"] = "oʻsish" if sa > 0.12 else "pasayish" if sa < -0.12 else "yon"
    else:
        res["trend"] = "yon"
    return res


# --------------------------------------------------------------------------------------
# Talab / taklif zonalari (supply / demand): "baza -> portlovchi sham" qoidasi
# --------------------------------------------------------------------------------------
def find_sd_zones(df: pd.DataFrame, lookback: int = 250) -> Dict[str, Any]:
    """Talab (demand) zonasi — kuchli yuqoriga portlovchi shamdan oldingi tor "baza" shamlari (taklif — teskarisi).
    Portlovchi sham: tana >= 1.4·ATR va yopilish diapazonning chetki 30% ida; baza: undan oldingi 1–4 ta tor sham (diapazon <= 1.05·ATR, tana <= 0.6·ATR).
    Zona narx YOPILISH bilan teshib oʻtilgach oʻchiriladi; unga qaytib kelgan har bir kirish "sinov" deb sanaladi (yangi zona kuchliroq)."""
    d = df.tail(lookback)
    n, off = len(d), len(df) - len(d)
    O, H, L, C = (d[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    atr = atr_series(df).tail(n).values.astype(float)
    price = float(C[-1])
    zones: List[Dict[str, Any]] = []
    for j in range(6, n - 1):
        a = atr[j - 1]
        if not np.isfinite(a) or a <= 0:
            continue
        body, rng = C[j] - O[j], H[j] - L[j]
        if abs(body) < 1.4 * a or rng <= 0:
            continue
        up = body > 0
        if (up and (C[j] - L[j]) < 0.7 * rng) or ((not up) and (H[j] - C[j]) < 0.7 * rng):
            continue
        base: List[int] = []
        k = j - 1
        while k >= max(0, j - 4) and np.isfinite(atr[k]) and (H[k] - L[k]) <= 1.05 * atr[k] and abs(C[k] - O[k]) <= 0.6 * atr[k]:
            base.append(k)
            k -= 1
        if not base:
            continue
        if up:
            zlo, zhi = float(min(L[k] for k in base)), float(max(max(O[k], C[k]) for k in base))
        else:
            zlo, zhi = float(min(min(O[k], C[k]) for k in base)), float(max(H[k] for k in base))
        if zhi <= zlo or (zhi - zlo) > 2.5 * a:
            continue
        broken, tests, inside = False, 0, False
        for m in range(j + 1, n):
            if (up and C[m] < zlo - 0.1 * a) or ((not up) and C[m] > zhi + 0.1 * a):
                broken = True
                break
            touch = (L[m] <= zhi) if up else (H[m] >= zlo)
            if touch and not inside:
                tests += 1
            inside = touch
        if broken:
            continue
        zones.append(dict(kind="demand" if up else "supply", lo=zlo, hi=zhi, i0=off + min(base), i1=off + j, tests=tests,
                          date=d.index[j], height=zhi - zlo, fresh=bool(tests == 0)))
    dem = sorted([z for z in zones if z["kind"] == "demand" and z["hi"] < price and z["hi"] > 0.6 * price], key=lambda z: price - z["hi"])
    sup = sorted([z for z in zones if z["kind"] == "supply" and z["lo"] > price and z["lo"] < 1.6 * price], key=lambda z: z["lo"] - price)

    def pick(lst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sel: List[Dict[str, Any]] = []
        for z in lst:
            if all(min(z["hi"], q["hi"]) - max(z["lo"], q["lo"]) <= 0.3 * min(z["height"], q["height"]) for q in sel):
                sel.append(z)
            if len(sel) == 2:
                break
        for z in sel:
            z["dist"] = (z["hi"] if z["kind"] == "demand" else z["lo"]) / price - 1.0
        return sel
    return dict(demand=pick(dem), supply=pick(sup))


# --------------------------------------------------------------------------------------
# Fibonachchi
# --------------------------------------------------------------------------------------
FIB_RET = (0.236, 0.382, 0.5, 0.618, 0.786)
FIB_EXT = (1.272, 1.618, 2.0)


def fib_levels(df: pd.DataFrame, T: Dict[str, Any]) -> Dict[str, Any]:
    """Soʻnggi yirik toʻlqin (katta daraja ZigZag; yetmasa oʻrta daraja) boʻyicha Fibonachchi tuzatish va kengayish darajalari."""
    piv = T["piv"]["major"]
    deg = "major"
    if len(piv) < 2:
        piv, deg = T["piv"]["mid"], "mid"
    if len(piv) < 2:
        return dict(ok=False)
    p_end, p_start = piv[-1], piv[-2]
    off = T["off"]
    up = p_end["k"] == 1
    lo_p, hi_p = (p_start, p_end) if up else (p_end, p_start)
    lo, hi = math.exp(lo_p["y"]), math.exp(hi_p["y"])
    price = float(df["Close"].iloc[-1])
    rng = hi - lo
    if rng <= 0:
        return dict(ok=False)
    if up:
        ret = [(r, hi - r * rng) for r in FIB_RET]
        ext = [(e, lo + e * rng) for e in FIB_EXT]
        now = (hi - price) / rng
    else:
        ret = [(r, lo + r * rng) for r in FIB_RET]
        ext = [(e, hi - e * rng) for e in FIB_EXT]
        now = (price - lo) / rng
    ext = [(e, p) for e, p in ext if p > 0.01 * price]                # manfiy yoki ma'nosiz (≈0) kengayish darajalari tushib qoladi
    return dict(ok=True, up=up, deg=DEG_UZ[deg], lo=lo, hi=hi, i_lo=off + lo_p["i"], i_hi=off + hi_p["i"], levels=ret, ext=ext,
                retr_now=float(now), price=price, date_lo=df.index[min(off + lo_p["i"], len(df) - 1)],
                date_hi=df.index[min(off + hi_p["i"], len(df) - 1)], bars=int(abs(hi_p["i"] - lo_p["i"])))

# --- Tasodifiy yurish (null) jadvallari: detektorlarning tasodifiy narxdagi ishlash chastotasi (selftest bilan tekshiriladi) ---
# Avtomatik yaratilgan (calibrate_null.py): sintetik tasodifiy yurish (t5 zarbalar, doimiy va stoxastik volatillik), 6000 ta 400-barli oyna.
ELLIOTT_NULL_X = (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)
ELLIOTT_NULL = {'i3': (0.7, 0.6915, 0.6817, 0.6708, 0.6603, 0.6493, 0.6355, 0.6207, 0.608, 0.5937, 0.5757, 0.5607), 'i4': (0.5607, 0.5152, 0.4645, 0.4377, 0.4017, 0.3692, 0.3352, 0.3003, 0.2705, 0.239, 0.2097, 0.1837), 'i5': (0.1357, 0.132, 0.1275, 0.1187, 0.113, 0.1007, 0.091, 0.0817, 0.0713, 0.0612, 0.0488, 0.028), 'i6': (0.18, 0.1753, 0.1698, 0.1628, 0.1493, 0.1362, 0.1188, 0.1053, 0.0882, 0.0662, 0.0363, 0.0), 'abc': (0.737, 0.7312, 0.7178, 0.7062, 0.6935, 0.678, 0.6447, 0.5977, 0.5472, 0.4833, 0.3303, 0.0)}
ELLIOTT_RULE_RATE = {'i3': 0.761, 'i4': 0.6038, 'i5': 0.1432, 'i6': 0.192, 'abc': 0.7435}
PAT_NULL_X = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
PAT_NULL = {'asc_triangle': (0.0493, 0.049, 0.0467, 0.0405, 0.0217, 0.0048), 'bear_flag': (0.032, 0.0272, 0.0173, 0.005, 0.0015, 0.0), 'bull_flag': (0.0287, 0.0245, 0.0142, 0.0045, 0.0012, 0.0003), 'desc_triangle': (0.045, 0.0443, 0.0422, 0.0358, 0.0178, 0.0038), 'double_bottom': (0.087, 0.0803, 0.06, 0.0397, 0.0165, 0.0027), 'double_top': (0.0783, 0.0715, 0.0563, 0.0365, 0.0158, 0.0027), 'falling_wedge': (0.032, 0.0315, 0.0293, 0.0245, 0.011, 0.0028), 'hs_bottom': (0.0232, 0.0218, 0.0168, 0.0107, 0.0058, 0.0017), 'hs_top': (0.0198, 0.0178, 0.0137, 0.0067, 0.004, 0.0007), 'pennant_dn': (0.0125, 0.0108, 0.0063, 0.0025, 0.0002, 0.0002), 'pennant_up': (0.0132, 0.0117, 0.0063, 0.0027, 0.0005, 0.0), 'rising_wedge': (0.0367, 0.0367, 0.0348, 0.0283, 0.0133, 0.0018), 'sym_triangle': (0.0163, 0.0157, 0.0142, 0.011, 0.0057, 0.0013)}
NULL_WINDOWS = 6000

# --- Tanlov hisobga olingan (family-wise) null: bir nechta variant ichidan ENG YAXSHISINI tanlash p-qiymatni sunʼiy kichraytiradi. ---
# Quyidagi jadvallar tasodifiy yurishda 'tanlangan eng yaxshi variantning p-qiymati <= x' boʻlish ehtimolini beradi:
# EW_ALL_FW — barcha Elliott nomzodlari ichidan; EW_FW — faqat 5–6 pivotli impulslar ichidan; PAT_FW — barcha topilgan patternlar ichidan.
# Avtomatik yaratilgan (calib_fw.py): tasodifiy yurishda (sintetik, t5 zarbalar) 'eng yaxshi variant p-qiymati <= x' ehtimoli, 12000 ta 400-barli oyna.
FW_X = (0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)
EW_ALL_FW = (0.0243, 0.0325, 0.0541, 0.0759, 0.1087, 0.1594, 0.2167, 0.2792, 0.3352, 0.4213, 0.5286, 0.5985, 0.7599, 0.9673, 1.0)
EW_FW = (0.0044, 0.0063, 0.011, 0.0152, 0.0238, 0.0592, 0.0939, 0.1403, 0.1823, 0.2521, 0.2767, 0.2767, 0.2767, 0.2767, 1.0)
PAT_FW = (0.0036, 0.0135, 0.0621, 0.1187, 0.2232, 0.2923, 0.3549, 0.3935, 0.4078, 0.4078, 0.4078, 0.4078, 0.4078, 0.4078, 1.0)
FW_WINDOWS = 12000


def fw_adjust(tab: Tuple[float, ...], p: Optional[float]) -> float:
    """Tanlov hisobga olingan p-qiymat: tasodifiy yurishda eng yaxshi variantning xomaki p-qiymati <= p boʻlish ehtimoli (FW_* jadvallaridan)."""
    if p is None or not np.isfinite(p):
        return float("nan")
    xs = FW_X
    if p <= xs[0]:
        return float(tab[0] * max(float(p), 0.0) / xs[0])
    if p >= xs[-1]:
        return 1.0
    return float(np.interp(math.log(p), np.log(xs), tab))



# --------------------------------------------------------------------------------------
# Chart-patternlar (mexanik qoidalar: pivotlar geometriyasi)
# --------------------------------------------------------------------------------------
PAT_UZ = {
    "double_top": "Ikkilamchi choʻqqi (Double Top)", "double_bottom": "Ikkilamchi tub (Double Bottom)",
    "hs_top": "Bosh va yelkalar (Head & Shoulders)", "hs_bottom": "Teskari bosh va yelkalar (Inverse H&S)",
    "asc_triangle": "Oʻsuvchi uchburchak", "desc_triangle": "Pasayuvchi uchburchak", "sym_triangle": "Simmetrik uchburchak",
    "rising_wedge": "Koʻtariluvchi pona (Rising Wedge)", "falling_wedge": "Tushuvchi pona (Falling Wedge)",
    "bull_flag": "Buqa bayrogʻi (Bull Flag)", "bear_flag": "Ayiq bayrogʻi (Bear Flag)",
    "pennant_up": "Vimpel (yuqoriga, Pennant)", "pennant_dn": "Vimpel (pastga, Pennant)"}
PAT_BIAS = {"double_top": -1, "double_bottom": 1, "hs_top": -1, "hs_bottom": 1, "asc_triangle": 1, "desc_triangle": -1, "sym_triangle": 0,
            "rising_wedge": -1, "falling_wedge": 1, "bull_flag": 1, "bear_flag": -1, "pennant_up": 1, "pennant_dn": -1}
PAT_STATE_FORM, PAT_STATE_OK = "shakllanmoqda", "tasdiqlangan"


def _seg(W: _W, i0: float, y0: float, i1: float, y1: float, role: str) -> Dict[str, Any]:
    return dict(i0=W.off + i0, p0=float(math.exp(y0)), i1=W.off + i1, p1=float(math.exp(y1)), role=role)


def _pt(W: _W, i: int, y: float, label: str = "") -> Dict[str, Any]:
    return dict(i=W.off + int(i), p=float(math.exp(y)), label=label)


def _usd2(x: float) -> str:
    return _money(x) if abs(x) < 1.0 else f"${x:,.2f}"


def _pat_double(W: _W, P: List[Dict[str, Any]], sign: int, theta: float, deg: str) -> List[Dict[str, Any]]:
    """Ikkilamchi choʻqqi (sign=+1) / tub (sign=−1). Qoida: ikki ekstremum farqi <= min(max(0.2θ, 1.5%), 4%), ular orasida kamida 1.2θ chuqurlik,
    ekstremumgacha kamida 1.0θ trend, 2-ekstremum tasdiqlangan (narx undan θ qaytgan), oraliq 8–160 bar. Tasdiq: boʻyin chizigʻidan (oradagi ekstremum) 0.5·ATR ortiq YOPILISH."""
    out: List[Dict[str, Any]] = []
    m = len(P)
    zc = sign * W.yc
    zl = W.yl if sign > 0 else -W.yh
    kind = "double_top" if sign > 0 else "double_bottom"
    for end in (m - 1, m - 2):
        if end < 3:
            continue
        p2, pm, p1, p0 = P[end], P[end - 1], P[end - 2], P[end - 3]
        if not (p2["k"] == sign and pm["k"] == -sign and p1["k"] == sign and p0["k"] == -sign):
            continue
        z1, z2, zm, z0 = (sign * q["y"] for q in (p1, p2, pm, p0))
        gap = p2["i"] - p1["i"]
        tol_pk = min(max(0.2 * theta, 0.015), 0.04)
        if not (8 <= gap <= 160) or abs(z1 - z2) > tol_pk or min(z1, z2) - zm < 1.2 * theta or z1 - z0 < 1.0 * theta:
            continue
        if not p2["ok"]:                      # 2-ekstremum tasdiqlanmagan (narx hali oʻsha yerda) — pattern hali shakllanmagan
            continue
        neck = float(zl[p1["i"]:p2["i"] + 1].min())
        ztop = max(z1, z2)
        brk = next((j for j in range(p2["i"] + 1, W.n) if zc[j] < neck - BRK_ATR * W.atr_l), None)
        if brk is None:
            if zc[-1] > ztop + 0.5 * W.atr_l or W.n - 1 - p2["i"] > 45:
                continue
            state, days = PAT_STATE_FORM, None
        else:
            days = W.n - 1 - brk
            if days > 40 or zc[-1] > neck + 1.0 * W.atr_l:
                continue
            state = PAT_STATE_OK
        height = (z1 + z2) / 2 - neck
        zt = neck - height
        a_, b_ = pm["i"] - p1["i"], p2["i"] - pm["i"]
        v1 = float(np.mean(W.V[max(0, p1["i"] - 2):p1["i"] + 3]))
        v2 = float(np.mean(W.V[max(0, p2["i"] - 2):p2["i"] + 3]))
        vol_ok = bool(v2 < v1)
        q = (0.4 * max(0.0, 1 - abs(z1 - z2) / tol_pk) + 0.3 * min(1.0, (min(z1, z2) - zm) / (1.6 * theta))
             + 0.15 * (min(a_, b_) / max(a_, b_, 1)) + 0.15 * (1.0 if vol_ok else 0.4))
        nm = "choʻqqi" if sign > 0 else "tub"
        dirw = "pastda" if sign > 0 else "yuqorida"
        calc = [f"1-{nm}: {W.date(p1['i']):%Y-%m-%d}, {_usd2(math.exp(sign * z1))}; 2-{nm}: {W.date(p2['i']):%Y-%m-%d}, {_usd2(math.exp(sign * z2))} "
                f"(farq {abs(math.exp(z1 - z2) - 1) * 100:.1f}%; ruxsat ≤ {tol_pk * 100:.1f}%).",
                f"Boʻyin chizigʻi (neckline) = ikki {nm} orasidagi ekstremum: {_usd2(math.exp(sign * neck))}.",
                f"Balandlik = oʻrtacha {nm} − boʻyin = {_usd2(abs(math.exp(sign * ((z1 + z2) / 2)) - math.exp(sign * neck)))}; "
                f"maqsad = boʻyin ∓ balandlik = {_usd2(math.exp(sign * zt))}.",
                f"Tasdiq: narx boʻyin chizigʻidan {_usd2(math.exp(sign * (neck - BRK_ATR * W.atr_l)))} {dirw}da yopilsa; bekor: {_usd2(math.exp(sign * ztop))} dan "
                f"{'yuqorida' if sign > 0 else 'pastda'} yopilish.",
                "2-ekstremumda hajm " + ("1-dan past (klassik belgi)." if vol_ok else "1-dan past emas (klassik belgi yoʻq).")]
        out.append(dict(kind=kind, name=PAT_UZ[kind], bias=PAT_BIAS[kind], state=state, days_since=days, quality=float(min(1.0, q)),
                        pts=[_pt(W, p1["i"], sign * z1, "1"), _pt(W, pm["i"], sign * zm, ""), _pt(W, p2["i"], sign * z2, "2")],
                        lines=[_seg(W, p1["i"], sign * neck, W.n - 1 + 12, sign * neck, "neck")],
                        trigger=float(math.exp(sign * (neck - BRK_ATR * W.atr_l))), target=float(math.exp(sign * zt)), invalid=float(math.exp(sign * ztop)),
                        height=float(abs(math.exp(sign * ((z1 + z2) / 2)) - math.exp(sign * neck))), start=W.off + p1["i"], end=W.off + p2["i"],
                        deg=DEG_UZ[deg], vol_ok=vol_ok, calc=calc, dir=-sign))
    return out


def _pat_hs(W: _W, P: List[Dict[str, Any]], sign: int, theta: float, deg: str) -> List[Dict[str, Any]]:
    """Bosh va yelkalar (sign=+1, yuqori) / teskari (sign=−1). Qoida: bosh ikkala yelkadan >= 0.4θ baland; yelkalar balandligi farqi
    <= 0.4·(bosh balandligi); boʻyin chizigʻi ikki oraliq pastdan (qiyaligi bosh balandligining 0.8 idan oshmasin); maqsad — bosh balandligi."""
    out: List[Dict[str, Any]] = []
    m = len(P)
    zc = sign * W.yc
    kind = "hs_top" if sign > 0 else "hs_bottom"
    for end in (m - 1, m - 2):
        if end < 4:
            continue
        rs, l2, hd, l1, ls = P[end], P[end - 1], P[end - 2], P[end - 3], P[end - 4]
        if not (rs["k"] == sign and l2["k"] == -sign and hd["k"] == sign and l1["k"] == -sign and ls["k"] == sign) or not rs["ok"]:
            continue
        zr, zl2, zh, zl1, zls = (sign * q["y"] for q in (rs, l2, hd, l1, ls))
        if zh - max(zls, zr) < 0.4 * theta:
            continue
        sl = (zl2 - zl1) / (l2["i"] - l1["i"])
        neck_at = lambda i: zl1 + sl * (i - l1["i"])
        hh = zh - neck_at(hd["i"])
        if hh < 0.8 * theta or abs(zls - zr) > 0.4 * hh or zr - neck_at(rs["i"]) < 0.4 * hh:
            continue
        if abs(sl * (rs["i"] - ls["i"])) > 0.8 * hh:
            continue
        ra = (hd["i"] - ls["i"]) / max(1, rs["i"] - hd["i"])
        if not (0.4 <= ra <= 2.5) or rs["i"] - ls["i"] > 220:
            continue
        brk = next((j for j in range(rs["i"] + 1, W.n) if zc[j] < neck_at(j) - BRK_ATR * W.atr_l), None)
        if brk is None:
            if zc[-1] > zh + 0.5 * W.atr_l or W.n - 1 - rs["i"] > 45:
                continue
            state, days = PAT_STATE_FORM, None
        else:
            days = W.n - 1 - brk
            if days > 40 or zc[-1] > neck_at(W.n - 1) + 1.0 * W.atr_l:
                continue
            state = PAT_STATE_OK
        i_ref = brk if brk is not None else W.n - 1
        zt = neck_at(i_ref) - hh
        sym = 1 - abs(zls - zr) / (0.4 * hh)
        q = 0.35 * max(0.0, sym) + 0.3 * min(1.0, (zh - max(zls, zr)) / (1.0 * theta)) + 0.2 * min(1.0, 1 - abs(math.log(ra)) / 1.0 if ra > 0 else 0) + 0.15 * (1 if abs(sl * (rs["i"] - ls["i"])) < 0.3 * hh else 0.5)
        v_h = float(np.mean(W.V[max(0, hd["i"] - 2):hd["i"] + 3]))
        v_l = float(np.mean(W.V[max(0, ls["i"] - 2):ls["i"] + 3]))
        v_r = float(np.mean(W.V[max(0, rs["i"] - 2):rs["i"] + 3]))
        vol_ok = bool(v_r < v_h)
        nm = "yelka" if sign > 0 else "teskari yelka"
        calc = [f"Chap yelka {W.date(ls['i']):%Y-%m-%d} {_usd2(math.exp(sign * zls))}; bosh {W.date(hd['i']):%Y-%m-%d} {_usd2(math.exp(sign * zh))}; "
                f"oʻng yelka {W.date(rs['i']):%Y-%m-%d} {_usd2(math.exp(sign * zr))}.",
                f"Boʻyin chizigʻi: {W.date(l1['i']):%Y-%m-%d} {_usd2(math.exp(sign * zl1))} va {W.date(l2['i']):%Y-%m-%d} {_usd2(math.exp(sign * zl2))} orqali; "
                f"bosh ostidagi qiymati {_usd2(math.exp(sign * neck_at(hd['i'])))}.",
                f"Bosh balandligi = bosh − boʻyin = {_usd2(abs(math.exp(sign * zh) - math.exp(sign * neck_at(hd['i']))))}; "
                f"maqsad = boʻyin ({_usd2(math.exp(sign * neck_at(i_ref)))}) ∓ bosh balandligi = {_usd2(math.exp(sign * zt))}.",
                f"Yelkalar simmetriyasi: farq {abs(math.exp(zls - zr) - 1) * 100:.1f}%; oʻng yelkada hajm " + ("boshdagidan past." if vol_ok else "boshdagidan past emas."),
                f"Tasdiq: narx boʻyin chizigʻidan 0.5·ATR ortiq {'pastda' if sign > 0 else 'yuqorida'} yopilsa; bekor: {_usd2(math.exp(sign * zh))} dan "
                f"{'yuqorida' if sign > 0 else 'pastda'} yopilish."]
        out.append(dict(kind=kind, name=PAT_UZ[kind], bias=PAT_BIAS[kind], state=state, days_since=days, quality=float(min(1.0, q)),
                        pts=[_pt(W, ls["i"], sign * zls, "LS"), _pt(W, hd["i"], sign * zh, "H"), _pt(W, rs["i"], sign * zr, "RS")],
                        lines=[_seg(W, l1["i"], sign * zl1, W.n - 1 + 12, sign * neck_at(W.n - 1 + 12), "neck")],
                        trigger=float(math.exp(sign * (neck_at(W.n - 1) - BRK_ATR * W.atr_l))), target=float(math.exp(sign * zt)),
                        invalid=float(math.exp(sign * zh)), height=float(abs(math.exp(sign * zh) - math.exp(sign * neck_at(hd["i"])))),
                        start=W.off + ls["i"], end=W.off + rs["i"], deg=DEG_UZ[deg], vol_ok=vol_ok, calc=calc, dir=-sign))
    return out


def _ls_line(pts: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    xs = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts], float)
    if len(pts) == 2:
        s = (ys[1] - ys[0]) / (xs[1] - xs[0])
        a = ys[0] - s * xs[0]
    else:
        s, a = np.polyfit(xs, ys, 1)
    return float(s), float(a), float(np.abs(ys - (a + s * xs)).max())


def _cls_lines(sn_u: float, sn_l: float, r: float) -> Optional[str]:
    flat = lambda x: abs(x) < FLAT_SN
    if r >= 0.80:
        return None
    if flat(sn_u) and sn_l > FLAT_SN:
        return "asc_triangle"
    if flat(sn_l) and sn_u < -FLAT_SN:
        return "desc_triangle"
    if sn_u < -FLAT_SN * 0.5 and sn_l > FLAT_SN * 0.5:
        return "sym_triangle"
    if sn_u > FLAT_SN and sn_l > FLAT_SN:
        return "rising_wedge"
    if sn_u < -FLAT_SN and sn_l < -FLAT_SN:
        return "falling_wedge"
    return None


def _pat_lines(W: _W, P: List[Dict[str, Any]], theta: float, deg: str) -> List[Dict[str, Any]]:
    """Yaqinlashuvchi ikki chiziq: uchburchaklar (oʻsuvchi/pasayuvchi/simmetrik) va ponalar. Chiziqlar tasdiqlangan pivotlar orqali
    (eng kichik kvadratlar), kamida 5 ta tasdiqlangan teginish (har chiziqda >= 2), yopilishlarning >= 90% i chiziqlar ichida, kenglik nisbati r = w_oxir/w_boshi < 0.8.
    Tasdiq: chiziqdan 0.5·ATR ortiq YOPILISH. Maqsad: boshlanishdagi kenglik (measured move) breakout nuqtasiga qoʻshiladi."""
    out: List[Dict[str, Any]] = []
    m = len(P)
    tol_f = max(0.6 * W.atr_l, 0.012)
    for k in range(5, min(10, m) + 1):
        S = P[m - k:]
        conf = [p for p in S if p["ok"]]
        Hh = [p for p in conf if p["k"] == 1]
        Ll = [p for p in conf if p["k"] == -1]
        if len(Hh) < 2 or len(Ll) < 2 or len(conf) < 5:
            continue
        i0, i_end = S[0]["i"], conf[-1]["i"]
        if i_end - i0 < 12 or W.n - 1 - i0 > 170:
            continue
        su, au, ru = _ls_line([(p["i"], p["y"]) for p in Hh])
        sl, al, rl = _ls_line([(p["i"], p["y"]) for p in Ll])
        if max(ru, rl) > 1.5 * tol_f:
            continue
        up_at = lambda i: au + su * i
        lo_at = lambda i: al + sl * i
        w0, w_end = up_at(i0) - lo_at(i0), up_at(i_end) - lo_at(i_end)
        if w0 < 1.5 * W.atr_l or w_end < 0.4 * W.atr_l:
            continue
        r = w_end / w0
        sn_u, sn_l = su * math.sqrt(20) / W.sig, sl * math.sqrt(20) / W.sig
        kind = _cls_lines(sn_u, sn_l, r)
        if kind is None:
            continue
        xs = np.arange(i0, i_end + 1)
        inside = (W.yc[i0:i_end + 1] >= lo_at(xs) - 0.5 * W.atr_l) & (W.yc[i0:i_end + 1] <= up_at(xs) + 0.5 * W.atr_l)
        if inside.mean() < 0.90:
            continue
        x_ap = (al - au) / (su - sl) if abs(su - sl) > 1e-12 else 1e9
        apex_in = x_ap - (W.n - 1)
        if apex_in > 150:
            continue
        brk = None
        for j in range(i_end + 1, W.n):
            if W.yc[j] > up_at(j) + BRK_ATR * W.atr_l:
                brk = (j, 1)
                break
            if W.yc[j] < lo_at(j) - BRK_ATR * W.atr_l:
                brk = (j, -1)
                break
        if brk is None:
            if (not (lo_at(W.n - 1) - BRK_ATR * W.atr_l <= W.yc[-1] <= up_at(W.n - 1) + BRK_ATR * W.atr_l) or apex_in < 0
                    or W.n - 1 - i_end > 30):
                continue
            state, days, bdir = PAT_STATE_FORM, None, PAT_BIAS[kind]
        else:
            if brk[0] - i_end > 45:
                continue
            days = W.n - 1 - brk[0]
            bdir = brk[1]
            ln_b = up_at if bdir > 0 else lo_at
            if days > 25 or (W.yc[-1] - ln_b(W.n - 1)) * bdir < -BRK_ATR * W.atr_l:
                continue
            state = PAT_STATE_OK
        h0 = max(w0, w_end)
        i_ref = brk[0] if brk else W.n - 1
        zt_up, zt_dn = up_at(i_ref) + h0, lo_at(i_ref) - h0
        main_dir = bdir if bdir != 0 else 1
        zt = zt_up if main_dir > 0 else zt_dn
        trig_up, trig_dn = up_at(W.n - 1) + BRK_ATR * W.atr_l, lo_at(W.n - 1) - BRK_ATR * W.atr_l
        touches = len(conf) + (1 if S[-1]["ok"] is False and abs(S[-1]["y"] - (up_at(S[-1]["i"]) if S[-1]["k"] == 1 else lo_at(S[-1]["i"]))) <= tol_f else 0)
        q_touch = min(1.0, (touches - 3) / 3.0)
        q_fit = max(0.0, 1.0 - (ru + rl) / (2.0 * tol_f))
        q_shape = _trap(r, 0.05, 0.25, 0.65, 0.85)
        q = 0.4 * q_touch + 0.3 * q_fit + 0.3 * q_shape
        if q < 0.30:
            continue
        calc = [f"Yuqori chiziq {len(Hh)} ta, pastki chiziq {len(Ll)} ta tasdiqlangan pivot orqali (eng kichik kvadratlar). Qiyaliklar (σ-birlikda, 20 bar): "
                f"yuqori {sn_u:+.2f}, pastki {sn_l:+.2f} (|·| < {FLAT_SN:.2f} — yassi).",
                f"Kenglik: boshida {(math.exp(w0) - 1) * 100:.1f}%, oxirida {(math.exp(w_end) - 1) * 100:.1f}% (nisbat {r:.2f}; < 0.8 — yaqinlashuv).",
                (f"Chiziqlar kesishishi (apex) ≈ {apex_in:.0f} bardan keyin." if apex_in >= 0 else "Chiziqlar allaqachon kesishgan."),
                f"Maqsad = breakout nuqtasidagi chiziq ± boshlangʻich kenglik ({_usd2(math.exp(up_at(i0)) - math.exp(lo_at(i0)))} ≈ {_usd2(math.exp(up_at(W.n - 1)) * (math.exp(h0) - 1))}).",
                f"Tasdiq: yuqoriga — {_usd2(math.exp(trig_up))} dan yuqori yopilish (maqsad {_usd2(math.exp(zt_up))}); pastga — {_usd2(math.exp(trig_dn))} dan past "
                f"yopilish (maqsad {_usd2(math.exp(zt_dn))})."]
        inv_y = (up_at(W.n - 1) if bdir > 0 else lo_at(W.n - 1)) if brk else (lo_at(W.n - 1) if main_dir > 0 else up_at(W.n - 1))
        calc.append((f"Bekor qilish: narx {_usd2(math.exp(inv_y))} (buzilgan chiziqning hozirgi qiymati) dan qaytib {'pastga' if bdir > 0 else 'yuqoriga'} yopilsa — breakout yolgʻon boʻlgan." if brk
                     else f"Bekor qilish: narx {_usd2(math.exp(inv_y))} (qarama-qarshi chiziqning hozirgi qiymati) dan tashqariga yopilsa, kutilayotgan yoʻnalish buziladi."))
        item = dict(kind=kind, name=PAT_UZ[kind], bias=PAT_BIAS[kind], state=state, days_since=days, quality=float(min(1.0, q)),
                    pts=[_pt(W, p["i"], p["y"], "") for p in S],
                    lines=[_seg(W, i0, up_at(i0), W.n - 1 + 15, up_at(W.n - 1 + 15), "upper"), _seg(W, i0, lo_at(i0), W.n - 1 + 15, lo_at(W.n - 1 + 15), "lower")],
                    trigger=float(math.exp(trig_up if main_dir > 0 else trig_dn)), trigger_up=float(math.exp(trig_up)), trigger_dn=float(math.exp(trig_dn)),
                    target=float(math.exp(zt)), target_up=float(math.exp(zt_up)), target_dn=float(math.exp(zt_dn)),
                    invalid=float(math.exp(inv_y)),
                    height=float(math.exp(up_at(W.n - 1)) * (math.exp(h0) - 1)), start=W.off + i0, end=W.off + W.n - 1,
                    deg=DEG_UZ[deg], vol_ok=True, calc=calc, dir=main_dir, apex_in=float(apex_in), touches=int(touches), ratio=float(r))
        if brk:
            item["broke_i"] = W.off + brk[0]
        out.append(item)
    return out


def _pat_flags(W: _W) -> List[Dict[str, Any]]:
    """Bayroq / vimpel (davom etish): keskin "ustun" (pole, >= 3σ·√davomiylik va >= 10%) + qarama-qarshi yoʻnalishda tor konsolidatsiya
    (<= ustunning 50% i), 7–28 bar. Bayroq — parallel kanal (r 0.55–1.3), vimpel — yaqinlashuvchi (r 0.2–0.55). Maqsad: ustun uzunligi breakout nuqtasiga qoʻshiladi."""
    out: List[Dict[str, Any]] = []
    n = W.n
    for f in (7, 10, 14, 20, 28):
        for lag in (0, 1, 2, 3):
            e = n - 1 - lag
            s0 = e - f + 1
            if s0 < 40 or e >= n:
                continue
            xs = np.arange(f, dtype=float)
            yhw, ylw, ycw = W.yh[s0:e + 1], W.yl[s0:e + 1], W.yc[s0:e + 1]
            bu, au = np.polyfit(xs, yhw, 1)
            bl, al = np.polyfit(xs, ylw, 1)
            up, lo = au + bu * xs, al + bl * xs
            w_s, w_e = au - al, (au + bu * (f - 1)) - (al + bl * (f - 1))
            if w_s <= 0 or w_e <= 0:
                continue
            r = w_e / w_s
            if not (0.2 <= r <= 1.3):
                continue
            if np.mean((ycw >= lo - 0.5 * W.atr_l) & (ycw <= up + 0.5 * W.atr_l)) < 0.9 or np.mean(np.abs(yhw - up)) > 0.9 * W.atr_l:
                continue
            for dr in (1, -1):
                if dr > 0:
                    ti = max(0, s0 - 3) + int(np.argmax(W.yh[max(0, s0 - 3):s0 + 4]))
                    a0 = max(0, ti - 25)
                    if ti - a0 < 3:
                        continue
                    si = a0 + int(np.argmin(W.yl[a0:ti + 1]))
                    pole = float(W.yh[ti] - W.yl[si])
                    extreme_in = float(W.yl[s0:e + 1].min())
                    retr = (W.yh[ti] - extreme_in) / pole if pole > 0 else 9.0
                    slope_ok = bu <= 0.1 * W.sig and bl <= 0.1 * W.sig
                else:
                    ti = max(0, s0 - 3) + int(np.argmin(W.yl[max(0, s0 - 3):s0 + 4]))
                    a0 = max(0, ti - 25)
                    if ti - a0 < 3:
                        continue
                    si = a0 + int(np.argmax(W.yh[a0:ti + 1]))
                    pole = float(W.yh[si] - W.yl[ti])
                    extreme_in = float(W.yh[s0:e + 1].max())
                    retr = (extreme_in - W.yl[ti]) / pole if pole > 0 else 9.0
                    slope_ok = bu >= -0.1 * W.sig and bl >= -0.1 * W.sig
                dp = ti - si
                if dp < 3 or dp > 25 or pole < 0.10 or pole / (W.sig * math.sqrt(dp)) < 3.0 or retr > 0.5 or not slope_ok:
                    continue
                z_p = pole / (W.sig * math.sqrt(dp))
                line = (lambda i: au + bu * (i - s0)) if dr > 0 else (lambda i: al + bl * (i - s0))
                brk = next((j for j in range(e + 1, n) if (W.yc[j] - line(j)) * dr > BRK_ATR * W.atr_l), None)
                if brk is None:
                    if (dr > 0 and W.yc[-1] < (al + bl * (n - 1 - s0)) - BRK_ATR * W.atr_l) or (dr < 0 and W.yc[-1] > (au + bu * (n - 1 - s0)) + BRK_ATR * W.atr_l):
                        continue
                    state, days = PAT_STATE_FORM, None
                else:
                    state, days = PAT_STATE_OK, n - 1 - brk
                kind = ("bull_flag" if dr > 0 else "bear_flag") if r >= 0.55 else ("pennant_up" if dr > 0 else "pennant_dn")
                i_ref = brk if brk is not None else n - 1
                lvl_now = line(n - 1)
                zt = line(i_ref) + dr * pole
                trig = lvl_now + dr * BRK_ATR * W.atr_l
                v_pole = float(np.mean(W.V[si:ti + 1]))
                v_flag = float(np.mean(W.V[s0:e + 1]))
                vol_ok = bool(v_flag < v_pole)
                q = (0.4 * min(1.0, 0.5 + (z_p - 3.0) / 4.0) + 0.3 * max(0.0, 1.0 - retr / 0.5) + 0.15 * (1.0 if vol_ok else 0.4)
                     + 0.15 * max(0.0, 1.0 - float(np.mean(np.abs(yhw - up)) + np.mean(np.abs(ylw - lo))) / (1.6 * W.atr_l)))
                inval = line(n - 1) if brk is not None else ((al + bl * (n - 1 - s0)) if dr > 0 else (au + bu * (n - 1 - s0)))
                calc = [f"Ustun (pole): {W.date(si):%Y-%m-%d} {_usd2(math.exp(W.yl[si] if dr > 0 else W.yh[si]))} → {W.date(ti):%Y-%m-%d} "
                        f"{_usd2(math.exp(W.yh[ti] if dr > 0 else W.yl[ti]))}: {(math.exp(pole) - 1) * 100:.0f}% {dp} barda ({z_p:.1f}σ — tasodifiy yurish uchun odatiy emas).",
                        f"Konsolidatsiya {f} bar, ustunning {retr * 100:.0f}% i qadar qaytgan (ruxsat ≤ 50%); kanal kengligi nisbati {r:.2f}"
                        + (" (parallel — bayroq)." if r >= 0.55 else " (yaqinlashuvchi — vimpel)."),
                        f"Maqsad = breakout darajasi ({_usd2(math.exp(line(i_ref)))}) ± ustun uzunligi = {_usd2(math.exp(zt))}.",
                        f"Tasdiq: {'yuqoriga' if dr > 0 else 'pastga'} {_usd2(math.exp(trig))} dan {'yuqori' if dr > 0 else 'past'} yopilish; bekor: " + (f"narx buzilgan chegara ({_usd2(math.exp(inval))}) ostiga/ustiga qaytib yopilsa." if brk is not None else f"kanalning qarama-qarshi chegarasi {_usd2(math.exp(inval))}."),
                        "Konsolidatsiyada hajm " + ("ustundagidan past (klassik belgi)." if vol_ok else "ustundagidan past emas.")]
                it = dict(kind=kind, name=PAT_UZ[kind], bias=PAT_BIAS[kind] if dr > 0 else -1, state=state, days_since=days, quality=float(min(1.0, q)),
                          pts=[_pt(W, si, W.yl[si] if dr > 0 else W.yh[si], "pole"), _pt(W, ti, W.yh[ti] if dr > 0 else W.yl[ti], "")],
                          lines=[_seg(W, s0, au, n - 1 + 12, au + bu * (n - 1 + 12 - s0), "upper"), _seg(W, s0, al, n - 1 + 12, al + bl * (n - 1 + 12 - s0), "lower"),
                                 _seg(W, si, W.yl[si] if dr > 0 else W.yh[si], ti, W.yh[ti] if dr > 0 else W.yl[ti], "pole")],
                          trigger=float(math.exp(trig)), target=float(math.exp(zt)), invalid=float(math.exp(inval)), height=float(math.exp(W.yc[-1]) * (math.exp(pole) - 1)),
                          start=W.off + si, end=W.off + n - 1, deg="kichik", vol_ok=vol_ok, calc=calc, dir=dr, ratio=float(r), z_pole=float(z_p))
                if brk is not None:
                    it["broke_i"] = W.off + brk
                out.append(it)
    return out


def find_patterns(df: pd.DataFrame, T: Dict[str, Any]) -> Dict[str, Any]:
    """Barcha chart-patternlar (ikkilamchi choʻqqi/tub, bosh-yelkalar, uchburchak/pona, bayroq/vimpel) — hozirgi grafikka tegishlilari,
    ahamiyati (sifat × yangilik) boʻyicha tartiblangan."""
    W = _W(df, T)
    items: List[Dict[str, Any]] = []
    for deg in ("minor", "mid"):
        P, th = T["piv"][deg], T["thr"][deg]
        for sign in (1, -1):
            items += _pat_double(W, P, sign, th, deg)
            items += _pat_hs(W, P, sign, th, deg)
        items += _pat_lines(W, P, th, deg)
    items += _pat_flags(W)
    items = [x for x in items if x["quality"] >= 0.35]
    # takrorlarni olib tashlash: bir xil turdagi va bir-biriga yaqin boshlanishli patternlardan eng sifatlisi
    items.sort(key=lambda x: -x["quality"])
    keep: List[Dict[str, Any]] = []
    conv = ("asc_triangle", "desc_triangle", "sym_triangle", "rising_wedge", "falling_wedge")
    rev = ("double_top", "double_bottom", "hs_top", "hs_bottom")

    def overlap(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        inter = min(a["end"], b["end"]) - max(a["start"], b["start"])
        return inter / max(1.0, min(a["end"] - a["start"], b["end"] - b["start"]))
    for x in items:
        dup = any(y["kind"] == x["kind"] and overlap(x, y) >= 0.5 for y in keep)
        sub = x["kind"] in conv and any(y["kind"] in rev and y["quality"] >= x["quality"] - 0.1 and overlap(x, y) >= 0.5 for y in keep)
        if not dup and not sub:
            keep.append(x)

    def rel(x: Dict[str, Any]) -> float:
        rec = 1.0 if (x["state"] == PAT_STATE_OK and (x["days_since"] or 0) <= 10) else 0.85 if x["state"] == PAT_STATE_FORM else 0.6
        return x["quality"] * rec
    keep.sort(key=lambda x: -rel(x))
    return dict(ok=True, items=keep[:6], n_found=len(keep))

# --------------------------------------------------------------------------------------
# Elliott toʻlqinlari: qatʼiy qoidalar bilan tekshiriladigan mexanik sanoq
# --------------------------------------------------------------------------------------
# Elliott sanogʻi subyektiv: bir grafikka bir nechta sanoq mos kelishi mumkin. Bu yerda faqat QOIDAGA mos (hard rules) nomzodlar olinadi,
# Fibonachchi nisbatlariga moslik 0–1 ball bilan oʻlchanadi va natija tasodifiy yurishdan olingan taqsimot bilan solishtiriladi
# ("tasodifiy narxda shunday yoki yaxshiroq moslik oynalarning necha foizida chiqadi"). Natija — bitta variant + muqobil, bashorat emas.
EW_STATE_UZ = {
    ("i", 3, 0): "2-toʻlqin (tuzatish) davom etmoqda — 3-toʻlqin boshlanishi kutiladi",
    ("i", 4, 0): "3-toʻlqin davom etmoqda",
    ("i", 5, 0): "4-toʻlqin (tuzatish) davom etmoqda — 5-toʻlqin kutiladi",
    ("i", 6, 0): "5-toʻlqin davom etmoqda / yakunlanmoqda",
    ("i", 6, 1): "5-toʻlqin yakunlandi, A-toʻlqin (tuzatish) boshlandi",
    ("i", 6, 2): "5-toʻlqin yakunlandi, B-toʻlqin (tuzatish ichida)",
    ("abc", 4, 0): "ABC tuzatishning C-toʻlqini davom etmoqda",
    ("abc", 4, 1): "ABC tuzatish tugagan boʻlishi mumkin — asosiy trend davomi kutiladi",
}


def _gauss_fib(x: float, targets: Tuple[float, ...], s: float = 0.2) -> float:
    if not np.isfinite(x) or x <= 0:
        return 0.0
    return float(max(math.exp(-0.5 * (math.log(x / t) / s) ** 2) for t in targets))


def _impulse_fit(z: List[float], W: _W, ix: List[int]) -> Optional[Dict[str, Any]]:
    """z — pivotlarning yoʻnalishga normallangan (impuls yuqoriga) log-narxi P0..P(k−1), k = 3..6. Qatʼiy qoidalar buzilsa None."""
    k = len(z)
    w1 = z[1] - z[0]
    if w1 <= 0:
        return None
    r2 = (z[1] - z[2]) / w1
    if not (0.0 < r2 < 1.0):                                   # 1-qoida: 2-toʻlqin 1-toʻlqinni 100% dan ortiq qaytarmaydi
        return None
    comps: Dict[str, Tuple[float, float]] = {"r2": (_trap(r2, 0.15, 0.382, 0.786, 0.95), 0.25)}
    ratios: Dict[str, float] = {"r2": r2}
    w3 = w5 = float("nan")
    if k >= 4:
        w3 = z[3] - z[2]
        if z[3] <= z[1]:                                       # 3-toʻlqin 1-toʻlqin tepasidan oshishi shart
            return None
        e3 = w3 / w1
        comps["e3"] = (_trap(e3, 0.6, 1.0, 2.618, 4.236) if k == 4 else _trap(e3, 1.0, 1.382, 2.618, 4.236), 0.30)
        ratios["e3"] = e3
    if k >= 5:
        r4 = (z[3] - z[4]) / w3
        if z[4] <= z[1]:                                       # 2-qoida: 4-toʻlqin 1-toʻlqin hududiga kirmaydi (ustma-ust tushmaydi)
            return None
        comps["r4"] = (_trap(r4, 0.10, 0.236, 0.5, 0.7), 0.20)
        comps["alt"] = (min(1.0, abs(r2 - r4) / 0.15), 0.05)   # alternatsiya: 2- va 4-toʻlqin xarakteri har xil
        ratios["r4"] = r4
    if k >= 6:
        w5 = z[5] - z[4]
        if z[5] <= z[3] or w3 < min(w1, w5):                   # 3-qoida: 3-toʻlqin eng qisqa boʻlmaydi; 5-toʻlqin 3-toʻlqin tepasidan oshadi
            return None
        e5 = w5 / w1
        comps["e5"] = (_gauss_fib(e5, (0.618, 1.0, 1.618)), 0.20)
        ratios["e5"] = e5
    wsum = sum(w for _, w in comps.values())
    score = sum(s * w for s, w in comps.values()) / wsum
    vol = {}
    if k >= 4:
        v1 = float(np.mean(W.V[ix[0]:ix[1] + 1]))
        v3 = float(np.mean(W.V[ix[2]:ix[3] + 1]))
        vol["v3_v1"] = v3 / v1 if v1 > 0 else float("nan")
        if k >= 6:
            v5 = float(np.mean(W.V[ix[4]:ix[5] + 1]))
            vol["v3_v5"] = v3 / v5 if v5 > 0 else float("nan")
    return dict(score=float(score), ratios=ratios, comps={a: float(b[0]) for a, b in comps.items()}, vol=vol, w1=w1, w3=w3, w5=w5)


def _abc_fit(z: List[float]) -> Optional[Dict[str, Any]]:
    """z — (Pa, Pb, Pc, Pd) oldingi trend yuqoriga boʻlgan holatga normallangan: A pastga, B yuqoriga, C pastga."""
    a_ = z[0] - z[1]
    if a_ <= 0:
        return None
    rb = (z[2] - z[1]) / a_
    rc = (z[2] - z[3]) / a_
    if not (0.2 < rb < 1.1) or z[3] >= z[1] or not (0.5 < rc < 2.0):
        return None
    f_rb = _trap(rb, 0.2, 0.382, 0.886, 1.1)
    f_rc = _gauss_fib(rc, (0.618, 1.0, 1.272, 1.618), 0.18)
    return dict(score=float(0.5 * f_rb + 0.5 * f_rc), ratios=dict(rb=rb, rc=rc), comps=dict(rb=f_rb, rc=f_rc), vol={}, w1=a_, w3=float("nan"), w5=float("nan"))


def _null_exceed(fam: str, s: float) -> float:
    """Tasodifiy yurishda eng yaxshi moslik >= s boʻlish ehtimoli (ELLIOTT_NULL jadvalidan, chiziqli interpolyatsiya)."""
    tab = ELLIOTT_NULL.get(fam)
    if not tab:
        return float("nan")
    xs, ps = ELLIOTT_NULL_X, tab
    if s <= xs[0]:
        return float(ps[0])
    if s >= xs[-1]:
        return float(ps[-1])
    return float(np.interp(s, xs, ps))


def _ew_candidates(W: _W, T: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for deg in ("minor", "mid", "major"):
        P = T["piv"][deg]
        m = len(P)
        for dr in (1, -1):
            for k in (3, 4, 5, 6):
                for off in ((0, 1, 2) if k == 6 else (0,)):
                    end = m - 1 - off
                    st = end - k + 1
                    if st < 0:
                        continue
                    seq = P[st:end + 1]
                    if seq[0]["k"] != -dr:
                        continue
                    z = [dr * q["y"] for q in seq]
                    later = P[end + 1:]
                    if later and any(dr * q["y"] >= z[-1] for q in later if q["k"] == dr):       # keyingi yuqorilar impuls tepasidan oshmasligi kerak
                        continue
                    f = _impulse_fit(z, W, [q["i"] for q in seq])
                    if f:
                        out.append(dict(fam="i%d" % k, kind="i", k=k, off=off, dir=dr, deg=deg, seq=seq, z=z, fit=f))
            for off in (0, 1):                                              # ABC tuzatish (zigzag/flat)
                end = m - 1 - off
                st = end - 3
                if st < 1:
                    continue
                seq = P[st:end + 1]
                dirp = seq[0]["k"]                                           # oldingi trend yoʻnalishi: Pa yuqori boʻlsa — yuqoriga
                if dirp != dr:
                    continue
                z = [dirp * q["y"] for q in seq]
                later = P[end + 1:]
                if later and any(dirp * q["y"] >= z[2] for q in later if q["k"] == dirp):
                    continue
                f = _abc_fit(z)
                if f:
                    out.append(dict(fam="abc", kind="abc", k=4, off=off, dir=dirp, deg=deg, seq=seq, z=z, fit=f))
    return out


def _ew_targets(c: Dict[str, Any], W: _W) -> Tuple[List[Dict[str, Any]], float, str, int]:
    """Holatga qarab maqsadlar (Fibonachchi), bekor qilish darajasi va yoʻnalish (+1 yuqoriga, −1 pastga) ni hisoblaydi."""
    dr, z, k, kind, off = c["dir"], c["z"], c["k"], c["kind"], c["off"]
    px = lambda zz: float(math.exp(dr * zz))
    tg: List[Dict[str, Any]] = []
    if kind == "i":
        w1 = z[1] - z[0]
        if k == 3:
            for m_ in (1.0, 1.618, 2.618):
                tg.append(dict(name=f"3-toʻlqin {m_:g}×W1", price=px(z[2] + m_ * w1), rule=f"3-toʻlqin = {m_:g} × 1-toʻlqin ({_usd2(px(z[0]))}→{_usd2(px(z[1]))}) 2-toʻlqin tubidan"))
            inv, inv_t, bias = px(z[0]), "1-toʻlqin boshi (P0) dan past/yuqori chiqish", dr
        elif k == 4:
            w3 = z[3] - z[2]
            for m_ in (1.618, 2.618):
                lv = z[2] + m_ * w1
                if lv > z[3]:
                    tg.append(dict(name=f"3-toʻlqin {m_:g}×W1", price=px(lv), rule=f"3-toʻlqin = {m_:g} × 1-toʻlqin, 2-toʻlqin tubidan"))
            for r_ in (0.236, 0.382, 0.5):
                tg.append(dict(name=f"4-toʻlqin {r_:g} tuzatish", price=px(z[3] - r_ * w3), rule=f"kutilayotgan 4-toʻlqin tuzatishi = 3-toʻlqinning {r_ * 100:.1f}% i", pull=True))
            inv, inv_t, bias = px(z[2]), "3-toʻlqin boshi (P2) dan oshib qaytish", dr
        elif k == 5:
            for m_ in (0.618, 1.0, 1.618):
                tg.append(dict(name=f"5-toʻlqin {m_:g}×W1", price=px(z[4] + m_ * w1), rule=f"5-toʻlqin = {m_:g} × 1-toʻlqin, 4-toʻlqin tubidan"))
            inv, inv_t, bias = px(z[1]), "4-toʻlqin 1-toʻlqin tepasi (P1) hududiga kirishi (ustma-ust tushish)", dr
        else:
            tot = z[5] - z[0]
            for r_ in (0.382, 0.5, 0.618):
                tg.append(dict(name=f"Tuzatish {r_:g}", price=px(z[5] - r_ * tot), rule=f"butun impulsning {r_ * 100:.1f}% i qaytishi (ABC tuzatish zonasi)"))
            inv, inv_t, bias = px(z[5]), "impuls tepasi (P5) dan yangi ekstremum (5-toʻlqin choʻzilgan)", -dr
    else:
        a_ = z[0] - z[1]
        if off == 0:
            for m_ in (1.0, 1.618):
                tg.append(dict(name=f"C = {m_:g}×A", price=px(z[2] - m_ * a_), rule=f"C-toʻlqin = {m_:g} × A-toʻlqin, B tepasidan"))
            inv, inv_t, bias = px(z[2]), "B-toʻlqin tepasidan oshib qaytish", -dr
        else:
            tg.append(dict(name="B tepasi", price=px(z[2]), rule="tuzatishdan keyingi birinchi qarshilik: B-toʻlqin tepasi"))
            tg.append(dict(name="A boshi", price=px(z[0]), rule="oldingi trend tepasi (A boshi)"))
            inv, inv_t, bias = px(z[3]), "C-toʻlqin tubidan yangi ekstremum", dr
    return tg, float(inv), inv_t, int(bias)


def find_elliott(df: pd.DataFrame, T: Dict[str, Any]) -> Dict[str, Any]:
    """Eng mos Elliott sanogʻi (+ muqobil). Nomzodlar: 3–6 pivotli impuls (1-2-3-4-5) va ABC tuzatish, uch darajada, ikki yoʻnalishda;
    hammasi qatʼiy qoidalar (2-toʻlqin 100% dan kam qaytadi, 3-toʻlqin eng qisqa emas, 4-toʻlqin 1-toʻlqin hududiga kirmaydi) bilan filtrlanadi,
    Fibonachchi moslik bali 0–1, "tasodifiylik" — tasodifiy yurishda shu yoki yaxshiroq ball chiqish ehtimoli."""
    W = _W(df, T)
    cands = _ew_candidates(W, T)
    price = float(W.C[-1])
    for c in cands:
        c["p_rand"] = _null_exceed(c["fam"], c["fit"]["score"])
    cands = [c for c in cands if c["fit"]["score"] >= 0.45]

    def rank(c: Dict[str, Any]) -> Tuple[float, float, float, int]:
        p = c["p_rand"] if np.isfinite(c["p_rand"]) else 1.0
        grp = 0 if (c["kind"] == "i" and c["k"] >= 5 and p <= 0.10) else 1 if p <= 0.10 else 2      # ahamiyatli toʻliq impulslar birinchi
        return (grp, p, -c["fit"]["score"], -c["k"])
    cands.sort(key=rank)
    res: Dict[str, Any] = dict(ok=True, best=None, alts=[], n_valid=len(cands))
    deg_uz = {"minor": "kichik", "mid": "oʻrta", "major": "katta"}

    def pack(c: Dict[str, Any]) -> Dict[str, Any]:
        tg, inv, inv_t, bias = _ew_targets(c, W)
        dr, seq, z = c["dir"], c["seq"], c["z"]
        labels = ["0", "1", "2", "3", "4", "5"] if c["kind"] == "i" else ["A0", "A", "B", "C"]
        pv = [dict(i=W.off + q["i"], p=float(math.exp(q["y"])), label=labels[j], ok=bool(q["ok"])) for j, q in enumerate(seq)]
        f = c["fit"]
        rules: List[Tuple[str, bool, str]] = []
        r_ = f["ratios"]
        if c["kind"] == "i":
            rules.append(("2-toʻlqin 1-toʻlqinni 100% dan kam qaytaradi", True, f"qaytish {r_['r2'] * 100:.0f}% (ideal 38.2–78.6%)"))
            if c["k"] >= 4:
                rules.append(("3-toʻlqin 1-toʻlqin tepasidan oshadi", True, f"3-toʻlqin = {r_['e3']:.2f} × 1-toʻlqin (ideal 1.62–2.62)"))
            if c["k"] >= 5:
                rules.append(("4-toʻlqin 1-toʻlqin hududiga kirmaydi", True, f"4-toʻlqin tuzatishi {r_['r4'] * 100:.0f}% (ideal 23.6–50%)"))
            if c["k"] >= 6:
                rules.append(("3-toʻlqin eng qisqa emas; 5-toʻlqin 3-toʻlqin tepasidan oshadi", True, f"5-toʻlqin = {r_['e5']:.2f} × 1-toʻlqin (Fibonachchi: 0.62 / 1.0 / 1.62)"))
        else:
            rules.append(("B-toʻlqin A ni 110% dan kam qaytaradi", True, f"B = {r_['rb'] * 100:.0f}% A (ideal 38.2–88.6%)"))
            rules.append(("C-toʻlqin A tubidan oshadi", True, f"C = {r_['rc']:.2f} × A (Fibonachchi: 0.62 / 1.0 / 1.27 / 1.62)"))
        vol = f.get("vol", {})
        if "v3_v1" in vol and np.isfinite(vol["v3_v1"]):
            rules.append(("Hajm: 3-toʻlqinda 1-toʻlqindagidan kam emas (maslahat)", bool(vol["v3_v1"] >= 1.0), f"3/1 hajm = {vol['v3_v1']:.2f}×"))
        return dict(fam=c["fam"], kind=c["kind"], k=c["k"], off=c["off"], dir=dr, deg=deg_uz[c["deg"]], score=f["score"], p_rand=c["p_rand"],
                    p_fw=fw_adjust(EW_FW if (c["kind"] == "i" and c["k"] >= 5) else EW_ALL_FW, c["p_rand"]),
                    state=EW_STATE_UZ.get((c["kind"], c["k"], c["off"]), ""), bias=bias, pivots=pv, ratios=r_, rules=rules, targets=tg,
                    invalid=inv, invalid_text=inv_t, labels=labels, price=price)
    if cands:
        res["best"] = pack(cands[0])
        seen = {(cands[0]["dir"], cands[0]["deg"])}
        for c in cands[1:]:
            key = (c["dir"], c["deg"])
            if key in seen or (np.isfinite(c["p_rand"]) and c["p_rand"] > 0.5):
                continue
            seen.add(key)
            res["alts"].append(pack(c))
            if len(res["alts"]) == 2:
                break
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


# ======================================================================================
# 6b. TEXNIK BLOKLARNI YIGʻISH, NARX DARAJALARI VA QISQA XULOSA
# ======================================================================================
def compute_technicals(df: pd.DataFrame) -> Dict[str, Any]:
    """Barcha texnik chizma detektorlari (pivotlar bir marta hisoblanadi). Bitta detektor xatosi butun tahlilni toʻxtatmaydi."""
    T = compute_pivots(df)
    out: Dict[str, Any] = dict(T=T)
    for key, fn in (("trend", find_trendlines), ("fib", fib_levels), ("patterns", find_patterns), ("elliott", find_elliott)):
        try:
            out[key] = fn(df, T)
        except Exception as e:
            log(f"[texnik] {key}: {type(e).__name__}: {e}")
            out[key] = dict(ok=False, error=type(e).__name__)
    try:
        out["sd"] = find_sd_zones(df)
    except Exception as e:
        log(f"[texnik] sd: {type(e).__name__}: {e}")
        out["sd"] = dict(demand=[], supply=[], error=type(e).__name__)
    return out


def pat_p_rand(kind: str, q: float) -> float:
    """Tasodifiy yurishda shu turdagi pattern sifati >= q boʻlish ehtimoli (PAT_NULL jadvalidan)."""
    tab = PAT_NULL.get(kind)
    if not tab:
        return float("nan")
    return float(np.interp(q, PAT_NULL_X, tab))


def _lvl(out: List[Dict[str, Any]], price: float, lid: str, name: str, src: str, role: str, p0: float) -> None:
    """Daraja qoʻshadi. role: 'resist' | 'support' | 'target' | 'invalid'."""
    if price and np.isfinite(price) and price > 0 and abs(price / p0 - 1.0) > 0.004 and 0.35 < price / p0 < 3.0:
        out.append(dict(id=lid, name=name, price=float(price), src=src, role=role, up=bool(price > p0)))


def collect_levels(R: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Qisqa xulosa va Monte-Karlo uchun nomzod narx darajalari; har biri manbasi bilan (S/R, baza, pattern, kanal, Fibonachchi, Elliott, zona)."""
    p0 = float(R["price"])
    out: List[Dict[str, Any]] = []
    lv = R["levels"]
    for i, x in enumerate(lv["resistance"][:2]):
        _lvl(out, x["price"], f"R{i + 1}", f"Qarshilik R{i + 1}", "Support/Resistance", "resist", p0)
    for i, x in enumerate(lv["support"][:2]):
        _lvl(out, x["price"], f"S{i + 1}", f"Qoʻllab-quvvatlash S{i + 1}", "Support/Resistance", "support", p0)
    bs = R.get("base") or {}
    if bs.get("ok"):
        if bs["state"] in ("inside", "breakout"):
            for x in bs["targets"]:
                _lvl(out, x["price"], x["name"], f"Baza maqsadi {x['name']}", "Baza (measured move)", "target", p0)
        if bs["state"] == "inside":
            _lvl(out, bs["hi"], "BH", "Baza yuqorisi (breakout darajasi)", "Baza", "resist", p0)
            _lvl(out, bs["lo"], "BL", "Baza pasti", "Baza", "support", p0)
    tech = R.get("tech") or {}
    for j, it in enumerate(((tech.get("patterns") or {}).get("items") or [])[:3]):
        if it["quality"] < 0.5:
            continue
        _lvl(out, it["target"], f"P{j + 1}", f"{it['name']} maqsadi", "Chart-pattern", "target", p0)
        _lvl(out, it["invalid"], f"PI{j + 1}", f"{it['name']}: bekor qilish", "Chart-pattern", "invalid", p0)
        if it["state"] == PAT_STATE_FORM:
            _lvl(out, it["trigger"], f"PT{j + 1}", f"{it['name']}: tasdiq darajasi", "Chart-pattern", "resist" if it["trigger"] > p0 else "support", p0)
    tr = tech.get("trend") or {}
    ch = tr.get("channel")
    if ch:
        _lvl(out, ch["top_now"], "CT", "Kanal yuqori chegarasi", "Trend kanali", "resist", p0)
        _lvl(out, ch["bot_now"], "CB", "Kanal pastki chegarasi", "Trend kanali", "support", p0)
    for key, lid, nm, role in (("support", "TS", "Trend chizigʻi (qoʻllab-quvvatlash)", "support"), ("resistance", "TR", "Trend chizigʻi (qarshilik)", "resist")):
        ln = tr.get(key)
        if ln and not ln["broke"]:
            _lvl(out, ln["now"], lid, nm, "Trend chizigʻi", role, p0)
    fb = tech.get("fib") or {}
    if fb.get("ok"):
        for r_, pr in fb["levels"]:
            _lvl(out, pr, f"F{r_:g}", f"Fibonachchi {r_:g}", "Fibonachchi tuzatish", "resist" if pr > p0 else "support", p0)
        for e_, pr in fb["ext"]:
            if (fb["up"] and pr > p0) or ((not fb["up"]) and pr < p0):
                _lvl(out, pr, f"FE{e_:g}", f"Fibonachchi kengayish {e_:g}", "Fibonachchi kengayish", "target", p0)
    ew = (tech.get("elliott") or {}).get("best")
    if ew and np.isfinite(ew["p_fw"]) and ew["p_fw"] <= 0.25:
        for j, t_ in enumerate(ew["targets"]):
            _lvl(out, t_["price"], f"E{j + 1}", f"Elliott: {t_['name']}", "Elliott", "target", p0)
        _lvl(out, ew["invalid"], "EI", "Elliott: bekor qilish", "Elliott", "invalid", p0)
    sd = tech.get("sd") or {}
    for i, z in enumerate(sd.get("demand", [])):
        _lvl(out, z["hi"], f"D{i + 1}", f"Talab zonasi {i + 1}", "Talab/taklif zonasi", "support", p0)
    for i, z in enumerate(sd.get("supply", [])):
        _lvl(out, z["lo"], f"Z{i + 1}", f"Taklif zonasi {i + 1}", "Talab/taklif zonasi", "resist", p0)
    _lvl(out, lv["hi52"], "H52", "52 haftalik yuqori", "52 hafta", "resist", p0)
    _lvl(out, lv["lo52"], "L52", "52 haftalik past", "52 hafta", "support", p0)
    seen, uniq = set(), []
    for x in out:
        if x["id"] not in seen:
            seen.add(x["id"])
            uniq.append(x)
    return uniq


def _cluster(levels: List[Dict[str, Any]], tol: float = 0.015) -> List[Dict[str, Any]]:
    """Bir-biriga <= tol yaqin darajalarni birlashtiradi (konfluens: bir nechta mustaqil manba bir joyni koʻrsatsa — ishonchliroq)."""
    cl: List[Dict[str, Any]] = []
    for x in sorted(levels, key=lambda z: z["price"]):
        if cl and x["price"] / cl[-1]["hi"] - 1.0 <= tol:
            cl[-1]["items"].append(x)
            cl[-1]["hi"] = x["price"]
        else:
            cl.append(dict(items=[x], lo=x["price"], hi=x["price"]))
    out = []
    for c in cl:
        its = c["items"]
        tg = [i for i in its if i["role"] == "target"]
        base = tg[0] if tg else its[0]
        srcs = sorted(set(i["src"] for i in its))
        out.append(dict(price=float(np.mean([i["price"] for i in its])), ids=[i["id"] for i in its], src=srcs, n_src=len(srcs),
                        names=[i["name"] for i in its], is_target=bool(tg), lead=base["name"], role=base["role"]))
    return out


def short_conclusion(R: Dict[str, Any]) -> Dict[str, Any]:
    """Hisobot oxiridagi qisqa xulosa: yoʻnalish, qaysi narx darajalarigacha koʻtarilishi (yoki tushishi) mumkinligi va bekor qilish darajasi.
    Hamma daraja aniq hisobdan olinadi (qarshilik, baza maqsadi, pattern maqsadi, Fibonachchi, Elliott, kanal) — taxmin emas; yoniga Monte-Karlo
    bilan 3 oyda (60 savdo kuni) shu darajaga yetish ehtimoli qoʻyiladi."""
    p0 = float(R["price"])
    con = R["conclusion"]
    lab = con["label"]
    lv = R.get("lvls") or []
    mc_l = {x["name"]: x["touch"] for x in (R.get("mc") or {}).get("levels", [])}

    def touch(cl: Dict[str, Any], h: int = 60) -> float:
        vals = [mc_l.get(i, {}).get(h) for i in cl["ids"]]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")
    above = _cluster([x for x in lv if x["up"] and x["role"] in ("resist", "target")])
    below = _cluster([x for x in lv if (not x["up"]) and x["role"] in ("support", "target")])
    for c in above + below:
        c["pct"] = c["price"] / p0 - 1.0
        c["p60"], c["p20"], c["p125"] = touch(c, 60), touch(c, 20), touch(c, 125)
    up_t = [c for c in above if c["is_target"] and c["pct"] <= 0.6]
    dn_t = [c for c in below if c["is_target"] and c["pct"] >= -0.6]
    ladder_up: List[Dict[str, Any]] = []
    if above:
        ladder_up.append(dict(above[0], tag="Birinchi toʻsiq"))
    cand = [c for c in up_t if c is not above[0]] if above else []
    if cand:
        ladder_up.append(dict(cand[0], tag="Asosiy maqsad"))
        if len(cand) > 1 and cand[-1]["price"] / cand[0]["price"] - 1 > 0.03:
            ladder_up.append(dict(cand[-1], tag="Uzoq maqsad"))
    elif len(above) > 1:
        ladder_up.append(dict(above[1], tag="Keyingi qarshilik"))
    ladder_dn: List[Dict[str, Any]] = []
    if below:
        ladder_dn.append(dict(below[-1], tag="Birinchi tayanch"))
        far = [c for c in dn_t if c is not below[-1]]
        if far:
            ladder_dn.append(dict(far[-1] if len(far) > 1 else far[0], tag="Pasayish maqsadi"))
        elif len(below) > 1:
            ladder_dn.append(dict(below[-2], tag="Keyingi tayanch"))
    inv = [x for x in lv if x["role"] == "invalid"]
    inv_dn = sorted([x for x in inv if not x["up"]], key=lambda x: p0 - x["price"])
    inv_up = sorted([x for x in inv if x["up"]], key=lambda x: x["price"] - p0)
    sup1 = below[-1] if below else None
    res1 = above[0] if above else None
    if lab == "Ijobiy":
        inv_p = inv_dn[0]["price"] if inv_dn and (sup1 is None or inv_dn[0]["price"] >= sup1["price"] * 0.97) else (sup1["price"] if sup1 else None)
        inv_why = (inv_dn[0]["name"] if inv_p is not None and inv_dn and inv_p == inv_dn[0]["price"] else "eng yaqin tayanch") if inv_p else ""
    elif lab == "Salbiy":
        inv_p = inv_up[0]["price"] if inv_up else (res1["price"] if res1 else None)
        inv_why = (inv_up[0]["name"] if inv_up else "eng yaqin qarshilik") if inv_p else ""
    else:
        inv_p, inv_why = None, ""
    rows = R.get("mc", {}).get("rows", [])
    r60 = next((r for r in rows if r["h"] == 60), None)
    return dict(label=lab, score=con["score"], up=ladder_up, down=ladder_dn, invalid=(dict(price=inv_p, why=inv_why) if inv_p else None),
                p_up_3m=(r60["p_up"] if r60 else float("nan")), support1=sup1, resist1=res1, n_levels=len(lv))


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
    elif R.get("lite"):
        add("AI (mashinaviy oʻqitish)", 0.0, 0.0, "Yengil rejim: bu aksiya AI universumiga kirmaydi — AI ball, fundamental va yangiliklar hisoblanmagan; "
            "xulosa faqat texnik qoidalarga asoslangan.", counted=False, kind="ai")
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
    tech = R.get("tech") or {}
    tr_ = tech.get("trend") or {}
    sl_, rl_ = tr_.get("support"), tr_.get("resistance")
    if sl_ and sl_["broke"] and (sl_["days_since"] if sl_["days_since"] is not None else 99) <= 10 and sl_["touches"] >= 3:
        add("Trend chizigʻi", -0.5, 0.25, f"Qoʻllab-quvvatlash trend chizigʻi ({sl_['touches']} teginish) {sl_['days_since']} savdo kuni oldin pastga buzildi.")
    elif rl_ and rl_["broke"] and (rl_["days_since"] if rl_["days_since"] is not None else 99) <= 10 and rl_["touches"] >= 3:
        add("Trend chizigʻi", 0.5, 0.25, f"Qarshilik trend chizigʻi ({rl_['touches']} teginish) {rl_['days_since']} savdo kuni oldin yuqoriga buzildi.")
    pts_ = (tech.get("patterns") or {}).get("items") or []
    if pts_:
        def _sg(x: Dict[str, Any]) -> int:
            return int(x["bias"] if x["bias"] != 0 else (x["dir"] if x["state"] == PAT_STATE_OK else 0))
        elig = [x for x in pts_ if x["quality"] >= 0.6 and _sg(x) != 0 and np.isfinite(pat_p_rand(x["kind"], x["quality"]))]
        pt = min(elig, key=lambda x: pat_p_rand(x["kind"], x["quality"])) if elig else pts_[0]
        pr = pat_p_rand(pt["kind"], pt["quality"])
        pfw = fw_adjust(PAT_FW, pr)
        sg_ = _sg(pt)
        cnt_ = bool(pt["quality"] >= 0.6 and np.isfinite(pfw) and pfw <= 0.05 and sg_ != 0)
        base_t = (f"{pt['name']} ({pt['state']}, sifat {pt['quality']:.2f}); tasodifiy narxda shu turdagi pattern shunday yoki yaxshiroq sifat bilan ~{pr * 100:.1f}% oynalarda chiqadi, "
                  f"barcha pattern turlari ichidan eng yaxshisini tanlashni hisobga olganda — {_p0(pfw)}")
        if cnt_:
            add("Chart-pattern", sg_ * (0.6 if pt["state"] == PAT_STATE_OK else 0.25), 0.5, base_t + ".")
        else:
            add("Chart-pattern", 0.0, 0.0, base_t + " — tasodifiydan ajralib turmaydi, shuning uchun xulosaga QOʻSHILMADI.", counted=False)
    ew_ = (tech.get("elliott") or {}).get("best")
    if ew_:
        pe = ew_.get("p_fw", float("nan"))
        if ew_["kind"] == "i" and ew_["k"] >= 5 and np.isfinite(pe) and pe <= 0.05 and ew_["bias"] != 0:
            add("Elliott", 0.5 * ew_["bias"], 0.25, f"{ew_['state']} (moslik {ew_['score']:.2f}; barcha variantlar ichidan eng yaxshisini tanlashni hisobga olganda tasodifiy narxda shunday moslik ~{pe * 100:.1f}% oynalarda).")
        else:
            add("Elliott", 0.0, 0.0, f"Eng mos sanoq: {ew_['state']} (moslik {ew_['score']:.2f}); barcha variantlar tanlovini hisobga olganda tasodifiy narxda shunday moslik {_p0(pe)} oynalarda chiqadi — "
                f"tasodifiydan ajralib turmaydi, shuning uchun xulosaga QOʻSHILMADI.", counted=False)
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


class LiteState:
    """Yengil rejim (AI universumidan tashqari aksiyalar): modelsiz, faqat texnik tahlil uchun State oʻrniga."""
    evals: Dict[int, Any] = {}
    calib: Dict[int, Any] = {}
    primary_h = PRIMARY_H
    built_at = None


def analyze_df(t: str, df: pd.DataFrame, market: pd.Series, st: Any, target: Optional[float] = None, ai_fn: Optional[Callable[[pd.Series], Any]] = None,
               fund: Optional[Dict[str, Any]] = None, news: Optional[Dict[str, Any]] = None, in_uni: bool = False, lite: bool = False,
               name: Optional[str] = None, flag: Optional[str] = None) -> Dict[str, Any]:
    """Bitta aksiyaning toʻliq texnik tahlili (narx DataFrame idan). analyze_ticker (toʻliq) va analyze_lite (yengil) umumiy yadrosi."""
    F = compute_features(df, market)
    last = F.iloc[-1]
    price = float(df["Close"].iloc[-1])
    ai = ai_fn(last) if (ai_fn is not None and len(df) >= WARMUP + 5) else None
    wy_code_series = wyckoff_code(df)
    wy_code = int(wy_code_series.iloc[-1]) if pd.notna(wy_code_series.iloc[-1]) else 0
    lv = find_levels(df)
    ev = wyckoff_events(df)
    if wy_code in (1, 3) and ev.get("range_ok"):          # trendsiz diapazonda Spring/Upthrust hodisasi faza yorligʻini aniqlashtiradi
        if ev.get("spring") and not ev.get("upthrust"):
            wy_code = 1
        elif ev.get("upthrust") and not ev.get("spring"):
            wy_code = 3
    fund = fund if fund is not None else dict(ok=False, error="hisoblanmadi")
    news = news if news is not None else dict(ok=False, error="hisoblanmadi", items=[])
    R: Dict[str, Any] = dict(ticker=t, in_universe=in_uni, df=df, last=last, price=price, asof=df.index[-1], ai=ai, wy_code=wy_code, wy_events=ev,
                             levels=lv, gann=gann_fan(df), base=find_base(df), candles=candles_last(df), fund=fund,
                             fund_score=score_fundamentals(fund, price), news=news, target=target, built_at=getattr(st, "built_at", None), lite=lite,
                             name=name, flag=flag)
    try:                                                                  # 20 kunlik oʻrtacha (median) dollar hajm: likvidlik ogohlantirishi uchun
        R["dvol"] = float((df["Close"] * df["Volume"]).tail(20).median())
    except Exception:
        R["dvol"] = float("nan")
    R["tech"] = compute_technicals(df)
    R["lvls"] = collect_levels(R)                                         # barcha nomzod darajalar: Monte-Karlo ehtimoli hammasi uchun bir yoʻlda
    R["mc"] = mc_forecast(df, target=target, levels=[(x["id"], x["price"]) for x in R["lvls"]])
    chg = {}
    c = df["Close"]
    for name, n in (("1 kun", 1), ("1 hafta", 5), ("1 oy", 21), ("3 oy", 63), ("6 oy", 126), ("1 yil", 252)):
        if len(c) > n:
            chg[name] = float(c.iloc[-1] / c.iloc[-1 - n] - 1)
    R["chg"] = chg
    R["conclusion"] = build_conclusion(R, st)
    R["short"] = short_conclusion(R)
    return R


LITE_MIN_BARS = 120


def analyze_lite(ticker: str, df: pd.DataFrame, market: pd.Series, name: Optional[str] = None, flag: Optional[str] = None) -> Dict[str, Any]:
    """Yengil tahlil: AI/fundamental/yangilik yoʻq, faqat narx tarixidan (texnik bloklar, Monte-Karlo, qisqa xulosa)."""
    t = normalize_ticker(ticker)
    if len(df) < LITE_MIN_BARS:
        raise DataError(f"'{t}' uchun tarix juda qisqa ({len(df)} kun) — kamida ~{LITE_MIN_BARS} savdo kuni kerak.")
    return analyze_df(t, df, market, LiteState(), lite=True, name=name, flag=flag)


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
    try:
        fund = f_fund.result(timeout=15)
    except Exception as e:
        fund = dict(ok=False, error=f"{type(e).__name__}")
    try:
        news = f_news.result(timeout=15)
    except Exception as e:
        news = dict(ok=False, error=f"{type(e).__name__}", items=[])
    pool.shutdown(wait=False)
    R = analyze_df(t, df, market, st, target=target, ai_fn=lambda last: ai_block(st, last), fund=fund, news=news, in_uni=in_uni)
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
    if abs(v) >= 100:
        return f"${v:,.0f}"
    if abs(v) < 1.0:
        return f"${v:,.4f}" if abs(v) >= 0.01 else f"${v:,.6f}"
    return f"${v:,.2f}"


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


# ======================================================================================
# 7b. SVG TEXNIK CHIZMALAR (ixcham va aniq; telefon va kompyuter uchun alohida oʻlcham)
# ======================================================================================
_SVG_N = [0]
SC_UP_, SC_DN_ = C_BLUE, C_RED
SC_SUP, SC_RES = "#0c8f3a", "#d03b3b"
SC_PAT, SC_EW, SC_FIB = "#7c4dcc", "#b87b00", "#6b6a64"
SVG_FONT = "system-ui,-apple-system,'Segoe UI',Roboto,sans-serif"


def _spread(ys: List[float], gap: Any, lo: float, hi: float) -> List[float]:
    """Yorliqlar bir-birini yopmasligi uchun y koordinatalarini (piksel) suradi: har yorliq oʻz balandligi (gap: son yoki roʻyxat) qadar joy oladi."""
    n = len(ys)
    if n == 0:
        return []
    hs = [float(gap)] * n if not isinstance(gap, (list, tuple)) else [float(g) for g in gap]
    order = sorted(range(n), key=lambda i: ys[i])
    out = [min(max(ys[i], lo), hi) for i in order]
    for k in range(1, n):
        out[k] = max(out[k], out[k - 1] + hs[order[k - 1]])
    if out[-1] > hi:
        out[-1] = hi
        for k in range(n - 2, -1, -1):
            out[k] = min(out[k], out[k + 1] - hs[order[k]])
    res = [0.0] * n
    for k, i in enumerate(order):
        res[i] = out[k]
    return res


def _nice_ticks(lo: float, hi: float, k: int = 5) -> List[float]:
    """Oraliqda taxminan k ta "chiroyli" (1-2-2.5-5) qadamli belgi."""
    span = hi - lo
    if span <= 0 or not np.isfinite(span):
        return [lo]
    mag = 10 ** math.floor(math.log10(span / k))
    step = mag
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if span / step <= k + 0.5:
            break
    v = math.ceil(lo / step - 1e-9) * step
    out: List[float] = []
    while v <= hi + 1e-9 and len(out) < 40:
        out.append(round(v, 8))
        v += step
    return out


def _pfmt(p: float, narrow: bool = False) -> str:
    return _money(p) if (narrow or abs(p) < 1.0) else f"${p:,.2f}"


class _Chart:
    """Bitta SVG narx grafigi: log-narx oʻqi, shamlar, hajm paneli va qatlamlar (S/R, zona, chiziq, pattern, Elliott...)."""

    def __init__(self, R: Dict[str, Any], bars: int, ahead: int, narrow: bool, vol: bool = True, mR: Optional[int] = None):
        df = R["df"]
        self.R, self.narrow = R, narrow
        self.n = int(min(bars, len(df)))
        self.d = df.tail(self.n)
        self.off = len(df) - self.n
        self.ahead = int(ahead)
        self.W, self.H = (470, 500) if narrow else (960, 480)
        self.mL = 46 if narrow else 58
        self.mR = int(mR) if mR else (92 if narrow else 150)
        self.fs = 14.0 if narrow else 12.0
        self.vol = vol
        self.pt = 6.0
        self.vb = float(self.H - 22)
        self.vt = self.vb - ((54 if narrow else 56) if vol else 0)
        self.pb = (self.vt - 8) if vol else self.vb
        self.x0, self.x1 = float(self.mL), float(self.W - self.mR)
        self.dx = (self.x1 - self.x0) / (self.n + self.ahead)
        self.o: List[str] = []
        self.labels: List[Tuple[float, str, str, bool, str]] = []
        self.legend: List[Tuple[str, str, str]] = []
        self.ly0 = self.ly1 = 0.0
        self.i_last = len(df) - 1

    # --- koordinatalar
    def X(self, i: float) -> float:
        return self.x0 + (i - self.off + 0.5) * self.dx

    def Y(self, p: float) -> float:
        return self.pb - (math.log(max(p, 1e-9)) - self.ly0) / (self.ly1 - self.ly0) * (self.pb - self.pt)

    def Yv(self, a: np.ndarray) -> np.ndarray:
        return self.pb - (np.log(np.maximum(a, 1e-9)) - self.ly0) / (self.ly1 - self.ly0) * (self.pb - self.pt)

    @property
    def i_end(self) -> int:
        return self.off + self.n - 1 + self.ahead

    def fit(self, hints: List[float], pad_f: float = 0.04) -> None:
        lo, hi = float(self.d["Low"].min()), float(self.d["High"].max())
        lo0, hi0 = lo, hi
        for p in hints:
            if p and np.isfinite(p) and p > 0 and lo0 / 1.4 <= p <= hi0 * 1.4:
                lo, hi = min(lo, p), max(hi, p)
        if math.log(hi) - math.log(lo) < 0.02:                    # narx deyarli oʻzgarmagan (toʻxtatilgan aksiya): oʻqni ±1% bilan ochamiz
            mid = math.sqrt(lo * hi)
            lo, hi = mid * 0.99, mid * 1.01
        pad = pad_f * (math.log(hi) - math.log(lo))
        self.ly0, self.ly1 = math.log(lo) - pad * 1.6, math.log(hi) + pad

    # --- primitivlar
    def line(self, i0: float, p0: float, i1: float, p1: float, col: str, w: float = 1.3, dash: str = "", op: float = 1.0) -> None:
        da = f' stroke-dasharray="{dash}"' if dash else ""
        oa = f' stroke-opacity="{op:g}"' if op < 1 else ""
        self.o.append(f'<path d="M{self.X(i0):.1f} {self.Y(p0):.1f}L{self.X(i1):.1f} {self.Y(p1):.1f}" stroke="{col}" stroke-width="{w:g}" fill="none"{da}{oa}/>')

    def hline(self, p: float, col: str, w: float = 1.0, dash: str = "4 3", i0: Optional[float] = None, op: float = 1.0) -> None:
        xa = self.x0 if i0 is None else min(max(self.x0, self.X(i0)), self.x1)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        oa = f' stroke-opacity="{op:g}"' if op < 1 else ""
        self.o.append(f'<path d="M{xa:.1f} {self.Y(p):.1f}H{self.x1:.1f}" stroke="{col}" stroke-width="{w:g}" fill="none"{da}{oa}/>')

    def band(self, plo: float, phi: float, col: str, op: float = 0.12, i0: Optional[float] = None, i1: Optional[float] = None) -> None:
        xa = self.x0 if i0 is None else min(max(self.x0, self.X(i0) - self.dx / 2), self.x1)
        xb = self.x1 if i1 is None else min(max(self.x0, self.X(i1) + self.dx / 2), self.x1)
        ya, yb = self.Y(phi), self.Y(plo)
        self.o.append(f'<rect x="{xa:.1f}" y="{ya:.1f}" width="{max(xb - xa, 0):.1f}" height="{max(yb - ya, 1.0):.1f}" fill="{col}" fill-opacity="{op:g}"/>')

    def poly(self, pts: List[Tuple[float, float]], col: str, op: float = 0.1) -> None:
        s = " ".join(f"{self.X(i):.1f},{self.Y(p):.1f}" for i, p in pts)
        self.o.append(f'<polygon points="{s}" fill="{col}" fill-opacity="{op:g}"/>')

    def polyline(self, pts: List[Tuple[float, float]], col: str, w: float = 1.5, dash: str = "", op: float = 1.0) -> None:
        if len(pts) < 2:
            return
        s = " ".join(f"{self.X(i):.1f},{self.Y(p):.1f}" for i, p in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        oa = f' stroke-opacity="{op:g}"' if op < 1 else ""
        self.o.append(f'<polyline points="{s}" fill="none" stroke="{col}" stroke-width="{w:g}" stroke-linejoin="round"{da}{oa}/>')

    def dot(self, i: float, p: float, col: str, r: float = 3.2) -> None:
        self.o.append(f'<circle cx="{self.X(i):.1f}" cy="{self.Y(p):.1f}" r="{r:g}" fill="{col}" stroke="{SURF}" stroke-width="1.2"/>')

    def txt(self, i: float, p: float, s: str, col: str = INK2, dx: float = 0, dy: float = 0, anchor: str = "start", bold: bool = False) -> None:
        w = ' font-weight="600"' if bold else ""
        self.o.append(f'<text x="{self.X(i) + dx:.1f}" y="{self.Y(p) + dy:.1f}" text-anchor="{anchor}" fill="{col}"{w}>{esc(s)}</text>')

    def label(self, p: float, s: str, col: str = INK2, bold: bool = False, s2: str = "") -> None:
        self.labels.append((self.Y(p), s, col, bold, s2))

    def leg(self, kind: str, col: str, text: str) -> None:
        if not any(x[2] == text for x in self.legend):
            self.legend.append((kind, col, text))

    # --- asosiy qatlam: shamlar, hajm, oʻqlar
    def base_svg(self) -> Tuple[str, str]:
        d = self.d
        O, H, L, C = (d[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
        xs = self.x0 + (np.arange(self.n) + 0.5) * self.dx
        yo, yh, yl, yc = (self.Yv(a) for a in (O, H, L, C))
        up = C >= O
        bw = max(1.0, round(0.6 * self.dx, 2))
        yc = np.where(np.abs(yc - yo) < 0.9, yo + np.where(up, -0.9, 0.9), yc)

        def seg(mask: np.ndarray, a: np.ndarray, b: np.ndarray) -> str:
            return "".join(f"M{x:.1f} {p:.1f}V{q:.1f}" for x, p, q, m in zip(xs, a, b, mask) if m)
        parts = []
        for mask, col in ((up, SC_UP_), (~up, SC_DN_)):
            w_ = seg(mask, yh, yl)
            b_ = seg(mask, yo, yc)
            if w_:
                parts.append(f'<path d="{w_}" stroke="{col}" stroke-width="1" fill="none"/>')
            if b_:
                parts.append(f'<path d="{b_}" stroke="{col}" stroke-width="{bw:g}" fill="none"/>')
        grid, axis = [], []
        lo_p, hi_p = math.exp(self.ly0), math.exp(self.ly1)
        for tv in _nice_ticks(lo_p, hi_p, 5 if self.narrow else 6):
            y = self.Y(tv)
            if self.pt + 4 <= y <= self.pb - 2:
                grid.append(f'<path d="M{self.x0:.1f} {y:.1f}H{self.x1:.1f}" stroke="{GRID}" stroke-width="0.8" fill="none"/>')
                axis.append(f'<text x="{self.x0 - 6:.1f}" y="{y + 4:.1f}" text-anchor="end" fill="{MUTED}">{_money(tv)}</text>')
        vol_svg = ""
        if self.vol:
            v = d["Volume"].values.astype(float)
            if v.size and v.max() > 0:
                vmax = float(np.percentile(v, 98)) or float(v.max())
                hv = np.minimum(v / vmax, 1.0) * (self.vb - self.vt)
                vol_svg = (f'<path d="{"".join(f"M{x:.1f} {self.vb:.1f}V{self.vb - h:.1f}" for x, h in zip(xs, hv))}" '
                           f'stroke="{AXIS}" stroke-width="{bw:g}" fill="none"/>')
        k = 3 if self.narrow else 6
        ticks = np.linspace(0, self.n - 1, k).round().astype(int)
        for j, ti in enumerate(ticks):
            fmt = "%b '%y" if self.narrow else "%d %b '%y"
            anc = "start" if j == 0 else "end" if j == k - 1 else "middle"
            axis.append(f'<text x="{xs[ti]:.1f}" y="{self.H - 6:.1f}" text-anchor="{anc}" fill="{MUTED}">{d.index[ti].strftime(fmt)}</text>')
        return "".join(grid) + "".join(parts) + vol_svg, "".join(axis)

    def svg(self, alt: str) -> str:
        _SVG_N[0] += 1
        uid = _SVG_N[0]
        base, axis = self.base_svg()
        gap = self.fs * 1.18
        out = [f'<svg class="ch" viewBox="0 0 {self.W} {self.H}" role="img" aria-label="{esc(alt)}" font-size="{self.fs:g}" font-family="{SVG_FONT}" '
               f'xmlns="http://www.w3.org/2000/svg"><rect width="{self.W}" height="{self.H}" fill="{SURF}"/>',
               f'<defs><clipPath id="c{uid}"><rect x="{self.x0:.1f}" y="{self.pt:.1f}" width="{self.x1 - self.x0:.1f}" height="{self.pb - self.pt:.1f}"/></clipPath></defs>',
               f'<g clip-path="url(#c{uid})">', base, "".join(self.o), "</g>", axis]
        # oʻng chetdagi yorliqlar
        if self.labels:
            labs = self.labels
            hs = [self.fs * (2.3 if x[4] else 1.18) for x in labs]
            ys = [x[0] for x in labs]
            lo_, hi_ = self.pt + self.fs * 0.6, self.pb - self.fs * 0.6
            adj = _spread(ys, hs, lo_, hi_)
            for (y0, s, col, bold, s2), y1 in zip(labs, adj):
                if self.pt - 2 <= y0 <= self.pb + 2 and abs(y1 - y0) > 3:
                    out.append(f'<path d="M{self.x1:.1f} {y0:.1f}L{self.x1 + 5:.1f} {y1 - 3.5:.1f}" stroke="{AXIS}" stroke-width="0.7" fill="none"/>')
                elif self.pt - 2 <= y0 <= self.pb + 2:
                    out.append(f'<path d="M{self.x1:.1f} {y0:.1f}h4" stroke="{col}" stroke-width="1" fill="none"/>')
                else:
                    continue
                w = ' font-weight="600"' if bold else ""
                t = f'<text x="{self.x1 + 7:.1f}" y="{y1:.1f}" fill="{col}"{w}>{esc(s)}'
                if s2:
                    t += f'<tspan x="{self.x1 + 7:.1f}" dy="{self.fs * 1.12:.1f}" fill="{MUTED}" font-size="{self.fs - 1.5:g}">{esc(s2)}</tspan>'
                out.append(t + "</text>")
        out.append("</svg>")
        return "".join(out)



def _legend_html(items: List[Tuple[str, str, str]]) -> str:
    sw = {"box": 'class="bx" style="background:{c}"', "band": 'class="bx" style="background:{c};opacity:.35"',
          "line": 'style="border-top:2.5px solid {c}"', "dash": 'style="border-top:2px dashed {c}"', "dot": 'style="border-top:2px dotted {c}"'}
    return '<div class="lg">' + "".join(f'<span><i {sw.get(k, sw["line"]).format(c=c)}></i>{esc(t)}</span>' for k, c, t in items) + "</div>"


# --------------------------------------------------------------------------------------
# Qatlamlar
# --------------------------------------------------------------------------------------
def _L_ma(c: _Chart) -> None:
    full = c.R["df"]["Close"]
    for n_, col, nm in ((50, C_ORANGE, "SMA50"), (200, C_AQUA, "SMA200")):
        s = full.rolling(n_).mean().iloc[c.off:].values
        pts = [(c.off + i, float(v)) for i, v in enumerate(s) if np.isfinite(v)]
        if len(pts) >= 2:
            c.polyline(pts, col, 1.5)
            c.leg("line", col, nm)


def _L_sr(c: _Chart) -> None:
    lv = c.R["levels"]
    for items, tag, col in ((lv["resistance"][:2], "R", SC_RES), (lv["support"][:2], "S", SC_SUP)):
        for i, it in enumerate(items):
            p = float(it["price"])
            lo_, hi_ = it.get("lo"), it.get("hi")
            if lo_ and hi_ and hi_ > lo_:
                c.band(lo_, hi_, col, 0.10)
            c.hline(p, MUTED, 0.9, "4 3")
            c.label(p, f"{tag}{i + 1} {_pfmt(p, c.narrow)}")
        if items:
            c.leg("band", col, "Qarshilik (R)" if tag == "R" else "Qoʻllab-quvvatlash (S)")


def _L_base(c: _Chart) -> None:
    bs = c.R.get("base") or {}
    if not (bs.get("ok") and bs["e_abs"] - c.off >= 0):
        return
    c.band(bs["lo"], bs["hi"], C_BLUE, 0.09, bs["s_abs"], bs["e_abs"])
    c.hline(bs["hi"], C_BLUE, 1.1, "1.5 2.5", bs["s_abs"])
    c.label(bs["hi"], "Baza usti" if c.narrow else f"Baza usti  {_pfmt(bs['hi'])}", C_BLUE)
    c.leg("band", C_BLUE, "Baza")
    if bs["state"] in ("inside", "breakout"):
        price = float(c.R["price"])
        for tg_ in bs["targets"]:
            if tg_["price"] > price:
                c.hline(tg_["price"], C_GOOD, 1.0, "5 3", c.off + c.n - 30)
                c.label(tg_["price"], (f"M{tg_['name'][1]} " + _money(tg_["price"])) if c.narrow else f"Maqsad {tg_['name'][1]}  {_pfmt(tg_['price'])}", C_GOOD)
        c.leg("dash", C_GOOD, "Baza maqsadi")


def _L_wy(c: _Chart) -> None:
    ev = c.R["wy_events"]
    idx = c.d.index
    for key, yk, col, txt_, dy in (("spring", "low", C_BLUE, "Spring", 15), ("upthrust", "high", C_RED, "Upthrust", -9)):
        e = ev.get(key)
        if e and e["date"] in idx:
            i = c.off + idx.get_loc(e["date"])
            c.dot(i, e[yk], col, 4.2)
            c.txt(i, e[yk], txt_, INK2, 0, dy, "middle")
    t = ev.get("test")
    if t and t["date"] in idx:
        i = c.off + idx.get_loc(t["date"])
        c.dot(i, t["low"], C_BLUE, 4.2)
        c.txt(i, t["low"], "Test", INK2, 0, 15, "middle")


def _L_gann(c: _Chart) -> None:
    gf = c.R["gann"]
    if not gf.get("ok"):
        return
    a = gf["anchor_pos"]
    i_a = max(a, c.off)
    ii = np.linspace(i_a, c.i_end, 9)
    for ln in gf["lines"]:
        if ln["name"] not in ("4x1", "2x1", "1x1", "1x2", "1x4"):
            continue
        pts = [(float(i), gf["anchor_low"] + ln["slope"] * (i - a)) for i in ii]
        pts = [(i, p) for i, p in pts if p > 0]
        is11 = ln["name"] == "1x1"
        c.polyline(pts, INK2 if is11 else AXIS, 1.3 if is11 else 0.9, "" if is11 else "1.5 2.5")
        if (is11 or ln["name"] in ("2x1", "1x2")) and pts and math.exp(c.ly0) <= pts[-1][1] <= math.exp(c.ly1):
            c.label(pts[-1][1], "Gann " + ln["name"].replace("x", "×"), INK2)
    if a >= c.off:
        c.dot(a, gf["anchor_low"], INK2, 3.8)
    c.leg("line", INK2, "Gann 1×1")


def _L_zones(c: _Chart) -> None:
    sd = (c.R.get("tech") or {}).get("sd") or {}
    for key, col, nm in (("demand", SC_SUP, "Talab zonasi"), ("supply", SC_RES, "Taklif zonasi")):
        zs = sd.get(key) or []
        for k, z in enumerate(zs):
            c.band(z["lo"], z["hi"], col, 0.17, z["i0"])
            c.hline(z["hi" if key == "demand" else "lo"], col, 0.8, "", z["i0"], 0.7)
            c.label((z["lo"] + z["hi"]) / 2, (f"{'T' if key == 'demand' else 'Z'}{k + 1} " + _money(z["hi" if key == "demand" else "lo"])) if c.narrow
                    else f"{nm.split()[0]} {k + 1}  {_pfmt(z['lo'])}–{_pfmt(z['hi'])}", col)
        if zs:
            c.leg("band", col, nm)


def _L_trend(c: _Chart) -> None:
    tr = (c.R.get("tech") or {}).get("trend") or {}
    rg = tr.get("regression")
    if rg and rg["r2"] >= 0.25:
        i0, ie = rg["i0"], c.i_end
        mid = lambda i: rg["p0"] * math.exp(rg["s"] * (i - i0))
        up_, lo_ = math.exp(2 * rg["sd"]), math.exp(-2 * rg["sd"])
        c.poly([(i0, mid(i0) * up_), (ie, mid(ie) * up_), (ie, mid(ie) * lo_), (i0, mid(i0) * lo_)], SC_FIB, 0.07)
        c.line(i0, mid(i0), ie, mid(ie), SC_FIB, 1.0, "5 3", 0.8)
        c.leg("band", SC_FIB, "Regressiya kanali (±2σ)")
    ch = tr.get("channel")
    if ch:
        for key in ("top", "bot"):
            ln = ch[key]
            i0 = max(ln["i0"], c.off - 50)
            c.line(i0, _lp(ln, i0), c.i_end, _lp(ln, c.i_end), "#2a78d6", 1.1, "2 3")
        c.leg("dot", "#2a78d6", "Trend kanali")
    ie = c.i_end
    for key, col, nm in (("support", SC_SUP, "Qoʻllab-quvvatlash chizigʻi"), ("resistance", SC_RES, "Qarshilik chizigʻi")):
        ln = tr.get(key)
        if not ln:
            continue
        end_solid = ln["broke_i"] if ln["broke"] else c.i_last
        colr = col if not ln["broke"] else MUTED
        c.line(ln["i0"], ln["p0"], end_solid, _lp(ln, end_solid), colr, 1.8)
        if not ln["broke"]:
            c.line(c.i_last, _lp(ln, c.i_last), ie, _lp(ln, ie), col, 1.3, "4 3", 0.8)
            c.label(_lp(ln, ie), ("TS " if key == "support" else "TR ") + _pfmt(_lp(ln, ie), c.narrow), col)
        c.dot(ln["i0"], ln["p0"], colr, 3.4)
        c.dot(ln["i1"], ln["p1"], colr, 3.4)
        c.leg("line", colr, nm if not ln["broke"] else "Buzilgan trend chizigʻi")


def _L_pat(c: _Chart) -> None:
    items = ((c.R.get("tech") or {}).get("patterns") or {}).get("items") or []
    items = [x for x in items if x["quality"] >= 0.35][:2]
    d = c.d
    for j, it in enumerate(items):
        tag = f"P{j + 1}"
        for sg in it["lines"]:
            role = sg["role"]
            if role == "neck":
                c.line(sg["i0"], sg["p0"], sg["i1"], sg["p1"], SC_PAT, 1.4, "5 3")
            elif role == "pole":
                c.line(sg["i0"], sg["p0"], sg["i1"], sg["p1"], SC_PAT, 1.2, "", 0.55)
            else:
                c.line(sg["i0"], sg["p0"], sg["i1"], sg["p1"], SC_PAT, 1.8)
        for k, pt in enumerate(it["pts"]):
            c.dot(pt["i"], pt["p"], SC_PAT, 3.6)
            ii = pt["i"] - c.off
            if 0 <= ii < c.n:
                hi_ = float(d["High"].iloc[ii]), float(d["Low"].iloc[ii])
                above = pt["p"] >= (hi_[0] + hi_[1]) / 2
                c.txt(pt["i"], pt["p"], (pt["label"] or ""), SC_PAT, 0, -8 if above else 17, "middle", True)
        if it["pts"]:
            xm = (it["pts"][0]["i"] + it["pts"][-1]["i"]) / 2.0
            pm = max(p["p"] for p in it["pts"])
            c.txt(xm, pm, tag, SC_PAT, 0, -20, "middle", True)
        i_from = min(it["end"], c.i_last)
        tcol = SC_SUP if it["dir"] > 0 else SC_RES
        for key, col, dash, nm in (("target", tcol, "5 3", "maqsad"), ("trigger", SC_PAT, "1.5 2.5", "tasdiq"), ("invalid", SC_RES, "1.5 2.5", "bekor")):
            v = it.get(key)
            if v and np.isfinite(v):
                c.hline(v, col, 1.1, dash, i_from)
                if c.narrow:
                    c.label(v, f"{tag} {nm}", col, False, _pfmt(v, True))
                else:
                    c.label(v, f"{tag} {nm} {_pfmt(v)}", col)
    if items:
        c.leg("line", SC_PAT, "Pattern chiziqlari")
        c.leg("dash", SC_SUP, "Maqsad (yuqoriga)")
        c.leg("dash", SC_RES, "Maqsad (pastga)")
        c.leg("dot", SC_PAT, "Tasdiq darajasi")
        c.leg("dot", SC_RES, "Bekor qilish")


def _ew_short(name: str) -> str:
    """Elliott maqsad nomini oʻng chet yorligʻi uchun qisqartiradi: '5-toʻlqin 0.618×W1' -> '5-t 0.618×W1'."""
    m = re.match(r"^(\d)-toʻlqin\s+(.+)$", name)
    if m:
        return f"{m.group(1)}-t {m.group(2)}"
    return name.replace("Tuzatish", "Tuz.").replace("tuzatish", "tuz.")


def _L_ew(c: _Chart) -> None:
    ew = ((c.R.get("tech") or {}).get("elliott") or {}).get("best")
    if not ew:
        return
    pv = ew["pivots"]
    c.polyline([(q["i"], q["p"]) for q in pv], SC_EW, 2.0)
    if len(pv) >= 2 and not pv[-1]["ok"]:
        c.line(pv[-2]["i"], pv[-2]["p"], pv[-1]["i"], pv[-1]["p"], SC_EW, 2.0, "3 3")
    r = 8.5 if c.narrow else 8.0
    for j, q in enumerate(pv):
        nxt = pv[j + 1]["p"] if j + 1 < len(pv) else None
        prv = pv[j - 1]["p"] if j > 0 else None
        is_high = (q["p"] > nxt) if nxt is not None else (q["p"] > prv if prv is not None else True)
        cx, cy = c.X(q["i"]), c.Y(q["p"]) + (-(r + 4) if is_high else (r + 4))
        c.o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:g}" fill="{SURF}" stroke="{SC_EW}" stroke-width="1.4"/>'
                   f'<text x="{cx:.1f}" y="{cy + 4.2:.1f}" text-anchor="middle" fill="{SC_EW}" font-weight="700" font-size="{c.fs - 1.5:g}">{esc(q["label"])}</text>')
    i0 = pv[-1]["i"]
    for k, t in enumerate(ew["targets"]):
        col = MUTED if t.get("pull") else SC_SUP if ew["bias"] > 0 else SC_RES
        c.hline(t["price"], col, 1.1, "5 3", i0)
        if c.narrow:
            c.label(t["price"], _ew_short(t["name"]), col, False, _pfmt(t["price"], True))
        else:
            c.label(t["price"], f"{_ew_short(t['name'])} {_pfmt(t['price'])}", col)
    if ew.get("invalid"):
        c.hline(ew["invalid"], SC_RES, 1.2, "1.5 2.5", i0)
        c.label(ew["invalid"], "Bekor" if c.narrow else f"Bekor qilish {_pfmt(ew['invalid'])}", SC_RES, False, _pfmt(ew["invalid"], True) if c.narrow else "")
    c.leg("line", SC_EW, "Elliott sanogʻi")
    c.leg("dash", SC_SUP, "Maqsad")
    c.leg("dot", SC_RES, "Bekor qilish")


def _L_fib(c: _Chart) -> None:
    fb = (c.R.get("tech") or {}).get("fib") or {}
    if not fb.get("ok"):
        return
    ia, ib = fb["i_lo"], fb["i_hi"]
    c.line(ia, fb["lo"], ib, fb["hi"], SC_FIB, 2.0)
    c.dot(ia, fb["lo"], SC_FIB, 3.8)
    c.dot(ib, fb["hi"], SC_FIB, 3.8)
    i0 = min(ia, ib)
    for r_, p in fb["levels"]:
        c.hline(p, SC_FIB, 0.9, "4 3", i0)
        c.label(p, f"{r_:g}  {_pfmt(p, c.narrow)}", SC_FIB)
    for e_, p in fb["ext"]:
        if math.exp(c.ly0) <= p <= math.exp(c.ly1):
            c.hline(p, SC_SUP if fb["up"] else SC_RES, 1.0, "5 3", max(ia, ib))
            c.label(p, f"{e_:g}  {_pfmt(p, c.narrow)}", SC_SUP if fb["up"] else SC_RES)
    c.leg("line", SC_FIB, "Soʻnggi yirik toʻlqin")
    c.leg("dash", SC_FIB, "Tuzatish darajalari")
    c.leg("dash", SC_SUP if fb["up"] else SC_RES, "Kengayish maqsadlari")


SHORT_TAG = {"Birinchi toʻsiq": "1-toʻsiq", "Asosiy maqsad": "Maqsad", "Uzoq maqsad": "Uzoq maqsad", "Keyingi qarshilik": "2-qarshilik",
             "Birinchi tayanch": "1-tayanch", "Pasayish maqsadi": "Pasayish", "Keyingi tayanch": "2-tayanch"}


def _L_ladder(c: _Chart) -> None:
    sh = c.R.get("short") or {}
    price = float(c.R["price"])
    c.hline(price, INK, 1.0, "2 2")
    c.label(price, f"Narx {_pfmt(price, c.narrow)}", INK, True)
    inv = sh.get("invalid")
    same = False
    for key, col in (("up", SC_SUP), ("down", SC_RES)):
        for x in sh.get(key) or []:
            is_inv = bool(inv and abs(inv["price"] / x["price"] - 1.0) < 0.002)
            same = same or is_inv
            c.hline(x["price"], col, 1.6 if is_inv else 1.3, "" if is_inv else "5 3")
            p60 = x.get("p60")
            s2 = f"{x['pct'] * 100:+.1f}%" + (f" · 3 oyda {p60 * 100:.0f}%" if p60 is not None and np.isfinite(p60) else "")
            c.label(x["price"], f"{SHORT_TAG.get(x['tag'], x['tag'])}{'/bekor' if is_inv else ''} {_pfmt(x['price'], c.narrow)}", col, is_inv, s2)
    if inv and not same:
        c.hline(inv["price"], SC_RES, 1.4, "")
        c.label(inv["price"], f"Bekor {_pfmt(inv['price'], c.narrow)}", SC_RES, True)
    if inv:
        c.leg("line", SC_RES, "Bekor qilish darajasi")
    c.leg("dash", SC_SUP, "Yuqoridagi darajalar")
    c.leg("dash", SC_RES, "Pastdagi darajalar")


CHART_LAYERS = {
    "main": (_L_base, _L_sr, _L_gann, _L_wy),
    "tech": (_L_zones, _L_sr, _L_trend),
    "zones": (_L_zones, _L_trend),
    "pat": (_L_pat,),
    "ew": (_L_ew,),
    "fib": (_L_fib,),
    "ladder": (_L_ladder,),
}


def _chart_geometry(R: Dict[str, Any], kind: str, narrow: bool) -> Tuple[int, int, List[float], Optional[int], bool]:
    """Grafik uchun: shamlar soni, kelajak barlar, diapazon maslahatlari, oʻng chet kengligi, MA qatlami kerakmi."""
    tech = R.get("tech") or {}
    n_all = len(R["df"])
    hints: List[float] = []
    ma = False
    mR = None
    if kind == "main":
        bars, ahead = (100, 22) if narrow else (170, 45)
        lv = R["levels"]
        hints += [x["price"] for x in lv["resistance"][:2] + lv["support"][:2]]
        bs = R.get("base") or {}
        if bs.get("ok") and bs["state"] in ("inside", "breakout"):
            hints += [x["price"] for x in bs["targets"] if x["price"] > R["price"]][:2]
        ma = True
    elif kind in ("zones", "tech"):
        bars, ahead = (110, 20) if narrow else (190, 20)
        sd = tech.get("sd") or {}
        for z in (sd.get("demand") or []) + (sd.get("supply") or []):
            hints += [z["lo"], z["hi"]]
        tr = tech.get("trend") or {}
        for k in ("support", "resistance"):
            if tr.get(k):
                hints += [tr[k]["now"], tr[k]["fut20"]]
        if tr.get("channel"):
            hints += [tr["channel"]["top_now"], tr["channel"]["bot_now"]]
        if kind == "tech":
            lv = R["levels"]
            hints += [x["price"] for x in lv["resistance"][:2] + lv["support"][:2]]
        ma = kind == "tech"
    elif kind == "pat":
        items = [x for x in ((tech.get("patterns") or {}).get("items") or []) if x["quality"] >= 0.35][:2]
        first = min([x["start"] for x in items] or [n_all - 100])
        bars, ahead = int(min(max(n_all - first + 25, 110 if not narrow else 80), 260)), (16 if narrow else 30)
        mR = 104 if narrow else None
        for it in items:
            hints += [it["target"], it["invalid"], it["trigger"]] + [p["p"] for p in it["pts"]]
    elif kind == "ew":
        ew = (tech.get("elliott") or {}).get("best") or {}
        pv = ew.get("pivots") or []
        first = min([q["i"] for q in pv] or [n_all - 120])
        bars, ahead = int(min(max(n_all - first + 25, 120), 320)), (16 if narrow else 30)
        mR = 108 if narrow else None
        hints += [q["p"] for q in pv] + [t["price"] for t in ew.get("targets", [])] + [ew.get("invalid") or 0]
    elif kind == "fib":
        fb = tech.get("fib") or {}
        first = min(fb.get("i_lo", n_all - 120), fb.get("i_hi", n_all - 120))
        bars, ahead = int(min(max(n_all - first + 20, 120), 320)), (16 if narrow else 30)
        hints += [fb.get("lo", 0), fb.get("hi", 0)] + [p for _, p in fb.get("levels", [])] + [p for _, p in fb.get("ext", [])][:2]
    else:  # ladder
        bars, ahead = (80, 12) if narrow else (130, 26)
        sh = R.get("short") or {}
        hints += [x["price"] for x in (sh.get("up") or []) + (sh.get("down") or [])] + [(sh.get("invalid") or {}).get("price", 0)]
        mR = 168 if narrow else 220
        ma = True
    return int(min(bars, n_all)), int(ahead), hints, mR, ma


def svg_chart(R: Dict[str, Any], kind: str, narrow: bool, alt: str) -> Tuple[str, List[Tuple[str, str, str]]]:
    bars, ahead, hints, mR, ma = _chart_geometry(R, kind, narrow)
    c = _Chart(R, bars, ahead, narrow, vol=(kind in ("main", "tech")) and not R.get("lite"), mR=mR)
    c.fit(hints, 0.07 if kind in ("pat", "ew", "fib") else 0.04)
    if ma and kind in ("main", "tech"):
        _L_ma(c)
    elif ma:
        s50 = R["df"]["Close"].rolling(50).mean().iloc[c.off:].values
        c.polyline([(c.off + i, float(v)) for i, v in enumerate(s50) if np.isfinite(v)], C_ORANGE, 1.3, "", 0.8)
        c.leg("line", C_ORANGE, "SMA50")
    for fn in CHART_LAYERS[kind]:
        fn(c)
    leg = [("box", SC_UP_, "Oʻsgan sham"), ("box", SC_DN_, "Tushgan sham")] + c.legend
    return c.svg(alt), leg


def chart_block(R: Dict[str, Any], kind: str, alt: str) -> str:
    """Kompyuter (keng) va telefon (tor) variantli SVG; CSS (.cw/.cn) ekran kengligiga qarab biridan foydalanadi.
    Yengil rejimda (minglab sahifa, hajm cheklangan) faqat bitta — tor — variant beriladi (.cl: har ekranda oʻrtada, kengligi cheklangan)."""
    if R.get("lite"):
        n, lg = svg_chart(R, kind, True, alt)
        return f'<div class="cl">{n}</div>{_legend_html(lg)}'
    w, lg = svg_chart(R, kind, False, alt)
    n, _ = svg_chart(R, kind, True, alt)
    return f'<div class="cw">{w}</div><div class="cn">{n}</div>{_legend_html(lg)}'


def svg_cone(R: Dict[str, Any], narrow: bool) -> str:
    mc = R["mc"]
    bands, P0 = np.asarray(mc["bands"], dtype=float), float(mc["P0"])
    hmax = bands.shape[1]
    W, H = (470, 330) if narrow else (960, 360)
    mL, mR, mT, mB = (46 if narrow else 58), (112 if narrow else 150), 8, 38
    fs = 14.0 if narrow else 12.0
    tg = mc.get("target")
    lo = min(float(bands[0].min()), P0, tg["price"] if tg else P0)
    hi = max(float(bands[4].max()), P0, tg["price"] if tg else P0)
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    x0, x1 = float(mL), float(W - mR)
    X = lambda d: x0 + d / hmax * (x1 - x0)
    Y = lambda p: (H - mB) - (p - lo) / (hi - lo) * (H - mB - mT)
    days = np.arange(1, hmax + 1)

    def band(a: np.ndarray, b: np.ndarray, col: str) -> str:
        top = " ".join(f"{X(d):.1f},{Y(v):.1f}" for d, v in zip(days, b))
        bot = " ".join(f"{X(d):.1f},{Y(v):.1f}" for d, v in zip(days[::-1], a[::-1]))
        return f'<polygon points="{X(0):.1f},{Y(P0):.1f} {top} {bot}" fill="{col}"/>'
    _SVG_N[0] += 1
    o = [f'<svg class="ch" viewBox="0 0 {W} {H}" role="img" aria-label="Monte-Karlo narx konusi" font-size="{fs:g}" font-family="{SVG_FONT}" '
         f'xmlns="http://www.w3.org/2000/svg"><rect width="{W}" height="{H}" fill="{SURF}"/>']
    for tv in _nice_ticks(lo, hi, 5):
        y = Y(tv)
        o.append(f'<path d="M{x0:.1f} {y:.1f}H{x1:.1f}" stroke="{GRID}" stroke-width="0.8" fill="none"/>'
                 f'<text x="{x0 - 6:.1f}" y="{y + 4:.1f}" text-anchor="end" fill="{MUTED}">{_money(tv)}</text>')
    o.append(band(bands[0], bands[4], BLUE_150))
    o.append(band(bands[1], bands[3], BLUE_300))
    o.append(f'<polyline points="{X(0):.1f},{Y(P0):.1f} {" ".join(f"{X(d):.1f},{Y(v):.1f}" for d, v in zip(days, bands[2]))}" fill="none" stroke="{C_BLUE}" stroke-width="2.2" stroke-linejoin="round"/>')
    o.append(f'<path d="M{x0:.1f} {Y(P0):.1f}H{x1:.1f}" stroke="{AXIS}" stroke-width="0.9" fill="none"/>')
    if tg:
        o.append(f'<path d="M{x0:.1f} {Y(tg["price"]):.1f}H{x1:.1f}" stroke="{INK2}" stroke-width="1.1" stroke-dasharray="4 3" fill="none"/>')
    mny = _money if narrow else (lambda v: _money(v) if abs(v) < 1.0 else f"${v:,.2f}")
    labs = [(Y(bands[4][-1]), f"95%: {mny(bands[4][-1])}"), (Y(bands[2][-1]), f"Median: {mny(bands[2][-1])}"), (Y(bands[0][-1]), f"5%: {mny(bands[0][-1])}")]
    if tg:
        labs.append((Y(tg["price"]), f"Maqsad: {mny(tg['price'])}"))
    adj = _spread([y for y, _ in labs], fs * 1.2, mT + fs * 0.6, H - mB - fs * 0.5)
    for (y0, s), y1 in zip(labs, adj):
        o.append(f'<text x="{x1 + 7:.1f}" y="{y1 + 4:.1f}" fill="{INK2}">{esc(s)}</text>')
    hs = [h for h in HORIZONS if 20 <= h <= hmax and (not narrow or h in (20, 60, 125))]
    for h in hs:
        o.append(f'<text x="{X(h):.1f}" y="{H - 22:.1f}" text-anchor="middle" fill="{MUTED}">{esc(HORIZON_UZ.get(h, str(h)))}'
                 f'<tspan x="{X(h):.1f}" dy="{fs * 1.15:.1f}" font-size="{fs - 1.5:g}">({h} kun)</tspan></text>')
    o.append("</svg>")
    return "".join(o)


def cone_block(R: Dict[str, Any]) -> str:
    lg = [("box", BLUE_150, "90% diapazon (5–95%)"), ("box", BLUE_300, "50% diapazon (25–75%)"), ("line", C_BLUE, "Median")]
    return f'<div class="cw">{svg_cone(R, False)}</div><div class="cn">{svg_cone(R, True)}</div>{_legend_html(lg)}'


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
.ai .cw{display:block}.ai .cn{display:none}.ai .cl{max-width:620px;margin:0 auto}
.ai svg.ch{display:block;width:100%;height:auto;border-radius:8px;margin:8px 0 2px}
.ai svg.ch text{paint-order:stroke;stroke:#fcfcfb;stroke-width:2.5px;stroke-linejoin:round}
.ai .lg{display:flex;flex-wrap:wrap;gap:3px 14px;font-size:.8rem;color:var(--ink2);margin:2px 2px 10px}
.ai .lg i{display:inline-block;width:16px;height:0;vertical-align:middle;margin-right:5px}
.ai .lg i.bx{height:9px;width:12px;border-radius:2px}
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
 .ai .cw{display:none}.ai .cn{display:block}
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
    if x is None or not np.isfinite(x):
        return "—"
    if d == 2 and abs(x) < 1.0:                    # 1 dollardan arzon aksiyalar: koʻproq oʻnlik xona ($0.0033 "$0.00" boʻlib qolmasin)
        d = 4 if abs(x) >= 0.01 else 6
    return f"${x:,.{d}f}"


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


# ======================================================================================
# 8b. TEXNIK BOʻLIMLAR (zona/chiziq, pattern, Elliott, Fibonachchi) VA OXIRGI «QISQA XULOSA»
# ======================================================================================
FULL_ANALYSIS_URL = ""        # build_site.py oʻrnatadi: toʻliq tahlilni soʻrash havolasi (GitHub Actions)


def _lt(R: Dict[str, Any]) -> Dict[str, Dict[int, float]]:
    return {x["name"]: x["touch"] for x in (R.get("mc") or {}).get("levels", [])}


def _p0(v: Optional[float]) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return "<1%" if v < 0.005 else f"{v * 100:.0f}%"


def _pcell(lt: Dict[str, Dict[int, float]], lid: str, h: int) -> str:
    return f'<td class="n">{_p0(lt.get(lid, {}).get(h))}</td>'


def sec_zones(R: Dict[str, Any]) -> str:
    t = R["ticker"]
    tech = R.get("tech") or {}
    sd, tr = tech.get("sd") or {}, tech.get("trend") or {}
    lt = _lt(R)
    n_last = len(R["df"]) - 1
    out = ["<h2>Zonalar va chiziqlar: talab/taklif zonalari, trend chiziqlari, kanal</h2>"]
    if not (sd.get("demand") or sd.get("supply") or tr.get("support") or tr.get("resistance") or tr.get("channel")):
        out.append('<p class="muted">Narxga yaqin ochiq talab/taklif zonasi yoki ishonchli trend chizigʻi topilmadi.</p>')
        return "".join(out)
    if not R.get("lite"):
        out.append(chart_block(R, "zones", f"{t}: talab/taklif zonalari va trend chiziqlari"))
    out.append('<p class="small muted">' + ("Zonalar va trend chiziqlari yuqoridagi narx grafigida chizilgan. " if R.get("lite") else "")
               + 'Boʻyalgan toʻgʻri toʻrtburchaklar — talab (yashil) va taklif (qizil) zonalari. Oʻng chetdagi yorliqlar (TS, TR) — '
               'trend chiziqlarining 20 savdo kuni keyingi davomidagi (punktir qism) qiymati.</p>')
    rows = []
    for key, nm, pre in (("supply", "Taklif", "Z"), ("demand", "Talab", "D")):
        for k, z in enumerate(sd.get(key) or []):
            rows.append(f'<tr><td>{nm} zonasi {k + 1}</td><td class="n">{usd_r(z["lo"])} – {usd_r(z["hi"])}</td><td class="n">{pc(z["dist"], 1, True)}</td>'
                        f'<td class="n">{z["date"]:%Y-%m-%d}</td><td class="n">{z["tests"]}</td>{_pcell(lt, f"{pre}{k + 1}", 60)}</tr>')
    out.append("<h3>Talab va taklif zonalari (supply / demand)</h3>")
    if rows:
        out.append(_table(["Zona", "Narx oraligʻi", "Hozirgidan (yaqin chegara)", "Shakllangan", "Sinovlar", "3 oyda yetish (MC)"], rows, [1, 2, 3, 4, 5],
                          hide_mobile=[3], wrap_first=True))
    else:
        out.append('<p class="muted">Soʻnggi ~1 yilda narxga yaqin ochiq (teshilmagan) zona topilmadi.</p>')
    out.append('<p class="small muted">Qoida: kuchli (tana ≥ 1.4×ATR) portlovchi shamdan oldingi 1–4 ta tor sham diapazoni = zona; zona narx YOPILISH bilan teshilsa oʻchiriladi. '
               '«Sinovlar» — narx zonaga necha marta qaytib kirgani (kamroq = “yangi” zona). Zonalarning kelajakdagi ish berishi bu yerda tekshirilmagan.</p>')
    out.append("<h3>Trend chiziqlari va kanal</h3>")
    rows, calc = [], []
    for key, nm in (("resistance", "Qarshilik chizigʻi"), ("support", "Qoʻllab-quvvatlash chizigʻi")):
        ln = tr.get(key)
        if not ln:
            rows.append(f'<tr class="dim"><td>{nm}</td><td colspan="6">mos chiziq topilmadi</td></tr>')
            continue
        st_ = f'buzilgan ({ln["days_since"]} kun oldin)' if ln["broke"] else "amal qilmoqda"
        rows.append(f'<tr><td>{nm}<div class="small muted">{esc(str(ln["deg"]))} daraja</div></td>'
                    f'<td style="white-space:normal">{ln["date0"]:%Y-%m-%d}: {usd(ln["p0"])} → {ln["date1"]:%Y-%m-%d}: {usd(ln["p1"])}</td>'
                    f'<td class="n">{pc(ln["slope_ann"], 0, True)}</td><td class="n">{ln["touches"]}</td><td>{st_}</td>'
                    f'<td class="n">{usd_r(ln["now"])} ({pc(ln["now"] / R["price"] - 1, 1, True)})</td><td class="n">{usd_r(ln["fut20"])}</td></tr>')
        calc.append(f"{nm}: qiyalik s = ln({ln['p1']:.2f}/{ln['p0']:.2f}) / {ln['i1'] - ln['i0']} bar = {ln['s']:+.5f}/bar (yillik {pc(ln['slope_ann'], 0, True)}); "
                    f"bugungi qiymat = {ln['p0']:.2f} × e^(s × {n_last - ln['i0']}) = {usd(ln['now'])}; 20 bardan keyin {usd(ln['fut20'])}.")
    out.append(_table(["Chiziq", "Tayanch nuqtalari", "Yillik qiyalik", "Teginishlar", "Holat", "Bugungi qiymat (narxdan)", "20 bardan keyin"], rows, [2, 3, 5, 6],
                      hide_mobile=[1, 2], wrap_first=True))
    bl = []
    ch, rg = tr.get("channel"), tr.get("regression")
    if ch:
        bl.append(f"<b>Trend kanali</b> ({'ikki chiziq deyarli parallel' if ch['kind'] == 'juftlik' else 'bitta chiziq va unga parallel ikkinchi chiziq'}): "
                  f"yuqori chegara {usd(ch['top_now'])}, pastki {usd(ch['bot_now'])}, kengligi {pc(ch['width'], 1)}; narx kanalning {pc(ch['pos'], 0)} qismida "
                  f"(0% — pastki, 100% — yuqori chegara). Teginishlar: yuqori {ch['touches_top']}, pastki {ch['touches_bot']}.")
    else:
        bl.append("Trend kanali: ikki parallel chiziq (kamida 2–3 teginish bilan) topilmadi.")
    if rg:
        bl.append(f"<b>Regressiya kanali</b> (oxirgi {rg['n']} bar log-narxiga toʻgʻri chiziq): yillik qiyalik {pc(rg['slope_ann'], 0, True)}, R² = {rg['r2']:.2f}; "
                  f"±2σ diapazon {usd(rg['lo'])} – {usd(rg['up'])} (oʻrtasi {usd(rg['mid'])}); narx markazdan {rg['z']:+.1f}σ da."
                  + (" R² past (< 0.25) — chiziqli trend ishonchsiz, grafikka chizilmadi." if rg["r2"] < 0.25 else ""))
    tr_uz = {"oʻsish": "oʻsish", "pasayish": "pasayish", "yon": "yonlama (trendsiz)"}.get(tr.get("trend", "yon"), "yonlama")
    bl.append(f"Umumiy trend yoʻnalishi (regressiyadan, R² ≥ 0.25 boʻlsa): <b>{tr_uz}</b>.")
    out.append('<ul class="small">' + "".join(f"<li>{x}</li>" for x in bl + [esc(c_) for c_ in calc]) + "</ul>")
    out.append('<p class="small muted">Trend chizigʻi qoidasi: ikki tasdiqlangan pivot (ZigZag, chegarasi 1.5·σ·√L) orqali log-narxda toʻgʻri chiziq; ikkinchi tayanchgacha hech bir '
               'YOPILISH chiziqning narigi tomonida boʻlmasligi shart; teginish — chiziqqa 0.6×ATR dan yaqin kelgan pivot (kamida 4 bar oraliq); 3 va undan koʻp teginish — '
               '“tasdiqlangan”. Bu mexanik qoida tajribali trader chizgan chiziqdan farq qilishi mumkin; chiziqlarning bashorat kuchi bu yerda tekshirilmagan.</p>')
    return "".join(out)


def sec_patterns(R: Dict[str, Any]) -> str:
    items = ((R.get("tech") or {}).get("patterns") or {}).get("items") or []
    out = ["<h2>Chart-patternlar (grafik shakllari)</h2>"]
    if not items:
        out.append('<p class="muted">Hozirgi grafikda qoidalarga mos (sifati ≥ 0.35) chart-pattern topilmadi: ikkilamchi choʻqqi/tub, bosh-yelkalar, uchburchaklar, '
                   'ponalar, bayroq/vimpel qidirildi.</p>')
        return "".join(out)
    out.append(chart_block(R, "pat", f"{R['ticker']}: chart-patternlar"))
    lt = _lt(R)
    price = R["price"]
    for j, it in enumerate(items[:3]):
        pr = pat_p_rand(it["kind"], it["quality"])
        ok = it["state"] == PAT_STATE_OK
        if it["bias"] == 0 and not ok:
            dtxt = "yoʻnalish noaniq (ikkala tomonga breakout mumkin)"
        else:
            dtxt = "kutilayotgan harakat: " + ("yuqoriga ▲" if it["dir"] > 0 else "pastga ▼")
        pfw = fw_adjust(PAT_FW, pr)
        sig_ = "tasodifiydan ajralib turadi" if np.isfinite(pfw) and pfw <= 0.05 else "tasodifiy narxda ham tez-tez uchraydi"
        out.append(f'<div class="card {"good" if ok else "warn"}"><b>P{j + 1}. {esc(it["name"])}</b> '
                   f'<span class="tag {"good" if ok else "warn"}">{esc(it["state"])}{f" ({it["days_since"]} kun oldin)" if ok and it.get("days_since") is not None else ""}</span> '
                   f'<span class="tag">sifat {it["quality"]:.2f}</span>'
                   f'<div class="small muted">{esc(str(it["deg"]))} daraja · {dtxt}. Tasodifiy narxda shu turdagi pattern shunday yoki yuqoriroq sifat bilan '
                   f'{_p0(pr)} oynalarda chiqadi; barcha pattern turlari ichidan eng yaxshisini tanlash hisobga olinsa — {_p0(pfw)} ({sig_}).</div>')
        rows = []
        for lab_, key, lid in (("Tasdiq darajasi", "trigger", f"PT{j + 1}"), ("Maqsad (measured move)", "target", f"P{j + 1}"), ("Bekor qilish", "invalid", f"PI{j + 1}")):
            v = it.get(key)
            if v is None or not np.isfinite(v):
                continue
            extra = " (allaqachon oʻtilgan)" if key == "trigger" and ok else ""
            rows.append(f'<tr><td>{lab_}{extra}</td><td class="n">{usd_r(v)}</td><td class="n">{pc(v / price - 1, 1, True)}</td>{_pcell(lt, lid, 20)}{_pcell(lt, lid, 60)}</tr>')
        out.append(_table(["Daraja", "Narx", "Hozirgidan", "1 oyda yetish (MC)", "3 oyda yetish (MC)"], rows, [1, 2, 3, 4], wrap_first=True))
        out.append('<ul class="small">' + "".join(f"<li>{esc(c_)}</li>" for c_ in it["calc"]) + "</ul></div>")
    out.append('<p class="small muted">Pattern qoidalari mexanik (ZigZag pivotlar geometriyasi). Muhim: tasodifiy narxda ham pattern aniqlagichlar tez-tez ishlaydi, shuning uchun '
               'har bir topilma yonida “tasodifiy narxda shunday sifat necha foiz oynalarda chiqishi” koʻrsatilgan; xulosaga faqat tanlov hisobga olingan koʻrsatkich ≤ 5% va sifat ≥ 0.6 boʻlgan pattern qoʻshiladi. '
               'Klassik pattern maqsadlarining tarixiy aniqligi bu yerda tekshirilmagan; “Monte-Karlo” ustunlari esa faqat volatillikdan hisoblangan statistik ehtimol.</p>')
    return "".join(out)


def sec_elliott(R: Dict[str, Any]) -> str:
    ew = (R.get("tech") or {}).get("elliott") or {}
    best = ew.get("best")
    out = ["<h2>Elliott toʻlqinlari (mexanik sanoq)</h2>"]
    if not best:
        out.append('<p class="muted">Hozirgi grafikda qatʼiy qoidalarga mos va Fibonachchi nisbatlariga yetarli darajada (moslik ≥ 0.45) toʻgʻri keladigan Elliott sanogʻi topilmadi.</p>')
        return "".join(out)
    out.append(chart_block(R, "ew", f"{R['ticker']}: Elliott sanogʻi"))
    lt = _lt(R)
    price = R["price"]
    kind_txt = (f"impuls (1-2-3-4-5), {best['k'] - 1} ta toʻlqin aniqlangan" if best["kind"] == "i" else "ABC tuzatish")
    reliable = best["kind"] == "i" and best["k"] >= 5 and np.isfinite(best["p_fw"]) and best["p_fw"] <= 0.05 and best["bias"] != 0
    fam_rate = ELLIOTT_RULE_RATE.get(best["fam"])
    out.append(f'<div class="card {"good" if reliable else "warn"}"><b>{esc(best["state"])}</b>'
               f'<div class="small muted">{kind_txt}; {"yuqoriga" if best["dir"] > 0 else "pastga"} yoʻnalgan, {esc(best["deg"])} daraja. '
               f'Fibonachchi moslik bali {best["score"]:.2f} (0–1). Tasodifiy narxda shunday yoki yaxshiroq moslik {_p0(best["p_rand"])} oynalarda chiqadi; '
               f'barcha variantlar ichidan eng yaxshisini tanlashni hisobga olsak — {_p0(best["p_fw"])}'
               + (f"; shu turdagi qoidaga mos sanoq tasodifiy narxning {fam_rate * 100:.0f}% oynalarida ham topiladi" if fam_rate is not None else "")
               + f'. Bu sanoq xulosaga {"QOʻSHILDI (kichik ogʻirlik)" if reliable else "QOʻSHILMADI (ishonchli emas)"}.</div></div>')
    rows = [f'<tr><td>{esc(a)}</td><td>{"✔ bajarildi" if ok_ else "✖ buzildi"}</td><td>{esc(b)}</td></tr>' for a, ok_, b in best["rules"]]
    out.append("<h3>Qoidalar va Fibonachchi nisbatlari</h3>")
    out.append(_table(["Qoida", "Holat", "Hisob"], rows, wrap_first=True))
    rows = []
    far = 0
    for j, t_ in enumerate(best["targets"]):
        if not (price / 5.0 <= t_["price"] <= price * 5.0):                       # hozirgi narxdan 5 baravardan uzoq daraja jadvalga kiritilmaydi
            far += 1
            continue
        rows.append(f'<tr><td>{esc(t_["name"])}{" (kutilayotgan tuzatish)" if t_.get("pull") else ""}</td><td class="n">{usd_r(t_["price"])}</td>'
                    f'<td class="n">{pc(t_["price"] / price - 1, 1, True)}</td><td>{esc(t_["rule"])}</td>{_pcell(lt, f"E{j + 1}", 60)}</tr>')
    out.append("<h3>Hisoblangan darajalar (Fibonachchi)</h3>")
    out.append(_table(["Daraja", "Narx", "Hozirgidan", "Hisob qoidasi", "3 oyda yetish (MC)"], rows, [1, 2, 4], hide_mobile=[3], wrap_first=True) if rows else
               '<p class="muted">Hisoblangan Elliott darajalarining hammasi hozirgi narxdan 5 baravardan uzoqda — jadvalga kiritilmadi.</p>')
    if far and rows:
        out.append(f'<p class="small muted">Hozirgi narxdan 5 baravardan uzoqdagi {far} ta daraja jadvalga kiritilmadi.</p>')
    out.append(f'<p class="small"><b>Bekor qilish:</b> {usd(best["invalid"])} ({pc(best["invalid"] / price - 1, 1, True)}) — {esc(best["invalid_text"])}.</p>')
    if ew.get("alts"):
        al = []
        for a in ew["alts"]:
            al.append(f'<li>{esc(a["state"])} ({esc(a["deg"])} daraja, {"yuqoriga" if a["dir"] > 0 else "pastga"}; moslik {a["score"]:.2f}, tasodifiy narxda {_p0(a["p_fw"])})</li>')
        out.append('<p class="small"><b>Muqobil sanoqlar</b> (xuddi shu grafikka mos boshqa talqinlar):</p><ul class="small">' + "".join(al) + "</ul>")
    out.append('<p class="small muted">Elliott sanogʻi subyektiv: bitta grafikka bir nechta sanoq mos kelishi mumkin. Bu yerda faqat QATʼIY qoidalarga mos (2-toʻlqin 1-toʻlqinni 100% dan kam qaytaradi, '
               '3-toʻlqin eng qisqa emas, 4-toʻlqin 1-toʻlqin hududiga kirmaydi) mexanik variant olinadi va Fibonachchi moslik bali tasodifiy yurishdan olingan taqsimot bilan solishtiriladi. '
               'Maqsadlar — Fibonachchi nisbatlaridan hisoblangan darajalar, bashorat emas; Elliott usulining tarixiy aniqligi bu yerda tekshirilmagan.</p>')
    return "".join(out)


def sec_fib(R: Dict[str, Any]) -> str:
    fb = (R.get("tech") or {}).get("fib") or {}
    out = ["<h2>Fibonachchi darajalari</h2>"]
    if not fb.get("ok"):
        out.append('<p class="muted">Fibonachchi uchun yetarli yirik toʻlqin (swing) topilmadi.</p>')
        return "".join(out)
    if not R.get("lite"):
        out.append(chart_block(R, "fib", f"{R['ticker']}: Fibonachchi"))
    lt = _lt(R)
    price = R["price"]
    rng = fb["hi"] - fb["lo"]
    near = min([(abs(p - price), f"F{r_:g}") for r_, p in fb["levels"]] or [(1e18, "")])[1]
    rows = []
    far = 0
    for r_, p in fb["levels"]:
        if not (price / 5.0 <= p <= price * 5.0):
            far += 1
            continue
        lid = f"F{r_:g}"
        rows.append(f'<tr><td>{"<b>" if lid == near else ""}Tuzatish {r_:g}{"</b>" if lid == near else ""}</td><td class="n">{usd_r(p)}</td><td class="n">{pc(p / price - 1, 1, True)}</td>'
                    f'<td>{"narxga eng yaqin" if lid == near else ""}</td>{_pcell(lt, lid, 60)}</tr>')
    for e_, p in fb["ext"]:
        lid = f"FE{e_:g}"
        if not (price / 5.0 <= p <= price * 5.0):
            far += 1
            continue
        rows.append(f'<tr><td>Kengayish {e_:g}</td><td class="n">{usd_r(p)}</td><td class="n">{pc(p / price - 1, 1, True)}</td><td>maqsad darajasi</td>{_pcell(lt, lid, 60)}</tr>')
    out.append(f'<p>Soʻnggi yirik toʻlqin ({esc(str(fb["deg"]))} daraja, {fb["bars"]} bar): <b>{fb["date_lo"]:%Y-%m-%d} {usd(fb["lo"])} → {fb["date_hi"]:%Y-%m-%d} {usd(fb["hi"])}</b> '
               f'({"yuqoriga" if fb["up"] else "pastga"} toʻlqin). Hozir narx shu toʻlqinning {fb["retr_now"] * 100:.0f}% qismini qaytargan.</p>')
    out.append(_table(["Daraja", "Narx", "Hozirgidan", "Izoh", "3 oyda yetish (MC)"], rows, [1, 2, 4], hide_mobile=[3]) if rows else
               '<p class="muted">Fibonachchi darajalarining hammasi hozirgi narxdan 5 baravardan uzoqda — jadvalga kiritilmadi.</p>')
    if far and rows:
        out.append(f'<p class="small muted">Hozirgi narxdan 5 baravardan uzoqdagi {far} ta daraja jadvalga kiritilmadi.</p>')
    out.append(f'<p class="small muted">Hisob: diapazon = {usd(fb["hi"])} − {usd(fb["lo"])} = {usd(rng)}. Tuzatish darajalari = '
               f'{"yuqori − r × diapazon" if fb["up"] else "past + r × diapazon"} (r = 0.236, 0.382, 0.5, 0.618, 0.786); kengayish = '
               f'{"past + e × diapazon" if fb["up"] else "yuqori − e × diapazon"} (e = 1.272, 1.618, 2.0). Fibonachchi darajalarining ish berishi isbotlanmagan; '
               f'ular boshqa manbalar bilan bir joyga tushsa (konfluens) “Qisqa xulosa”da kuchliroq hisoblanadi.</p>')
    return "".join(out)


def _near_high(R: Dict[str, Any]) -> bool:
    try:
        return float(R["price"]) >= 0.9 * float(R["levels"]["hi52"])
    except Exception:
        return True


def short_text(R: Dict[str, Any]) -> str:
    sh = R["short"]
    lab, up, dn, inv = sh["label"], sh["up"], sh["down"], sh["invalid"]

    def d(x: Dict[str, Any]) -> str:
        return f'{usd(x["price"])} ({pc(x["pct"], 1, True)})'
    s: List[str] = []
    if lab == "Ijobiy":
        s.append(f"Yoʻnalish ijobiy (ball {sh['score']:+.2f}).")
        if up:
            t_ = f"Yuqorida birinchi toʻsiq {d(up[0])}"
            if len(up) > 1:
                t_ += f"; undan oshib mustahkamlansa, hisoblangan maqsad {d(up[1])}"
            if len(up) > 2:
                t_ += f", uzoq maqsad {d(up[2])}"
            s.append(t_ + ".")
        else:
            s.append("Yuqorida hisoblangan toʻsiq yoki maqsad darajasi yoʻq" + (" (narx 52 haftalik yuqori atrofida)." if _near_high(R) else " (narxdan 3 baravar ichida daraja topilmadi)."))
        if inv:
            s.append(f"Narx {usd(inv['price'])} dan pastga yopilsa ({inv['why']}), ijobiy ssenariy bekor boʻladi.")
        if dn:
            s.append(f"Pastda birinchi tayanch {d(dn[0])}" + (f", undan keyin {d(dn[1])}" if len(dn) > 1 else "") + ".")
    elif lab == "Salbiy":
        s.append(f"Yoʻnalish salbiy (ball {sh['score']:+.2f}).")
        if dn:
            t_ = f"Pastda birinchi tayanch {d(dn[0])}"
            if len(dn) > 1:
                t_ += f"; undan pastga yopilsa, hisoblangan daraja {d(dn[1])}"
            s.append(t_ + ".")
        else:
            s.append("Pastda hisoblangan tayanch yoki maqsad darajasi yoʻq.")
        if inv:
            s.append(f"Narx {usd(inv['price'])} dan yuqoriga yopilsa ({inv['why']}), salbiy ssenariy bekor boʻladi.")
        if up:
            s.append(f"Yuqorida birinchi toʻsiq {d(up[0])}" + (f", undan keyin {d(up[1])}" if len(up) > 1 else "") + ".")
    else:
        s.append(f"Yoʻnalish neytral (ball {sh['score']:+.2f}): aniq ustunlik yoʻq.")
        if up and dn:
            s.append(f"Narx hozir {d(dn[0])} (pastda) va {d(up[0])} (yuqorida) oraligʻida; {usd(up[0]['price'])} dan yuqoriga yoki {usd(dn[0]['price'])} dan pastga "
                     f"yopilishi yoʻnalishni belgilaydi.")
        elif up:
            s.append(f"Yuqorida birinchi toʻsiq {d(up[0])}.")
        elif dn:
            s.append(f"Pastda birinchi tayanch {d(dn[0])}.")
    return " ".join(s)


def _ladder_table(items: List[Dict[str, Any]]) -> str:
    rows = []
    for x in items:
        srcs = " · ".join(x["src"])
        conf = f' <span class="tag good">{x["n_src"]} manba — konfluens</span>' if x["n_src"] >= 3 else (f' <span class="tag">{x["n_src"]} manba</span>' if x["n_src"] == 2 else "")
        rows.append(f'<tr><td style="white-space:normal"><b>{esc(x["tag"])}</b>{conf}<div class="small muted">{esc(srcs)}</div></td>'
                    f'<td class="n"><b>{usd_r(x["price"])}</b></td><td class="n">{pc(x["pct"], 1, True)}</td>'
                    f'<td class="n">{_p0(x.get("p20"))}</td><td class="n">{_p0(x.get("p60"))}</td></tr>')
    return _table(["Bosqich va asos", "Daraja", "Hozirgidan", "1 oyda yetish (MC)", "3 oyda yetish (MC)"], rows, [1, 2, 3, 4], wrap_first=True)


def sec_short(R: Dict[str, Any]) -> str:
    sh = R.get("short")
    if not sh:
        return ""
    t, price, con = R["ticker"], R["price"], R["conclusion"]
    lab = sh["label"]
    arrow = "▲" if lab == "Ijobiy" else "▼" if lab == "Salbiy" else "●"
    cls = "good" if lab == "Ijobiy" else "bad" if lab == "Salbiy" else "warn"
    out = [f'<h2 id="qisqa">Qisqa xulosa: {esc(t)} qayergacha koʻtarilishi yoki tushishi mumkin</h2>',
           f'<div class="card {cls}"><div class="row"><div><div class="lbl">Umumiy yoʻnalish</div><div class="big">{arrow} {esc(lab)}</div>'
           f'<div class="lbl">ball {con["score"]:+.2f} · ishonch: {esc(con["confidence"])}</div></div>'
           f'<div><div class="lbl">Hozirgi narx ({R["asof"]:%Y-%m-%d})</div><div class="big">{usd(price)}</div>'
           f'<div class="lbl">{sh["n_levels"]} ta hisoblangan daraja ichidan tanlandi</div></div></div>'
           f'<p>{esc(short_text(R))}</p></div>']
    out.append("<h3>▲ Yuqoriga: qaysi darajalargacha</h3>")
    out.append(_ladder_table(sh["up"]) if sh["up"] else ('<p class="muted">Yuqorida hisoblangan daraja yoʻq: narx 52 haftalik yuqori atrofida, tarixiy qarshilik, '
                                                         'baza/pattern maqsadi va Fibonachchi kengayishi topilmadi.</p>' if _near_high(R) else
                                                         '<p class="muted">Yuqorida narxdan 3 baravar ichida hisoblangan daraja (qarshilik, baza/pattern maqsadi, Fibonachchi kengayishi) topilmadi.</p>'))
    out.append("<h3>▼ Pastga: qaysi darajalargacha</h3>")
    out.append(_ladder_table(sh["down"]) if sh["down"] else '<p class="muted">Pastda hisoblangan tayanch yoki maqsad daraja topilmadi.</p>')
    if sh["invalid"]:
        out.append(f'<p><b>Bekor qilish darajasi:</b> {usd(sh["invalid"]["price"])} ({pc(sh["invalid"]["price"] / price - 1, 1, True)}) — '
                   f'{esc(sh["invalid"]["why"])}. Narx bu darajadan {"pastga" if lab == "Ijobiy" else "yuqoriga"} yopilsa, yuqoridagi xulosa oʻz kuchini yoʻqotadi.</p>')
    out.append(chart_block(R, "ladder", f"{t}: hisoblangan darajalar"))
    ai_counted = bool(con.get("ai_counted"))
    ai_note = ("Yengil rejim: AI, fundamental va yangilik tahlili qoʻshilmagan — xulosa faqat texnik qoidalarga asoslangan. " if R.get("lite")
               else "AI bloki xulosaga qoʻshildi (modelning tarixiy sinovi kamida zaif dalil koʻrsatgan). " if ai_counted
               else "AI bloki bu xulosaga qoʻshilmagan (modelning tarixiy ustunligi isbotlanmagan). ")
    out.append('<p class="small muted"><b>Bu — hisoblangan darajalar xaritasi, bashorat emas.</b> Har bir daraja aniq qoidadan chiqadi (manba “asos” ustunida: tarixiy support/resistance, '
               'baza measured-move, pattern maqsadi, Fibonachchi, Elliott, talab/taklif zonasi, trend kanali, 52 haftalik chegara). Bir nechta mustaqil manba bir joyni koʻrsatsa '
               '(konfluens) daraja kuchliroq hisoblanadi, lekin bu kafolat emas. «Yetish ehtimoli» — faqat volatillikdan hisoblangan Monte-Karlo (drift — bozor-prior): u '
               'darajaning “ish berishini” emas, narxning oʻsha yerga borishi statistik qanchalik mumkinligini koʻrsatadi. Texnik qoidalarning (pattern, Elliott, zona, chiziq) '
               'bashorat kuchi bu yerda tekshirilmagan. ' + esc(ai_note) + 'Moliyaviy maslahat emas.</p>')
    return "".join(out)


def _usd_short(v: float) -> str:
    if v >= 1e9:
        return f"${v / 1e9:.1f} mlrd"
    if v >= 1e6:
        return f"${v / 1e6:.1f} mln"
    if v >= 1e3:
        return f"${v / 1e3:.0f} ming"
    return f"${v:.0f}"


def caution_note(R: Dict[str, Any]) -> str:
    """Texnik tahlil ishonchini pasaytiradigan holatlar: past likvidlik, penny stock, qisqa tarix, birja belgisi."""
    msgs: List[str] = []
    dv = R.get("dvol")
    if dv is not None and np.isfinite(dv) and dv < 1e6:
        msgs.append(f"oʻrtacha kunlik savdo hajmi atigi ≈ {_usd_short(dv)} (oxirgi 20 kun mediani): likvidlik past — narx sakrashlari keng, "
                    f"texnik darajalar va Monte-Karlo ishonchsizroq")
    if R["price"] < 1.0:
        msgs.append("narx $1 dan past (penny stock): oʻta volatil, chiziq va zonalar tez buziladi")
    n = len(R["df"])
    if n < 260:
        msgs.append(f"narx tarixi qisqa ({n} savdo kuni): 200 kunlik oʻrtacha va 52 haftalik koʻrsatkichlar toʻliq emas, patternlar kam sinalgan")
    if R.get("flag"):
        msgs.append(str(R["flag"]))
    if not msgs:
        return ""
    return '<div class="card warn small"><b>Ehtiyot boʻling:</b> ' + "; ".join(esc(m) for m in msgs) + ".</div>"


def lite_notice(R: Dict[str, Any]) -> str:
    link = (f' <a href="{esc(FULL_ANALYSIS_URL)}" target="_blank" rel="noopener noreferrer">Toʻliq tahlilni soʻrash</a>.' if FULL_ANALYSIS_URL else "")
    return ('<div class="card warn small"><b>Yengil tahlil.</b> Bu aksiya uchun faqat narx tarixiga asoslangan texnik tahlil hisoblangan (support/resistance, zonalar, chiziqlar, '
            'patternlar, Elliott, Fibonachchi, Wyckoff, baza, Monte-Karlo). AI ball, fundamental koʻrsatkichlar va yangiliklar yoʻq.' + link + '</div>')


def render_analysis_body(R: Dict[str, Any], st: Any) -> str:
    t, df, last, price = R["ticker"], R["df"], R["last"], R["price"]
    con = R["conclusion"]
    fund = R["fund"]
    name = (fund.get("shortName") if fund.get("ok") else None) or R.get("name")
    chips = "".join(f'<span class="tag">{esc(k)}: {pc(v, 1, True)}</span>' for k, v in R["chg"].items())
    lab = con["label"]
    arrow = "▲" if lab == "Ijobiy" else "▼" if lab == "Salbiy" else "●"
    cls = "good" if lab == "Ijobiy" else "bad" if lab == "Salbiy" else "warn"
    out = [f'<div class="card info"><div class="row"><div><div class="lbl">{esc(t)}{" · " + esc(name) if name else ""}</div>'
           f'<div class="big">{usd(price)}</div><div class="lbl">{R["asof"]:%Y-%m-%d} dagi yopilish narxi</div><div class="chips">{chips}</div></div>'
           f'<div class="card {cls}" style="margin:0;min-width:230px"><div class="lbl">Umumiy xulosa</div><div class="big">{arrow} {esc(lab)}</div>'
           f'<div class="lbl">ball {con["score"]:+.2f} · ishonch: {esc(con["confidence"])}</div>'
           f'<div class="lbl"><a href="#qisqa">↓ Qisqa xulosa: qayergacha?</a></div></div></div></div>']
    if R.get("lite"):
        out.append(lite_notice(R))
    elif not R["in_universe"]:
        out.append('<div class="card warn small">Bu aksiya AI universumiga kirmaydi: model unga ham qoʻllanadi, lekin u asosan yirik, likvid aksiyalarda '
                   'oʻqitilgan — kichik yoki juda volatil kompaniyalar uchun natija ishonchsizroq.</div>')
    out.append(caution_note(R))
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
               '(trend, momentum, Wyckoff, RSI, sham, Gann, trend chizigʻi, chart-pattern, Elliott, fundamental, yangilik) klassik qoidalar boʻlib, ularning bashorat kuchi '
               'bu yerda tekshirilmagan — shuning uchun ularga kichik ogʻirlik berilgan. Chart-pattern va Elliott faqat tasodifiy narxdan statistik ajralib tursagina '
               '(p ≤ 5% va ≤ 10%) qoʻshiladi, aks holda faqat maʼlumot sifatida koʻrsatiladi.</p>')
    # --- grafik
    if R.get("lite"):
        out.append("<h2>Narx grafigi: talab/taklif zonalari, support/resistance va trend chiziqlari</h2>")
        out.append(chart_block(R, "tech", f"{t} narx grafigi"))
    else:
        out.append("<h2>Narx grafigi, support/resistance, Gann yelpigʻichi va Wyckoff hodisalari</h2>")
        out.append(chart_block(R, "main", f"{t} narx grafigi"))
    # --- AI
    n_ai = len(out)
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
    if R.get("lite"):
        del out[n_ai:]
    # --- Monte-Karlo
    mc = R["mc"]
    t1, t2 = _mc_tables(R)
    out.append("<h2>Narx qaysi diapazonda boʻlishi mumkin (Monte-Karlo)</h2>")
    out.append(f'<p class="small muted">Kunlik volatillik {mc["sigma_d"] * 100:.2f}% (yillik {mc["sigma_ann"] * 100:.0f}%). '
               f'Bu AI emas — statistik “volatillik konusi”: drift bozor-prior ({MC_DRIFT * 252 * 100:.0f}% yillik), aksiyaning oʻz oʻtmish daromadi emas; '
               f'zarbalar ogʻir dumli (t-taqsimot).</p>')
    out.append("" if R.get("lite") else cone_block(R))
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
    elif price >= 0.9 * lv["hi52"]:
        sc.append(f'<li>Narx 52 haftalik yuqori ({usd(lv["hi52"])}) atrofida: yuqorida tarixiy qarshilik yoʻq (narx “kashf etilmagan” hududda).</li>')
    else:
        sc.append(f'<li>Yuqorida narxdan 3 baravar ichida tarixiy qarshilik darajasi topilmadi (52 haftalik yuqori {usd(lv["hi52"])} — hozirgi narxdan ancha uzoqda).</li>')
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
    # --- texnik chizmalar: zonalar/chiziqlar, patternlar, Elliott, Fibonachchi
    out.append(sec_zones(R))
    out.append(sec_patterns(R))
    out.append(sec_elliott(R))
    out.append(sec_fib(R))
    # --- fundamental
    n_f = len(out)
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
    if R.get("lite"):
        del out[n_f:]
    out.append(sec_short(R))
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

**1. Maʼlumot.** AI universumi: ~170 ta yirik AQSh aksiyasi (NYSE/NASDAQ) va SPY indeksining 8 yillik kunlik narx tarixi (Yahoo Finance; zaxira — Stooq). Qolgan barcha aksiyalar uchun 12-bandga qarang.
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

**10. Texnik chizmalar: zonalar, chiziqlar, patternlar, Fibonachchi, Elliott.** Hammasi mexanik qoidalar bilan, log-narxda hisoblanadi (har bir hisob sahifada koʻrsatiladi):
**ZigZag pivotlar** (chegara = 1.5·σ·√L; L = 6 / 20 / 60 kun — kichik / oʻrta / katta daraja). **Trend chiziqlari** — ikki tasdiqlangan pivot orqali; ikkinchi tayanchgacha
hech bir YOPILISH chiziqning narigi tomonida boʻlmasligi shart; teginish — chiziqqa 0.6·ATR dan yaqin kelgan pivot (kamida 4 bar oraliq). **Kanal** — chiziqqa parallel ikkinchi chiziq;
**regressiya kanali** — oxirgi 100 bar log-narxiga toʻgʻri chiziq ±2σ. **Talab/taklif zonalari** — kuchli portlovchi shamdan (tana ≥ 1.4·ATR) oldingi 1–4 ta tor sham diapazoni; narx YOPILISH bilan
teshsa oʻchiriladi. **Fibonachchi** — soʻnggi yirik toʻlqinning tuzatish (23.6–78.6%) va kengayish (127.2 / 161.8 / 200%) darajalari. **Chart-patternlar** — ikkilamchi choʻqqi/tub,
bosh-yelkalar, uchburchaklar, ponalar, bayroq/vimpel: holati (shakllanmoqda / tasdiqlangan), tasdiq darajasi, maqsadi (measured move) va bekor qilish darajasi hisoblanadi.
**Elliott** — QATʼIY qoidalarga mos impuls (1-2-3-4-5) yoki ABC sanogʻi va Fibonachchi moslik bali. **Halol eslatma:** bu detektorlar tasodifiy narxda ham tez-tez ishlaydi
(masalan, qoidaga mos Elliott sanogʻi tasodifiy narx oynalarining turiga qarab 14–76% ida topiladi), shuning uchun har bir topilma yonida “tasodifiy narxda shunday sifat necha foiz oynalarda chiqadi”
koʻrsatilgan. Bir nechta variantdan ENG YAXSHISINI tanlash p-qiymatni sunʼiy kichraytiradi (tasodifiy narxda ham oynalarning ~35% ida kamida bitta pattern “yaxshi” koʻrinadi), shuning uchun
ikkinchi, tanlov hisobga olingan (family-wise) p-qiymat ham koʻrsatiladi (tasodifiy yurishning 12 000 ta oynasida hisoblangan jadval) va umumiy xulosaga FAQAT shu koʻrsatkich ≤ 5% boʻlganlari
(pattern: va sifat ≥ 0.6; Elliott: 5–6 pivotli impuls) kichik ogʻirlik bilan qoʻshiladi — tasodifiy narxda bu hodisa ≤ 5% oynada yuz beradi. Elliott sanogʻi subyektiv:
bitta grafikka bir nechta sanoq mos kelishi mumkin, bu yerda faqat mexanik variant olinadi. Bu bloklarning real bozordagi bashorat kuchi **tekshirilmagan**.

**11. «Qisqa xulosa» (har bir sahifa oxirida).** Barcha nomzod darajalar (support/resistance, baza maqsadi, pattern maqsadi, Fibonachchi, Elliott, zona, kanal, 52 haftalik chegara) aniq qoidadan chiqadi;
bir-biriga 1.5% dan yaqin darajalar birlashtiriladi (konfluens: bir nechta mustaqil manba bir joyni koʻrsatsa — daraja kuchliroq). Yuqoriga va pastga “narvon” tuziladi, yoniga Monte-Karlo bilan
1 va 3 oyda shu darajaga yetish ehtimoli qoʻyiladi, xulosa bekor boʻladigan daraja koʻrsatiladi. Bu — hisoblangan darajalar xaritasi: bitta nuqtali narx bashorati yoki “taxmin” yoʻq; hozirgi narxdan
3 baravardan uzoq darajalar kiritilmaydi.

**12. Qamrov: toʻliq va yengil tahlil.** AI universumi va qoʻshimcha tikerlar uchun *toʻliq* tahlil (AI ball, fundamental, yangilik, barcha texnik bloklar). Qolgan barcha NYSE / NASDAQ / NYSE American oddiy aksiyalari va yirik ETFlar
(roʻyxat har kuni nasdaqtrader.com dan olinadi) uchun *yengil* tahlil: faqat narx tarixidan (2.5 yil) — texnik bloklar, Monte-Karlo va «Qisqa xulosa»; AI, fundamental va yangilik yoʻq
(ular har bir aksiya uchun alohida soʻrov talab qiladi, minglab aksiya uchun sigʻmaydi). Yahoo narx bermagan, 120 savdo kunidan kam tarixli yoki savdosi toʻxtagan tikerlar saytda boʻlmaydi.
Past likvidlik (oʻrtacha kunlik savdo < $1 mln), narx < $1 yoki qisqa tarix boʻlsa, sahifada ogohlantirish chiqadi.

## Cheklovlar (halol)

* **Mukammal AI yoʻq.** Bozor raqobatli: oddiy, ochiq maʼlumotdagi ustunliklar odatda juda kichik yoki yoʻq.
* **Survivorship bias:** universum — bugungi yirik kompaniyalar; tushib ketganlari yoʻq. Backtest natijalari odatda yuqoriroq koʻrinadi.
* **Kichik va illikvid aksiyalar** (masalan, $12 lik kichik kompaniya) uchun AI model oʻqitilmagan — ishonchsizroq; ularning texnik darajalari ham tez-tez buziladi (past likvidlik ogohlantirishi chiqadi).
* **Texnik bloklar** (zona, chiziq, pattern, Fibonachchi, Elliott) tasodifiy narxda ham “topiladi” va ularning bashorat kuchi tekshirilmagan: ular hisoblangan darajalar xaritasini beradi, kelajakni bashorat qilmaydi.
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


def _st_path_df(knots: List[Tuple[int, float]], noise: float = 0.004, seed: int = 0) -> pd.DataFrame:
    """Tugun nuqtalaridan (bar, narx) log-chiziqli yoʻl + kichik shovqin: ideal chizma shakllarini sintez qilish (selftest uchun)."""
    rng = np.random.default_rng(seed)
    bars = [k[0] for k in knots]
    lp = np.interp(np.arange(bars[-1] + 1), bars, np.log([k[1] for k in knots]))
    n = len(lp)
    lp = lp + rng.normal(0, noise, n)
    close = np.exp(lp)
    opn = np.empty(n)
    opn[0] = close[0]
    opn[1:] = close[:-1] * np.exp(rng.normal(0, noise * 0.3, n - 1))
    hi = np.maximum(opn, close) * np.exp(np.abs(rng.normal(0, noise * 0.9, n)))
    lo = np.minimum(opn, close) * np.exp(-np.abs(rng.normal(0, noise * 0.9, n)))
    return pd.DataFrame({"Open": opn, "High": hi, "Low": lo, "Close": close, "Volume": np.exp(rng.normal(14, 0.2, n))},
                        index=pd.bdate_range("2021-01-04", periods=n))


def _st_walk_df(n: int, seed: int, sig: float = 0.018) -> pd.DataFrame:
    """Tasodifiy yurish (t5 zarbalar) — texnik detektorlarning yolgʻon ishlash chastotasini oʻlchash uchun."""
    rng = np.random.default_rng(seed)
    ret = rng.standard_t(5, n) / math.sqrt(5 / 3) * sig + 0.0003
    close = 100 * np.exp(np.cumsum(ret))
    opn = np.r_[close[0], close[:-1]]
    hi = np.maximum(opn, close) * np.exp(np.abs(rng.normal(0, 0.45 * sig, n)))
    lo = np.minimum(opn, close) * np.exp(-np.abs(rng.normal(0, 0.45 * sig, n)))
    return pd.DataFrame({"Open": opn, "High": hi, "Low": lo, "Close": close, "Volume": np.exp(rng.normal(14, 0.3, n))},
                        index=pd.bdate_range("2020-01-02", periods=n))


def _selftest_ta(rep: Callable[[str, bool, str], None]) -> None:
    """Texnik tahlil bloklari: ideal shakllar topiladi, tasodifiy narxda xulosaga yolgʻon qoʻshilish kam, qisqa xulosa izchil."""
    # (a) ideal chart-patternlar
    shapes = (
        ("double_top", [(0, 70), (150, 140), (175, 126), (200, 139), (215, 124), (222, 121)], 2),
        ("double_bottom", [(0, 200), (150, 100), (175, 114), (200, 101), (215, 118), (222, 122)], 3),
        ("hs_top", [(0, 70), (150, 130), (165, 118), (185, 145), (205, 119), (225, 131), (240, 117), (250, 110)], 4),
        ("hs_bottom", [(0, 200), (150, 130), (165, 142), (185, 112), (205, 141), (225, 129), (240, 142), (250, 150)], 6),
    )
    got = []
    for kind, knots, sd in shapes:
        df = _st_path_df(knots, seed=sd)
        items = find_patterns(df, compute_pivots(df))["items"]
        got.append((kind, any(it["kind"] == kind and it["state"] == PAT_STATE_OK for it in items)))
    rep("Ideal chart-patternlar (ikkilamchi choʻqqi/tub, bosh-yelkalar) topiladi", all(g for _, g in got),
        ", ".join(f"{k}:{'ha' if g else 'YOʻQ'}" for k, g in got))
    # (b) ideal Elliott impuls (4-toʻlqin davom etmoqda): qoidalar, yoʻnalish, bekor qilish darajasi
    w1 = 0.18
    y = [math.log(100)]
    y += [y[0] + w1, y[0] + w1 - 0.6 * w1]
    y += [y[2] + 1.618 * w1]
    y += [y[3] - 0.38 * (y[3] - y[2])]
    y += [y[4] + w1]
    ek = [(0, 130.0)] + [(b, math.exp(v)) for b, v in zip((60, 80, 92, 125, 140, 160), y)]
    df = _st_path_df(ek, seed=1).iloc[:144]
    b = find_elliott(df, compute_pivots(df)).get("best")
    rep("Ideal Elliott impuls (1-2-3-4, 4-toʻlqin davom etmoqda) topiladi",
        bool(b) and b["kind"] == "i" and b["k"] >= 5 and b["dir"] == 1 and b["score"] >= 0.9 and b["invalid"] < float(df["Close"].iloc[-1]),
        (f"k={b['k']}, yoʻnalish={'↑' if b['dir'] > 0 else '↓'}, moslik {b['score']:.2f}, bekor qilish {b['invalid']:.1f}, tanlov hisobga olingan p={b['p_fw'] * 100:.1f}%" if b else "topilmadi"))
    # (c) ideal oʻsish kanali
    n_ = 300
    xs = np.arange(n_)
    lp = np.log(50) + 0.0025 * xs + 0.05 * np.sin(2 * np.pi * xs / 40)
    rng = np.random.default_rng(3)
    cl = np.exp(lp + rng.normal(0, 0.003, n_))
    op = np.r_[cl[0], cl[:-1]]
    dfc = pd.DataFrame({"Open": op, "High": np.maximum(op, cl) * 1.004, "Low": np.minimum(op, cl) * 0.996, "Close": cl, "Volume": np.full(n_, 1e6)},
                       index=pd.bdate_range("2021-01-04", periods=n_))
    tl = find_trendlines(dfc, compute_pivots(dfc))
    ch = tl.get("channel")
    rep("Ideal oʻsish kanali: trend va kanal chizigʻi topiladi",
        bool(ch) and tl.get("trend") == "oʻsish" and ch["touches_top"] >= 4 and ch["touches_bot"] >= 4,
        f"trend={tl.get('trend')}, teginishlar yuqori/past: {ch['touches_top']}/{ch['touches_bot']}" if ch else "kanal topilmadi")
    # (d) tasodifiy narxda texnik bloklarning xulosaga "yolgʻon" qoʻshilish chastotasi (tanlov hisobga olingan p bilan <= ~5%)
    n_w = 240
    ew_c = pat_c = 0
    for sd in range(n_w):
        d = _st_walk_df(450, 70000 + sd)
        T = compute_pivots(d)
        bb = find_elliott(d, T).get("best")
        ew_c += int(bool(bb) and bb["kind"] == "i" and bb["k"] >= 5 and np.isfinite(bb["p_fw"]) and bb["p_fw"] <= 0.05 and bb["bias"] != 0)
        its = find_patterns(d, T)["items"]
        sg = lambda x: x["bias"] if x["bias"] != 0 else (x["dir"] if x["state"] == PAT_STATE_OK else 0)
        el = [x for x in its if x["quality"] >= 0.6 and sg(x) != 0 and np.isfinite(pat_p_rand(x["kind"], x["quality"]))]
        if el:
            pt = min(el, key=lambda x: pat_p_rand(x["kind"], x["quality"]))
            pat_c += int(fw_adjust(PAT_FW, pat_p_rand(pt["kind"], pt["quality"])) <= 0.05)
    rep("Tasodifiy narxda Elliott/pattern xulosaga 'yolgʻon' qoʻshilmaydi (≤ ~5%, tanlov hisobga olingan)",
        ew_c / n_w <= 0.10 and pat_c / n_w <= 0.10,
        f"{n_w} ta tasodifiy oynada: Elliott {ew_c / n_w * 100:.1f}%, pattern {pat_c / n_w * 100:.1f}% (tanlov hisobga olinmasa ~23% va ~31% boʻlardi)")
    # (e) "Qisqa xulosa" izchilligi va hisobotda taqiqlangan soʻzlar yoʻqligi
    bad: List[str] = []
    for sd in range(8):
        d = _st_walk_df(300 + 40 * sd, 81000 + sd, sig=0.012 + 0.004 * (sd % 4))
        mk = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(900 + sd).normal(0.0003, 0.01, len(d)))), index=d.index)
        try:
            R = analyze_lite(f"T{sd}", d, mk, name="Test Co.", flag=None)
        except Exception as e:                                       # noqa: BLE001
            bad.append(f"#{sd}: {type(e).__name__}: {e}")
            continue
        sh, px = R["short"], float(R["price"])
        ups, dns = [x["price"] for x in sh["up"]], [x["price"] for x in sh["down"]]
        if not (all(u > px for u in ups) and ups == sorted(ups)):
            bad.append(f"#{sd}: yuqori darajalar tartibsiz")
        if not (all(u < px for u in dns) and dns == sorted(dns, reverse=True)):
            bad.append(f"#{sd}: pastki darajalar tartibsiz")
        if any(not (0.0 <= x["p60"] <= 1.0) for x in sh["up"] + sh["down"]):
            bad.append(f"#{sd}: ehtimol [0,1] dan tashqarida")
        txt = re.sub(r"<[^>]+>", " ", render_analysis_body(R, LiteState()))
        if "Qisqa xulosa" not in txt or re.search(r"(?<![A-Za-z])(nan|inf)(?![A-Za-z])|taxminan", txt):
            bad.append(f"#{sd}: hisobotda 'Qisqa xulosa' yoʻq yoki nan/inf/taxminan bor")
    rep("Qisqa xulosa izchil (darajalar tartibi, ehtimollar, hisobotda nan/«taxminan» yoʻq)", not bad, "8 ta sintetik aksiya OK" if not bad else "; ".join(bad[:4]))


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
    _selftest_ta(rep)                                                # (7) texnik tahlil: ideal shakllar, tasodifiy narxda yolgʻon ishlash chastotasi, qisqa xulosa
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
