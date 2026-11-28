# -*- coding: utf-8 -*-
"""
AKSIYA AI — statik sayt quruvchi (GitHub Pages uchun: bepul va doim ochiq)
==========================================================================
Dvigatel (app.py) bilan bir marta hisoblab, har bir aksiya uchun tayyor HTML sahifa yozadi.
Sayt serversiz ishlaydi: telefon va kompyuter brauzerida darhol ochiladi; GitHub Actions uni har kuni yangilaydi.

    python build_site.py --out site                      # real maʼlumot (Yahoo; zaxira Stooq)
    python build_site.py --out site --synthetic          # internetsiz sinov (sintetik bozor)
    python build_site.py --save-extras "PLTR, SOFI"      # tikerlarni extra_tickers.txt ga qoʻshish

Qamrov: (1) TOʻLIQ tahlil — AI universumi (~175 yirik aksiya) va extra_tickers.txt: AI ball, fundamental, yangilik, texnik bloklar;
(2) YENGIL tahlil — NYSE/NASDAQ/NYSE American dagi QOLGAN BARCHA oddiy aksiyalar (+ yirik ETFlar): narx tarixidan hisoblangan texnik tahlil
(zonalar, trend chiziqlari, patternlar, Elliott, Fibonachchi, Monte-Karlo va «Qisqa xulosa»). Roʻyxat har kuni nasdaqtrader.com dan olinadi.
Yengil bosqich muvaffaqiyatsiz boʻlsa ham, toʻliq sahifalar bilan sayt joylanadi.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html as _h
import json
import math
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import app
from app import CSS, DISCLAIMER, HORIZON_UZ, WYCKOFF_UZ, esc, pc, usd

MAX_EXTRA = 60                                   # extra_tickers.txt dagi eng koʻp tiker
EXTRA_FILE = "extra_tickers.txt"
WORKFLOW_FILE = "build-site.yml"
TZ_TASH = dt.timezone(dt.timedelta(hours=5))     # Toshkent / Samarqand vaqti
WY_SHORT = {0: "—", 1: "Accumulation", 2: "Markup", 3: "Distribution", 4: "Markdown"}


# ======================================================================================
# 1. TIKERLAR RUXSATI
# ======================================================================================
def parse_tickers(text: str) -> List[str]:
    out: List[str] = []
    for line in (text or "").splitlines():
        line = line.split("#", 1)[0]
        for tok in re.split(r"[,\s;]+", line.strip()):
            if not tok:
                continue
            try:
                t = app.normalize_ticker(tok)
            except app.DataError:
                app.log(f"Tiker oʻtkazib yuborildi (notoʻgʻri): {tok!r}")
                continue
            if t not in out:
                out.append(t)
    return out


def read_extra(path: str = EXTRA_FILE, env_text: str = "") -> List[str]:
    txt = ""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            txt = f.read()
    allt = parse_tickers(txt)
    for t in parse_tickers(env_text):
        if t not in allt:
            allt.append(t)
    if len(allt) > MAX_EXTRA:
        app.log(f"Diqqat: extra tikerlar {len(allt)} ta; faqat birinchi {MAX_EXTRA} tasi olinadi.")
    return allt[:MAX_EXTRA]


def save_extras(new_text: str, path: str = EXTRA_FILE) -> List[str]:
    """Yangi tikerlarni extra_tickers.txt oxiriga qoʻshadi (mavjudlari takrorlanmaydi, MAX_EXTRA dan oshmaydi)."""
    cur = parse_tickers(open(path, encoding="utf-8").read()) if os.path.exists(path) else []
    added: List[str] = []
    for t in parse_tickers(new_text):
        if t not in cur and len(cur) + len(added) < MAX_EXTRA:
            added.append(t)
    if added:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n# Actions orqali qoʻshilgan (%s)\n%s\n" % (dt.date.today().isoformat(), " ".join(added)))
    return added


# ======================================================================================
# 2. FUNDAMENTAL VA YANGILIKLARNI OLDINDAN (PARALLEL) YUKLASH
# ======================================================================================
_FUND: Dict[str, Dict[str, Any]] = {}
_NEWS: Dict[str, Dict[str, Any]] = {}
_RATE = re.compile(r"too many|429|rate", re.I)


def _retry(fn, t: str, tries: int = 3) -> Dict[str, Any]:
    r: Dict[str, Any] = {}
    for k in range(tries):
        r = fn(t)
        if r.get("ok") or not _RATE.search(str(r.get("error", ""))):
            return r
        time.sleep(4 * (k + 1))
    return r


def prefetch_info(tickers: List[str], offline: bool = False, workers: int = 6) -> None:
    """analyze_ticker ichidagi fetch_fundamentals/fetch_news ni keshli versiyaga almashtiradi (tarmoq 1 marta, parallel)."""
    orig_f, orig_n = app.fetch_fundamentals, app.fetch_news
    if offline:
        for t in tickers:
            _FUND[t] = dict(ok=False, error="oflayn (sintetik) rejim")
            _NEWS[t] = dict(ok=False, items=[], error="oflayn (sintetik) rejim")
    else:
        fails = {"n": 0}
        stop = threading.Event()

        def job(t: str) -> None:
            if stop.is_set():
                _FUND[t] = dict(ok=False, error="Yahoo cheklovi (sayt qurilishida)")
                _NEWS[t] = dict(ok=False, items=[], error="Yahoo cheklovi (sayt qurilishida)")
                return
            f = _retry(orig_f, t)
            n = _retry(orig_n, t)
            _FUND[t], _NEWS[t] = f, n
            if not f.get("ok") and not n.get("ok"):
                fails["n"] += 1
                if fails["n"] >= 15 and not any(v.get("ok") for v in list(_FUND.values())[:60]):
                    stop.set()                       # Yahoo butunlay yopiq: qolganlarini kutib oʻtirmaymiz
            else:
                fails["n"] = 0

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(job, tickers))
        ok_f = sum(1 for v in _FUND.values() if v.get("ok"))
        ok_n = sum(1 for v in _NEWS.values() if v.get("ok") and v.get("items"))
        app.log(f"Fundamental {ok_f}/{len(tickers)}, yangilik {ok_n}/{len(tickers)} ({time.time() - t0:.0f} s)")
    if offline:                                       # sintetik rejimda tarmoqqa umuman chiqilmaydi
        app.fetch_fundamentals = lambda t: _FUND.get(t) or dict(ok=False, error="oflayn (sintetik) rejim")
        app.fetch_news = lambda t, n=8: _NEWS.get(t) or dict(ok=False, items=[], error="oflayn (sintetik) rejim")
    else:
        app.fetch_fundamentals = lambda t: _FUND.get(t) or orig_f(t)
        app.fetch_news = lambda t, n=8: _NEWS.get(t) or orig_n(t, n)


# ======================================================================================
# 2b. CHEKLANMAGAN REJIM, 1-QISM: BARCHA NYSE/NASDAQ AKSIYALARI RUYXATI VA NARX TARIXINI YUKLASH
# ======================================================================================
LITE_YEARS = 2.5                                   # yengil tahlil uchun narx tarixi (yil): ~630 savdo kuni
SYMDIR_URLS = {
    "nasdaq": ["https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
               "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt"],
    "other": ["https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
              "ftp://ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt"],
}
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
EX_OK = {"Q": "NASDAQ", "N": "NYSE", "A": "NYSE American"}       # roʻyxatga olinadigan birjalar
# Aksiya boʻlmagan qimmatli qogʻozlar (varrant, huquq, imtiyozli aksiya, obligatsiya, fond) nomidan aniqlanadi
_BAD_NAME = re.compile(r"\b(warrants?|rights?|preferred|pfd|notes?|debentures?|subordinated|funds?|etn|etf)\b|%", re.I)
_NAME_TAIL = re.compile(r"\s*(?:-\s*)?(?:(?:Class|Series)\s+[A-Z0-9]+\s+)?(?:Common\s+Stock|Common\s+Shares?|Ordinary\s+Shares?|Capital\s+Stock|"
                        r"Shares\s+of\s+Beneficial\s+Interest|Limited\s+Partnership|Common\s+Units?|American\s+Depositary\s+(?:Shares?|Receipts?)|"
                        r"Depositary\s+Shares?|Class\s+[A-Z]\b).*$", re.I)
# Eng koʻp soʻraladigan yirik ETFlar (ETF belgisi bor qatorlar faqat shu roʻyxatdan olinadi)
CURATED_ETFS = ("SPY QQQ DIA IWM VTI VOO IVV VEA VWO EFA EEM IEMG AGG BND TLT IEF SHY LQD HYG GLD SLV IAU USO UNG DBC "
                "XLK XLF XLE XLV XLY XLP XLI XLU XLB XLRE XLC SMH SOXX XBI IBB KRE KBE ITB XHB XOP OIH GDX GDXJ "
                "ARKK ARKG VNQ VIG SCHD VYM VT VXUS IJH IJR MDY RSP QQQM SPLG SCHG VUG VTV VB VO "
                "TQQQ SQQQ SOXL SOXS SPXL SPXS UPRO TNA TZA UVXY VXX SVXY IBIT FBTC BITO ETHA EWJ EWZ FXI KWEB MCHI INDA EWY EWT").split()
_SYM_OK = re.compile(r"^[A-Z0-9]{1,6}(?:[.\-][A-Z0-9]{1,2})?$")
_SYM_BAD_TAIL = re.compile(r"-(WS|WT|RT|U|R|W|PR[A-Z]?)$")
_LITE_UA = "Mozilla/5.0 (compatible; AksiyaAI/%s; +https://github.com/%s)"


def sym_to_yahoo(sym: str) -> Optional[str]:
    """Birja yozuvi (BRK.B) -> Yahoo (BRK-B). Imtiyozli aksiya/varrant shaklidagi yozuvlar (BAC$L, ABC.WS) rad etiladi."""
    s = (sym or "").strip().upper()
    if not _SYM_OK.match(s):
        return None
    s = s.replace(".", "-")
    return None if _SYM_BAD_TAIL.search(s) else s


def clean_name(name: str) -> str:
    """'Apple Inc. - Common Stock' -> 'Apple Inc.'; juda uzun nomlar qisqartiriladi."""
    n = re.sub(r"\s+", " ", (name or "").strip())
    m = _NAME_TAIL.sub("", n).strip(" -,")
    n = m if len(m) >= 2 else n
    return n if len(n) <= 44 else n[:43].rstrip(" ,.-") + "…"


def _pipe_rows(txt: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in (txt or "").splitlines():
        if not line.strip():
            continue
        p = [x.strip() for x in line.split("|")]
        head = p[0].lower()
        if head in ("symbol", "act symbol") or head.startswith("file creation time"):
            continue
        rows.append(p)
    return rows


def parse_nasdaqlisted(txt: str) -> List[Dict[str, Any]]:
    """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares"""
    out = []
    for p in _pipe_rows(txt):
        if len(p) < 7:
            continue
        out.append(dict(sym=p[0], name=p[1], ex="Q", test=(p[3] == "Y"), st=(p[4] or "N"), etf=(p[6] == "Y")))
    return out


def parse_otherlisted(txt: str) -> List[Dict[str, Any]]:
    """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol"""
    out = []
    for p in _pipe_rows(txt):
        if len(p) < 7:
            continue
        out.append(dict(sym=p[0], name=p[1], ex=p[2], test=(p[6] == "Y"), st="N", etf=(p[4] == "Y")))
    return out


def parse_sec_json(obj: Any) -> List[Dict[str, Any]]:
    """Zaxira manba: SEC company_tickers_exchange.json  ({fields:[cik,name,ticker,exchange], data:[[...]]})."""
    out = []
    try:
        f = obj["fields"]
        ix = {k: f.index(k) for k in ("name", "ticker", "exchange")}
        for r in obj["data"]:
            ex = {"nasdaq": "Q", "nyse": "N"}.get(str(r[ix["exchange"]]).lower())
            if ex:
                out.append(dict(sym=str(r[ix["ticker"]]), name=str(r[ix["name"]]), ex=ex, test=False, st="N", etf=False))
    except Exception:
        return []
    return out


def _http_text(url: str, timeout: float = 30.0, tries: int = 3) -> Optional[str]:
    ua = _LITE_UA % (app.APP_VERSION, os.environ.get("GITHUB_REPOSITORY", "aksiya-ai"))
    for k in range(tries):
        try:
            if url.startswith("ftp://"):
                import urllib.request
                with urllib.request.urlopen(url, timeout=timeout) as r:
                    txt = r.read().decode("utf-8", "replace")
            else:
                import requests
                r = requests.get(url, timeout=timeout, headers={"User-Agent": ua})
                if r.status_code != 200:
                    app.log(f"  {url}: HTTP {r.status_code}")
                    txt = ""
                else:
                    txt = r.text
            if txt and txt.strip():
                return txt
        except Exception as e:
            app.log(f"  {url}: {type(e).__name__}: {str(e)[:100]}")
        time.sleep(2 * (k + 1))
    return None


def _http_first(urls: List[str]) -> Optional[str]:
    for i, u in enumerate(urls):
        txt = _http_text(u, tries=3 if i == 0 else 1)
        if txt:
            return txt
    return None


def is_common_stock(r: Dict[str, Any]) -> bool:
    return not _BAD_NAME.search(r.get("name", ""))


def select_universe(rows: List[Dict[str, Any]], exclude: Optional[set] = None) -> List[Dict[str, Any]]:
    """Xom qatorlardan: sinov/varrant/imtiyozli aksiya/ETF (roʻyxatdagilardan tashqari) olib tashlanadi; Yahoo shaklidagi tiker, takrorlarsiz."""
    exclude = exclude or set()
    etf_ok = set(CURATED_ETFS)
    seen: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        t = sym_to_yahoo(r.get("sym", ""))
        if not t or t in exclude or r.get("test"):
            continue
        if r.get("etf"):                                   # ETF: faqat tanlangan yirik ETFlar, har qanday birjadan (SPY — NYSE Arca)
            if t not in etf_ok:
                continue
        elif r.get("ex") not in EX_OK or not is_common_stock(r):
            continue
        if t in seen:
            continue
        flag = None
        if r.get("st") in ("D", "E", "Q", "G", "H", "J", "K"):
            flag = {"D": "Nasdaq: listing talablariga javob bermaydi (deficient)", "E": "Nasdaq: hisobotlarni oʻz vaqtida topshirmagan (delinquent)",
                    "Q": "Nasdaq: bankrotlik jarayonida"}.get(r["st"], "Nasdaq: moliyaviy holati boʻyicha ogohlantirish belgisi bor")
        seen[t] = dict(t=t, name=clean_name(r.get("name", "")), ex=r["ex"], etf=bool(r.get("etf")), flag=flag)
    return list(seen.values())


def load_symbol_universe(cache_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str]:
    """(xom qatorlar, manba tavsifi). Tartib: nasdaqtrader (https, ftp) -> SEC -> oxirgi saqlangan nusxa."""
    rows: List[Dict[str, Any]] = []
    src: List[str] = []
    t1 = _http_first(SYMDIR_URLS["nasdaq"])
    t2 = _http_first(SYMDIR_URLS["other"])
    if t1:
        rows += parse_nasdaqlisted(t1)
        src.append("nasdaqlisted")
    if t2:
        rows += parse_otherlisted(t2)
        src.append("otherlisted")
    if len(rows) < 5000:                                   # ikkala fayl kutilganda ~10 000+ qator boʻladi
        txt = _http_text(SEC_TICKERS_URL, tries=2)
        if txt:
            try:
                have = {r["sym"] for r in rows}
                add = [r for r in parse_sec_json(json.loads(txt)) if r["sym"] not in have]
                rows += add
                if add:
                    src.append("SEC")
            except Exception as e:
                app.log(f"  SEC roʻyxati oʻqilmadi: {type(e).__name__}")
    if cache_path:
        try:
            if len(rows) >= 1500:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(dict(saved=dt.date.today().isoformat(), rows=rows), f)
            elif os.path.exists(cache_path):
                obj = json.load(open(cache_path, encoding="utf-8"))
                age = (dt.date.today() - dt.date.fromisoformat(obj["saved"])).days
                if age <= 45 and len(obj["rows"]) > len(rows):
                    rows, src = obj["rows"], [f"saqlangan nusxa ({age} kun oldingi)"]
        except Exception as e:
            app.log(f"  roʻyxat keshi: {type(e).__name__}: {str(e)[:80]}")
    return rows, "+".join(src) or "yoʻq"


# ----- narx tarixini yuklash (parallel, rate-limit uchun umumiy "toʻxtash darvozasi", qayta urinish, Stooq/kesh zaxirasi)
_RATE_RE = re.compile(r"too many|rate.?limit|\b429\b|YFRateLimit", re.I)
_GONE_RE = re.compile(r"delisted|no data found|no price data|not found|\b404\b|symbol may be|invalid", re.I)


def _kind(err: str) -> str:
    return "rate" if _RATE_RE.search(err or "") else "gone" if _GONE_RE.search(err or "") else "other"


class _Gate:
    """Bitta oqim Yahoo cheklovini koʻrsa — hamma oqimlar birga kutadi (cheklovni kuchaytirmaymiz)."""

    def __init__(self) -> None:
        self.until = 0.0
        self.hits = 0
        self.lock = threading.Lock()

    def wait(self) -> None:
        d = self.until - time.time()
        if d > 0:
            time.sleep(min(d, 180.0))

    def trip(self, secs: float) -> None:
        with self.lock:
            self.until = max(self.until, time.time() + secs)
            self.hits += 1


def _yf_fetch(t: str, start: str) -> Tuple[Optional[pd.DataFrame], str]:
    import yfinance as yf
    try:
        df = yf.Ticker(t).history(start=start, interval="1d", auto_adjust=True)
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:150]}"
    d = app._clean_ohlcv(df)
    if d is None:
        err = ""
        try:
            err = str(getattr(yf.shared, "_ERRORS", {}).get(t, ""))
        except Exception:
            pass
        return None, err or "boʻsh javob"
    return app._drop_partial_today(d), ""


def _stooq_fetch(t: str, start: str) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        d = app._stooq_one(t, LITE_YEARS)
        return (app._drop_partial_today(d), "") if d is not None else (None, "Stooq: boʻsh")
    except Exception as e:
        return None, f"Stooq {type(e).__name__}: {str(e)[:100]}"


def download_lite(tickers: List[str], years: float = LITE_YEARS, budget_s: float = 4200.0, workers: int = 8,
                  fetcher: Optional[Callable[[str, str], Tuple[Optional[pd.DataFrame], str]]] = None, passes: int = 3,
                  log_every: int = 500) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Narx tarixini parallel yuklaydi. Qaytaradi: (maʼlumot, muvaffaqiyatsizlar {tiker: sabab-turi}).
    Yahoo cheklasa umumiy pauza (10→20→40 s), keyin qolganlar sekinroq qayta uriniladi; vaqt tugasa — qolganlari 'time'."""
    fetcher = fetcher or _yf_fetch
    start = app._start_date(years)
    data: Dict[str, pd.DataFrame] = {}
    status: Dict[str, str] = {}
    deadline = time.time() + budget_s
    gate = _Gate()
    st = {"consec": 0, "ok": 0, "abort": False}
    lock = threading.Lock()
    import logging
    yl = logging.getLogger("yfinance")
    old_lvl = yl.level
    yl.setLevel(logging.CRITICAL)                          # minglab "possibly delisted" xabarlarini yashiramiz

    def job(t: str, tries: int) -> Tuple[str, Optional[pd.DataFrame], str]:
        err = ""
        try:
            for attempt in range(tries):
                if time.time() > deadline or st["abort"]:
                    return t, None, "time"
                gate.wait()
                d, err = fetcher(t, start)
                if d is not None:
                    with lock:
                        st["consec"] = 0
                        st["ok"] += 1
                    return t, d, "ok"
                k = _kind(err)
                if k == "gone":
                    return t, None, "gone"
                if k == "rate":
                    gate.trip(10 * (2 ** attempt))
                    continue
                time.sleep(0.4 * (attempt + 1))
        except Exception as e:                                # kutilmagan xato bitta tikerni oʻtkazib yuboradi, butun yuklashni emas
            err = f"{type(e).__name__}: {str(e)[:100]}"
        k = _kind(err)
        with lock:
            st["consec"] += 1
            if st["consec"] >= 60 and st["ok"] == 0:       # birinchi 60 ta urinishdan birortasi ham kelmadi — Yahoo butunlay yopiq
                st["abort"] = True
        return t, None, k

    pending = list(tickers)
    t0 = time.time()
    try:
        for ps in range(1, passes + 1):
            if not pending or time.time() > deadline or st["abort"]:
                break
            nw = workers if ps == 1 else max(2, workers // 3)
            if ps > 1:
                time.sleep(min(30.0, max(0.0, deadline - time.time())))
            done = 0
            nxt: List[str] = []
            with ThreadPoolExecutor(max_workers=nw) as ex:
                for t, d, k in ex.map(lambda x: job(x, 3 if ps == 1 else 2), pending):
                    done += 1
                    if d is not None:
                        data[t] = d
                        status.pop(t, None)
                    else:
                        status[t] = k
                        if k in ("rate", "other", "time"):
                            nxt.append(t)
                    if done % log_every == 0:
                        app.log(f"  yuklash (oʻtish {ps}): {done}/{len(pending)}, tayyor {len(data)} ta, {time.time() - t0:.0f} s, Yahoo pauzalari: {gate.hits}")
            pending = nxt
            app.log(f"  yuklash: oʻtish {ps} tugadi — tayyor {len(data)}, qayta uriniladigan {len(pending)}, 'delisted/yoʻq' {sum(1 for v in status.values() if v == 'gone')}")
    finally:
        yl.setLevel(old_lvl)
    return data, status


def stooq_rescue(missing: List[str], data: Dict[str, pd.DataFrame], status: Dict[str, str], budget_s: float = 300.0, workers: int = 4) -> int:
    """Yahoo bermagan tikerlarni Stooq'dan olishga urinadi (kunlik limit boʻlsa, ketma-ket xatodan keyin toʻxtaydi)."""
    if not missing or budget_s <= 0:
        return 0
    deadline = time.time() + budget_s
    st = {"fail": 0}
    lock = threading.Lock()
    got = 0

    def job(t: str) -> Tuple[str, Optional[pd.DataFrame]]:
        if time.time() > deadline or st["fail"] >= 15:
            return t, None
        d, _ = _stooq_fetch(t, "")
        with lock:
            st["fail"] = 0 if d is not None else st["fail"] + 1
        return t, d

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for t, d in ex.map(job, missing):
            if d is not None and len(d) >= app.LITE_MIN_BARS:
                data[t] = d
                status.pop(t, None)
                got += 1
    return got


def _price_cache_path() -> str:
    return os.path.join(app.CACHE_DIR, "lite_prices.pkl")


def load_price_cache(max_age_days: int = 7) -> Dict[str, pd.DataFrame]:
    """Oldingi muvaffaqiyatli yuklash nusxasi (GitHub Actions keshi orqali): yangi yuklanmagan tikerlar uchun zaxira."""
    p = _price_cache_path()
    try:
        if os.path.exists(p):
            import pickle
            with open(p, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and (dt.date.today() - dt.date.fromisoformat(obj.get("saved", "2000-01-01"))).days <= max_age_days:
                return {t: d.astype("float64") for t, d in obj.get("data", {}).items()}
    except Exception as e:
        app.log(f"  narx keshi oʻqilmadi: {type(e).__name__}")
    return {}


def save_price_cache(data: Dict[str, pd.DataFrame]) -> None:
    try:
        import pickle
        os.makedirs(app.CACHE_DIR, exist_ok=True)
        small = {t: d.astype("float32") for t, d in data.items()}
        with open(_price_cache_path(), "wb") as f:
            pickle.dump(dict(saved=dt.date.today().isoformat(), data=small), f, protocol=4)
    except Exception as e:
        app.log(f"  narx keshi saqlanmadi: {type(e).__name__}")


# ======================================================================================
# 3. HTML QISMLARI
# ======================================================================================
PAGE_CSS = """<style>
:root{color-scheme:light}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:#fff;color:#0b0b0b;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.top{position:sticky;top:0;z-index:20;background:#fff;border-bottom:1px solid #d9d8d0}
.top .in{max-width:1080px;margin:0 auto;padding:8px 14px;display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center}
.brand{font-weight:700;font-size:1.08rem;text-decoration:none;color:#0b0b0b}
.nav{display:flex;gap:2px;flex-wrap:wrap}
.nav a{padding:5px 10px;border-radius:8px;text-decoration:none;color:#52514e;font-size:.9rem;white-space:nowrap}
.nav a.on{background:#eaf1fb;color:#1a5fb4;font-weight:600}
.srch{display:flex;gap:6px;margin:0 0 0 auto}
.srch input{padding:7px 10px;border:1px solid #c9c8c0;border-radius:8px;font-size:16px;width:170px;min-width:0}
.btn,.srch button{padding:7px 13px;border:0;border-radius:8px;background:#2a78d6;color:#fff;font-size:.9rem;cursor:pointer;text-decoration:none;display:inline-block}
.btn.alt{background:#eef0f3;color:#1c1c1c}
#srchmsg{max-width:1080px;margin:0 auto;padding:0 14px}
#srchmsg .card{margin:8px 0;background:#fff7e6;border:1px solid #f0d28a;border-left:5px solid #fab219;border-radius:10px;padding:10px 14px;font-size:.9rem;line-height:1.5}
.wrapp{max-width:1080px;margin:0 auto;padding:6px 14px 30px}
.tools{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}
footer{max-width:1080px;margin:0 auto;padding:6px 14px 34px;font-size:.78rem;color:#6b6a66;line-height:1.5}
table.sortable th.s,table.scr th.s{cursor:pointer;user-select:none}
table.sortable th.s:after,table.scr th.s:after{content:" ↕";opacity:.35}
table.sortable th.s.asc:after,table.scr th.s.asc:after{content:" ▲";opacity:1}
table.sortable th.s.desc:after,table.scr th.s.desc:after{content:" ▼";opacity:1}
.flt{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
.flt input,.flt select{padding:8px 10px;border:1px solid #c9c8c0;border-radius:8px;font-size:16px;background:#fff;min-width:0}
.flt input{flex:1;min-width:150px}
.calcrow{display:flex;gap:8px;margin:8px 0}
.calcrow input{padding:9px 11px;border:1px solid #c9c8c0;border-radius:8px;font-size:16px;flex:1;min-width:0;max-width:260px}
.ai a.tl{font-weight:700;text-decoration:none}
table.scr .nm{white-space:normal}
@media (max-width:640px){table.scr .nm{max-width:86px;line-height:1.25}table.scr th{white-space:normal}.ai table.scr td,.ai table.scr th{padding:6px 3px}.ai table.scr .tag{padding:0 6px;font-size:.72rem}.ai table.scr td.n{font-size:.76rem}.top{position:static}.srch{margin:0;width:100%}.srch input{flex:1;width:auto}.nav a{padding:5px 8px;font-size:.84rem}.wrapp{padding:4px 8px 26px}}
@media print{.top,footer,.noprint,#srchmsg{display:none!important}}
</style>"""

SITE_JS = r"""(function(){
var A=window.AKSIYA||{repo:"",n:{}};
function norm(s){return (s||'').toUpperCase().replace(/\s+/g,'').replace(/\./g,'-').replace(/[^A-Z0-9\-\^=]/g,'');}
function base(){var s=document.currentScript&&document.currentScript.src||'';return s.replace(/site\.js(\?.*)?$/,'');}
function esc(s){return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
var B=base(),T=null,tLoading=false,tQ=[];
function buildT(){T={};(window.AKSIYA_T||[]).forEach(function(r){T[r[0]]=r;});}
function loadT(cb){
  if(T){cb(true);return;}
  if(window.AKSIYA_T){buildT();cb(true);return;}
  tQ.push(cb);if(tLoading){return;}tLoading=true;
  var s=document.createElement('script');s.src=B+'tickers.js';
  s.onload=function(){buildT();var q=tQ;tQ=[];q.forEach(function(f){f(true);});};
  s.onerror=function(){var q=tQ;tQ=[];tLoading=false;q.forEach(function(f){f(false);});};
  document.head.appendChild(s);
}
function notFound(Tk){
  var n=(A.n&&A.n.all)||0,wf=A.repo?('https://github.com/'+A.repo+'/actions/workflows/build-site.yml'):'';
  return '<div class="card"><b>'+esc(Tk)+'</b> tayyor tahlillar roʻyxatida yoʻq. Sayt NYSE, NASDAQ va NYSE American dagi '+(n?n.toLocaleString('en-US')+' ta ':'')+'oddiy aksiyani va yirik ETFlarni qamraydi. Mumkin sabablar: '
   +'(1) yozilishi — sinf aksiyalari tire bilan (<b>BRK-B</b>, <b>BF-B</b>); (2) tiker bu birjalarda emas (OTC, chet el birjasi) yoki aksiya emas (varrant, imtiyozli aksiya); '
   +'(3) tarixi 120 savdo kunidan kam (yangi IPO); (4) Yahoo bugun narx bermagan (keyingi kunlik qurilishda qayta uriniladi).<br>'
   +'<b>Majburan qoʻshish:</b> GitHub → Actions → «Saytni qurish va joylash» → <i>Run workflow</i> → tikerlar maydoniga <b>'+esc(Tk)+'</b> yozing'
   +(wf?' — <a href="'+wf+'" target="_blank" rel="noopener">shu yerda</a>':'')
   +'. Narx manbasida tiker topilsa, u toʻliq tahlil bilan (AI ball, fundamental, yangilik) saytga qoʻshiladi va keyingi kunlarda ham yangilanib turadi.</div>';
}
window.goT=function(f){
  var Tk=norm(f.q.value),box=document.getElementById('srchmsg');
  if(!Tk){return false;}
  loadT(function(ok){
    if(!ok||T[Tk]){location.href=B+'t/'+Tk+'.html';return;}   // roʻyxat yuklanmasa — toʻgʻridan-toʻgʻri urinamiz (yoʻq boʻlsa 404 sahifasi tushuntiradi)
    if(box){box.innerHTML=notFound(Tk);}
  });
  return false;
};
(function(){
  var inp=document.querySelector('.srch input'),dl=document.getElementById('tl');
  function suggest(){
    if(!T||!dl||!inp){return;}
    var low=(inp.value||'').toLowerCase().trim(),v=norm(inp.value),L=window.AKSIYA_T||[],a=[],b=[],i,r;
    if(!low){dl.innerHTML='';return;}
    for(i=0;i<L.length&&a.length<10;i++){r=L[i];if(v&&r[0].indexOf(v)===0){a.push(r);}}
    if(low.length>=2){for(i=0;i<L.length&&a.length+b.length<10;i++){r=L[i];if(!(v&&r[0].indexOf(v)===0)&&String(r[1]||'').toLowerCase().indexOf(low)>=0){b.push(r);}}}
    dl.innerHTML=a.concat(b).map(function(r){return '<option value="'+esc(r[0])+'">'+esc(r[1])+'</option>';}).join('');
  }
  if(inp){inp.addEventListener('focus',function(){loadT(function(){});});inp.addEventListener('input',function(){loadT(suggest);});}
  if(document.body&&document.body.getAttribute('data-nf')){            // 404 sahifasi: manzildagi tikerni tushuntiramiz
    var m=location.pathname.match(/\/t\/([A-Za-z0-9\-]+)\.html$/),box=document.getElementById('srchmsg');
    if(m&&box){loadT(function(ok){if(ok){box.innerHTML=notFound(m[1].toUpperCase());}});}
  }
})();
// jadval: saralash va filtr
document.querySelectorAll('table.sortable').forEach(function(tb){
  var ths=tb.querySelectorAll('th.s'),body=tb.tBodies[0];
  ths.forEach(function(th){
    th.addEventListener('click',function(){
      var k=parseInt(th.getAttribute('data-k')),asc=!th.classList.contains('asc');
      ths.forEach(function(x){x.classList.remove('asc','desc');});
      th.classList.add(asc?'asc':'desc');
      var rows=Array.prototype.slice.call(body.rows);
      rows.sort(function(a,b){
        var x=a.cells[k].getAttribute('data-v'),y=b.cells[k].getAttribute('data-v');
        var nx=parseFloat(x),ny=parseFloat(y),bad=function(v){return isNaN(v);};
        if(bad(nx)&&bad(ny)){return 0;} if(bad(nx)){return 1;} if(bad(ny)){return -1;}
        return asc?nx-ny:ny-nx;});
      rows.forEach(function(r){body.appendChild(r);});
    });
  });
});
var q=document.getElementById('flt-q'),lb=document.getElementById('flt-l');
function filt(){
  var s=(q&&q.value||'').toLowerCase().trim(),l=lb?lb.value:'',n=0;
  document.querySelectorAll('table.sortable tbody tr').forEach(function(r){
    var ok=(!s||(r.getAttribute('data-s')||'').indexOf(s)>=0)&&(!l||r.getAttribute('data-l')===l);
    r.style.display=ok?'':'none'; if(ok){n++;}});
  var c=document.getElementById('flt-n'); if(c){c.textContent=n+' ta';}
}
if(q){q.addEventListener('input',filt);} if(lb){lb.addEventListener('change',filt);}
// maqsad narx kalkulyatori (faqat aksiya sahifasida)
var G=window.AKSIYA_G,inp=document.getElementById('tgt'),out=document.getElementById('tgtout');
if(G&&inp&&out){
  var look=function(a,x){var q=Math.abs(x)/G.step,i=Math.floor(q);if(i>=a.length-1){return a[a.length-1]/10000;}var f=q-i;return (a[i]*(1-f)+a[i+1]*f)/10000;};
  var fp=function(p){if(p<0.0005){return '<0.1%';}if(p>0.9995){return '>99.9%';}return (p*100).toFixed(1)+'%';};
  var parse=function(s){s=(s||'').replace(/[$\s]/g,'');if(!s){return NaN;}
    if(s.indexOf(',')>=0&&s.indexOf('.')>=0){s=s.replace(/,/g,'');}else{s=s.replace(',','.');}
    var v=parseFloat(s);return (isFinite(v)&&v>0)?v:NaN;};
  var calc=function(){
    var T=parse(inp.value);
    if(isNaN(T)){out.innerHTML=inp.value.trim()?'<p class="small muted">Maqsad narx tushunarsiz yozilgan (masalan: 250 yoki 250.5).</p>':'';return;}
    var x=Math.log(T/G.P0),up=x>=0,pct=(T/G.P0-1)*100;
    if(x>G.step*(G.tu[0].length-1)||-x>G.step*(G.td[0].length-1)){
      out.innerHTML='<p class="small muted">Maqsad narx hozirgi narxdan juda uzoq ('+(pct>=0?'+':'')+pct.toFixed(0)+'%): hisoblash jadvali shu oraliqdan tashqarida; ehtimol juda kichik.</p>';return;}
    var rows='';
    for(var j=0;j<G.hs.length;j++){
      rows+='<tr><td>'+G.lab[j]+'</td><td class="n"><b>'+fp(look(up?G.tu[j]:G.td[j],x))+'</b></td><td class="n">'+fp(look(up?G.eu[j]:G.ed[j],x))+'</td></tr>';}
    out.innerHTML='<h3>Maqsad narx: $'+T.toFixed(2)+' ('+(pct>=0?'+':'')+pct.toFixed(1)+'%)</h3><div class="wrap"><table><thead><tr><th>Ufq</th>'
      +'<th class="n">Davr ichida kamida bir marta yetish ehtimoli</th><th class="n">Ufq oxirida shu darajada yoki undan '+(up?'yuqorida':'pastda')+' boʻlish ehtimoli</th></tr></thead><tbody>'+rows+'</tbody></table></div>'
      +'<p class="small muted">Monte-Karlo (20 000 yoʻl), AI emas. Bu — tarixiy volatillikdan hisoblangan statistik ehtimol, narx shunday boʻladi degan bashorat emas.</p>';
  };
  inp.addEventListener('input',calc);
  var btn=document.getElementById('tgtgo'); if(btn){btn.addEventListener('click',calc);}
  inp.addEventListener('keydown',function(e){if(e.key==='Enter'){e.preventDefault();calc();}});
}
// ilova rejimi (PWA): service worker (oflayn + oʻrnatish) va «Oʻrnatish» tugmasi
if('serviceWorker' in navigator&&(location.protocol==='https:'||location.hostname==='localhost'||location.hostname==='127.0.0.1')){
  window.addEventListener('load',function(){navigator.serviceWorker.register(B+'sw.js').catch(function(){});});
}
(function(){
  var dp=null,btn=document.getElementById('pwa-install'),st=document.getElementById('pwa-state'),hint=document.getElementById('pwa-hint');
  window.addEventListener('beforeinstallprompt',function(e){if(!btn){return;}e.preventDefault();dp=e;btn.style.display='';if(hint){hint.textContent='';}});   // boshqa sahifalarda brauzerning oʻz taklifi (mini-banner) ishlayveradi
  window.addEventListener('appinstalled',function(){dp=null;if(btn){btn.style.display='none';}if(hint){hint.textContent='✅ Oʻrnatildi! Endi ilovani bosh ekran yoki ilovalar roʻyxatidan oching.';}});
  if(!btn){return;}
  btn.addEventListener('click',function(){if(!dp){return;}dp.prompt();if(dp.userChoice){dp.userChoice.then(function(){dp=null;btn.style.display='none';});}});
  var ua=navigator.userAgent||'',ios=/iPad|iPhone|iPod/.test(ua)||(navigator.platform==='MacIntel'&&navigator.maxTouchPoints>1),
      plat=ios?'ios':(/Android/i.test(ua)?'android':'desktop'),
      sa=(window.matchMedia&&window.matchMedia('(display-mode: standalone)').matches)||navigator.standalone===true;
  if(sa&&st){st.style.display='';}
  var d=document.getElementById('pwa-'+plat);if(d){d.open=true;}
  if(hint&&!sa&&!dp){hint.textContent=(plat==='ios')?'iPhone’da tugma yoʻq — Safari’da Ulashish menyusidan qoʻshiladi (pastda).':'Tugma koʻrinmasa — pastdagi qadamlardan foydalaning.';}
})();
})();
"""


def search_form() -> str:
    return ('<form class="srch noprint" onsubmit="return goT(this)" autocomplete="off">'
            '<input name="q" placeholder="Tiker: AAPL, NVDA, AMD…" list="tl" autocapitalize="characters" '
            'autocorrect="off" spellcheck="false" aria-label="Tiker"><datalist id="tl"></datalist><button type="submit">Tahlil</button></form>')


def page(title: str, body: str, active: str, depth: int = 0, extra_js: str = "", css_link: bool = False, head_extra: str = "",
         body_attr: str = "", tail_js: str = "") -> str:
    p = "../" * depth
    tabs = [("index", "🏆 Tavsiyalar"), ("screener", "🧭 Skaner"), ("stocks", "🔍 AI roʻyxati"), ("quality", "📊 Sifat"), ("method", "ℹ️ Usul"),
            ("install", "📲 Ilova")]
    nav = "".join(f'<a href="{p}{k}.html"{" class=on" if k == active else ""}>{lab}</a>' for k, lab in tabs)
    foot = (f'<footer>Aksiya AI v{app.APP_VERSION} · sayt qurilgan: {META["built_local"]} (Toshkent vaqti) · narx maʼlumoti sanasi: '
            f'{META["asof"]} · manba: Yahoo Finance. Bu — taʼlim va tadqiqot vositasi, moliyaviy maslahat EMAS. '
            f'Sahifa sayt qurilgandagi holatni koʻrsatadi (real vaqtda yangilanmaydi).</footer>')
    css = f'<link rel="stylesheet" href="{p}style.css">' if css_link else f"{CSS}{PAGE_CSS}"
    tail = f"<script>{tail_js}</script>" if tail_js else ""
    return ('<!doctype html><html lang="uz"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta name="color-scheme" content="light"><meta name="robots" content="noindex,nofollow">'
            f'{head_extra}<title>{esc(title)}</title>{css}{pwa_head(p)}</head><body{(" " + body_attr) if body_attr else ""}>'
            f'<header class="top"><div class="in"><a class="brand" href="{p}index.html">📈 Aksiya AI</a>'
            f'<nav class="nav">{nav}</nav>{search_form()}</div><div id="srchmsg"></div></header>'
            f'{body}{foot}<script src="{p}data.js"></script>{extra_js}<script src="{p}site.js"></script>{tail}</body></html>')


# ======================================================================================
# 3b. ILOVA REJIMI (PWA): telefon va kompyuterga ilova qilib oʻrnatish
#     manifest.json + belgilar (icons/) + sw.js (service worker) + «📲 Ilova» sahifasi. Hammasi statik fayllar — server kerak emas.
# ======================================================================================
PWA_NAME = "Aksiya AI"
PWA_COLOR = "#ffffff"                 # sayt sarlavhasi oq — ilova oynasi/holat satri ham oq


def pwa_head(p: str) -> str:
    """Har bir sahifaning <head> qismiga: manifest, belgi (brauzer yorligʻi va iOS), mavzu rangi."""
    return (f'<link rel="manifest" href="{p}manifest.json"><meta name="theme-color" content="{PWA_COLOR}">'
            f'<link rel="icon" type="image/png" sizes="192x192" href="{p}icons/icon-192.png">'
            f'<link rel="apple-touch-icon" href="{p}icons/apple-touch-icon.png">'
            '<meta name="mobile-web-app-capable" content="yes"><meta name="apple-mobile-web-app-capable" content="yes">'
            f'<meta name="apple-mobile-web-app-title" content="{PWA_NAME}"><meta name="apple-mobile-web-app-status-bar-style" content="default">')


def _icon_png(size: int, rounded: bool, alpha: bool = True):
    """Belgi: koʻk gradient fon + oʻsayotgan 4 ta sham (birinchisi tushgan, qizil). Pillow bilan 4x katta chizilib, silliqlab kichraytiriladi.
    rounded=True — burchaklari yumaloq (oddiy belgi); False — toʻliq fon (maskable va iOS belgisi: tizim oʻzi qirqadi).
    Chizma markaziy doira (radiusi 0,4) ichida turadi, shuning uchun maskable qirqishda kesilmaydi."""
    from PIL import Image, ImageDraw
    S = 4
    N = size * S
    yy = np.linspace(0.0, 1.0, N, dtype=np.float32)[:, None, None]
    top = np.array([36, 121, 220], dtype=np.float32)[None, None, :]
    bot = np.array([8, 34, 74], dtype=np.float32)[None, None, :]
    arr = (top * (1.0 - yy) + bot * yy) * np.ones((1, N, 1), dtype=np.float32)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    d = ImageDraw.Draw(img)

    def px(u: float) -> float:
        return u * N

    def price_y(v: float) -> float:                       # narx (0..1) -> y (tepa = 0)
        return px(0.765 - 0.52 * v)
    xs = [0.268, 0.423, 0.578, 0.733]
    # (past, yuqori, ochilish, yopilish): chapdan oʻngga oʻsish
    candles = [(0.08, 0.44, 0.36, 0.16), (0.14, 0.54, 0.22, 0.46), (0.32, 0.74, 0.40, 0.66), (0.50, 0.99, 0.56, 0.92)]
    w = 0.108
    for x, (lo, hi, op, cl) in zip(xs, candles):
        col = (52, 211, 120, 255) if cl >= op else (255, 96, 96, 255)
        d.line([(px(x), price_y(lo)), (px(x), price_y(hi))], fill=col, width=max(2, int(0.022 * N)))
        y1, y2 = sorted((price_y(op), price_y(cl)))
        d.rounded_rectangle([px(x - w / 2), y1, px(x + w / 2), y2], radius=0.018 * N, fill=col,
                            outline=(255, 255, 255, 255), width=max(2, int(0.009 * N)))
    if rounded:
        m = Image.new("L", (N, N), 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, N - 1, N - 1], radius=int(0.2237 * N), fill=255)
        img.putalpha(m)
    img = img.resize((size, size), Image.LANCZOS)
    return img if alpha else img.convert("RGB")


_ICON_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
             '<stop offset="0" stop-color="#2479dc"/><stop offset="1" stop-color="#08224a"/></linearGradient></defs>'
             '<rect width="512" height="512" rx="114" fill="url(#g)"/>'
             '<g stroke="#fff" stroke-width="5" stroke-linejoin="round"><rect x="110" y="291" width="55" height="62" rx="9" fill="#ff6060"/>'
             '<rect x="189" y="231" width="55" height="100" rx="9" fill="#34d378"/><rect x="268" y="178" width="55" height="130" rx="9" fill="#34d378"/>'
             '<rect x="347" y="116" width="55" height="172" rx="9" fill="#34d378"/></g></svg>')


def make_icons(icon_dir: str) -> List[Dict[str, str]]:
    """Belgilarni yozadi va manifest uchun roʻyxat qaytaradi. Pillow boʻlmasa/xato boʻlsa — SVG zaxira (sayt baribir qurilaveradi)."""
    os.makedirs(icon_dir, exist_ok=True)
    try:
        _icon_png(192, True).save(os.path.join(icon_dir, "icon-192.png"), optimize=True)
        _icon_png(512, True).save(os.path.join(icon_dir, "icon-512.png"), optimize=True)
        _icon_png(512, False, alpha=False).save(os.path.join(icon_dir, "maskable-512.png"), optimize=True)
        _icon_png(180, False, alpha=False).save(os.path.join(icon_dir, "apple-touch-icon.png"), optimize=True)
        return [{"src": "icons/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
                {"src": "icons/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any"},
                {"src": "icons/maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"}]
    except Exception as e:                                                      # noqa: BLE001
        app.log(f"  ilova belgilari (PNG) yaratilmadi ({type(e).__name__}: {str(e)[:80]}) — SVG zaxira ishlatildi")
        with open(os.path.join(icon_dir, "icon.svg"), "w", encoding="utf-8") as f:
            f.write(_ICON_SVG)
        return [{"src": "icons/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any"}]


def pwa_manifest(icons: List[Dict[str, str]]) -> Dict[str, Any]:
    png = next((i["src"] for i in icons if i["src"].endswith("192.png")), icons[0]["src"])
    sc = [{"name": "Skaner — barcha aksiyalar", "short_name": "Skaner", "url": "./screener.html", "icons": [{"src": png, "sizes": icons[0]["sizes"]}]},
          {"name": "Tavsiyalar", "short_name": "Tavsiyalar", "url": "./index.html", "icons": [{"src": png, "sizes": icons[0]["sizes"]}]}]
    return {"name": "Aksiya AI — AQSh aksiyalari tahlili", "short_name": PWA_NAME, "lang": "uz", "dir": "ltr",
            "description": "NYSE va NASDAQ aksiyalari: texnik tahlil, zonalar, trend chiziqlari, Fibonachchi, Elliott, Monte-Karlo va qisqa xulosa.",
            "start_url": "./index.html", "scope": "./", "display": "standalone", "background_color": PWA_COLOR, "theme_color": PWA_COLOR,
            "categories": ["finance", "education"], "icons": icons, "shortcuts": sc}


SW_JS = r"""/* Aksiya AI service worker — qurilish __STAMP__.
   Tarmoq birinchi: internet bor paytda har doim eng yangi sahifa; internet yoʻq boʻlsa — oxirgi koʻrilgan sahifalar. */
var SHELL='aksiya-shell-__STAMP__',PAGES='aksiya-pages',MAXP=250;
var CORE=['./','index.html','screener.html','style.css','data.js','site.js','manifest.json','icons/icon-192.png'];
self.addEventListener('install',function(e){
  e.waitUntil(caches.open(SHELL).then(function(c){
    return Promise.all(CORE.map(function(u){return c.add(u).catch(function(){});}));
  }).then(function(){return self.skipWaiting();}));
});
self.addEventListener('activate',function(e){
  e.waitUntil(caches.keys().then(function(ks){
    return Promise.all(ks.filter(function(k){return k.indexOf('aksiya-shell-')===0&&k!==SHELL;}).map(function(k){return caches.delete(k);}));
  }).then(function(){return self.clients.claim();}));
});
function offline(){
  var h='<!doctype html><html lang="uz"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Internet yoʻq</title></head>'
   +'<body style="font-family:system-ui,sans-serif;max-width:560px;margin:12vh auto;padding:0 18px;line-height:1.5"><h2>📡 Internet yoʻq</h2>'
   +'<p>Bu sahifa hali ochilmagan, shuning uchun qurilmada saqlanmagan. Internetga ulangach qayta urinib koʻring. Oldin koʻrgan sahifalaringiz internetsiz ham ochiladi.</p>'
   +'<p><a href="./index.html">← Bosh sahifa</a></p></body></html>';
  return new Response(h,{status:503,headers:{'Content-Type':'text/html; charset=utf-8'}});
}
self.addEventListener('fetch',function(e){
  var r=e.request;
  if(r.method!=='GET'){return;}
  var u=new URL(r.url);
  if(u.origin!==self.location.origin){return;}
  e.respondWith(fetch(r).then(function(resp){
    if(resp&&resp.status===200&&resp.type==='basic'){
      var cp=resp.clone();
      e.waitUntil(caches.open(PAGES).then(function(c){
        return c.put(r,cp).then(function(){return c.keys();}).then(function(ks){if(ks.length>MAXP){return c.delete(ks[0]);}});
      }).catch(function(){}));
    }
    return resp;
  }).catch(function(){
    return caches.match(r,{ignoreSearch:true}).then(function(m){return m||(r.mode==='navigate'?offline():Response.error());});
  }));
});
"""


def install_page() -> str:
    body = (
        '<div class="wrapp"><div class="ai"><h2>📲 Ilova sifatida oʻrnatish</h2>'
        '<p>Bu saytni telefon yoki kompyuteringizga <b>ilova</b> qilib oʻrnatish mumkin: oʻz belgisi (ikonka) boʻladi, brauzer satrisiz alohida oynada ochiladi, '
        'internet boʻlmasa oxirgi koʻrilgan sahifalar ochiladi. Doʻkondan yuklash va toʻlov kerak emas — ilova shu saytning oʻzi, shuning uchun har kuni '
        'yangilanadi (alohida yangilash shart emas).</p>'
        '<div class="card good" id="pwa-state" style="display:none"><b>✅ Siz hozir ilova rejimida ishlayapsiz.</b></div>'
        '<p><button class="btn" id="pwa-install" type="button" style="display:none">📲 Hozir oʻrnatish</button> <span class="small muted" id="pwa-hint"></span></p>'
        '<details id="pwa-android"><summary><b>📱 Android telefon (Chrome)</b></summary><ol>'
        '<li>Saytni <b>Chrome</b> da oching.</li><li>Yuqori oʻng burchakdagi <b>⋮</b> (uch nuqta) menyusini bosing.</li>'
        '<li><b>«Ilovani oʻrnatish»</b> (yoki «Bosh ekranga qoʻshish») ni tanlang va <b>«Oʻrnatish»</b> ni tasdiqlang.</li>'
        '<li>Belgi bosh ekranda va ilovalar roʻyxatida paydo boʻladi. Ochilganda brauzer satri boʻlmaydi.</li></ol></details>'
        '<details id="pwa-ios"><summary><b>📱 iPhone / iPad (Safari)</b></summary><ol>'
        '<li>Saytni <b>Safari</b> da oching (iOS 16.4 dan boshlab Chrome/Edge ham mumkin).</li>'
        '<li>Pastdagi (iPad’da tepadagi) <b>Ulashish</b> tugmasini (□ va ↑ belgisi) bosing.</li>'
        '<li>Roʻyxatni pastga suring va <b>«Bosh ekranga qoʻshish»</b> ni tanlang, keyin <b>«Qoʻshish»</b> ni bosing.</li>'
        '<li>Belgi bosh ekranda paydo boʻladi.</li></ol><p class="small muted">iPhone’da bu sahifada «Oʻrnatish» tugmasi chiqmaydi — Apple faqat Ulashish menyusi orqali qoʻshishga ruxsat beradi.</p></details>'
        '<details id="pwa-desktop"><summary><b>💻 Kompyuter (Chrome, Edge)</b></summary><ol>'
        '<li>Saytni <b>Chrome</b> yoki <b>Edge</b> da oching.</li>'
        '<li>Manzil satrining <b>oʻng chetida</b> «Oʻrnatish» belgisi (monitor va pastga oʻq) chiqadi — uni bosing va <b>«Oʻrnatish»</b> ni tasdiqlang.</li>'
        '<li>Belgi koʻrinmasa: Chrome — <b>⋮ → «Saqlash va ulashish» → «Sahifani ilova sifatida oʻrnatish»</b>; Edge — <b>⋯ → «Ilovalar» → «Bu saytni ilova sifatida oʻrnatish»</b>.</li>'
        '<li>Ilova Start menyusida va ish stolida paydo boʻladi; uni vazifalar panelida mahkamlab qoʻyishingiz mumkin.</li></ol>'
        '<p class="small muted">Mac’dagi Safari (14 va yangi): <b>File → Add to Dock</b>.</p></details>'
        '<h3>Bilishingiz kerak</h3><ul class="small">'
        '<li><b>Oflayn rejim:</b> faqat oldin ochgan sahifalaringiz internetsiz ochiladi; yangilari uchun internet kerak. Narxlar sayt qurilgan paytdagi (har ish kuni yangilanadi).</li>'
        '<li><b>Oʻchirish:</b> ilovani boshqa ilovalar kabi oʻchirasiz (Android/iPhone — belgini bosib turib; kompyuter — ilova oynasida ⋮ → «Oʻchirish»).</li>'
        '<li>Bu — taʼlim va tadqiqot vositasi, moliyaviy maslahat emas.</li></ul></div></div>')
    return page("Ilova sifatida oʻrnatish — Aksiya AI", body, "install")


def write_pwa(out: str) -> Dict[str, Any]:
    """manifest.json, sw.js va icons/ ni yozadi (qurilish tamgʻasi service worker nomiga kiradi — har qurilishda yangilanadi)."""
    icons = make_icons(os.path.join(out, "icons"))
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(pwa_manifest(icons), f, ensure_ascii=False, indent=1)
    stamp = re.sub(r"\D", "", META.get("built_local", "")) or "0"
    with open(os.path.join(out, "sw.js"), "w", encoding="utf-8") as f:
        f.write(SW_JS.replace("__STAMP__", stamp))
    return dict(icons=[i["src"] for i in icons], sw=stamp)


def inline_md(s: str) -> str:
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)


def md_to_html(md: str) -> str:
    out: List[str] = []
    ul = False
    for raw in md.strip().splitlines():
        line = raw.rstrip()
        if not line.strip():
            if ul:
                out.append("</ul>")
                ul = False
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m:
            if ul:
                out.append("</ul>")
                ul = False
            lv = min(3, len(m.group(1)) + 1)
            out.append(f"<h{lv}>{inline_md(m.group(2))}</h{lv}>")
        elif line.startswith("* "):
            if not ul:
                out.append("<ul>")
                ul = True
            out.append(f"<li>{inline_md(line[2:])}</li>")
        else:
            if ul:
                out.append("</ul>")
                ul = False
            out.append(f"<p>{inline_md(line)}</p>")
    if ul:
        out.append("</ul>")
    return "".join(out)


def static_note_md(n_full: int, n_lite: int) -> str:
    return f"""
## Bu sayt qanday ishlaydi (statik variant)

* Sayt **serversiz**: har bir aksiya uchun sahifa oldindan hisoblab qoʻyilgan, shuning uchun telefon va kompyuterda **darhol** ochiladi va doim ishlaydi (GitHub Pages).
* GitHub Actions saytni **har ish kuni** AQSh bozori yopilgandan keyin qaytadan quradi: aksiyalar roʻyxati va narxlar yangilanadi, model qayta oʻqitiladi va halol tekshiruvdan oʻtkaziladi.
* Sahifada koʻrinadigan narx — **sayt qurilgan paytdagi** yopilish narxi (real vaqt emas).
* **Ikki xil sahifa.** *Toʻliq tahlil* ({n_full} ta: AI universumi va `extra_tickers.txt`): AI ball, fundamental, yangilik va barcha texnik bloklar. *Yengil tahlil* ({n_lite:,} ta: NYSE, NASDAQ va NYSE American dagi qolgan barcha oddiy aksiyalar va yirik ETFlar): faqat narx tarixidan hisoblangan texnik tahlil — zonalar, trend chiziqlari, patternlar, Elliott, Fibonachchi, Wyckoff, baza, Monte-Karlo va «Qisqa xulosa». Yengil sahifalarda AI, fundamental va yangilik **yoʻq** (ular har bir aksiya uchun Yahoo'ga alohida soʻrov talab qiladi, minglab aksiya uchun esa bu sigʻmaydi).
* **Qamrov chegarasi (halol):** aksiyalar roʻyxati har kuni nasdaqtrader.com dan olinadi. Roʻyxatda yoʻq tikerlar (OTC, chet el birjalari), 120 savdo kunidan kam tarixli (yangi IPO) yoki Yahoo narx bermagan tikerlar saytda boʻlmaydi — qidiruv buni aniq aytadi. Istalgan tikerni **majburan** qoʻshish: GitHub → Actions → *Run workflow* → tikerni yozing (toʻliq tahlil bilan).
* **Maqsad narx** kalkulyatori aksiya sahifasida brauzerning oʻzida ishlaydi (Monte-Karlo ehtimollar jadvalidan interpolyatsiya; yengil sahifalarda 1/3/6 oy ufqlari).
* **«Qisqa xulosa»** — qoidalardan chiqqan daraja xaritasi (yuqoriga va pastga qaysi narxlargacha), bashorat emas; texnik qoidalarning bashorat kuchi bu yerda tekshirilmagan.
"""


# ======================================================================================
# 4. BITTA AKSIYA SAHIFASI
# ======================================================================================
def calc_card(R: Dict[str, Any], lite: bool = False) -> Tuple[str, str]:
    G = app.mc_barrier_grid(R["df"], horizons=[20, 60, 125]) if lite else app.mc_barrier_grid(R["df"])
    G["lab"] = [HORIZON_UZ.get(h, str(h)) for h in G["hs"]]
    G["P0"] = round(G["P0"], 4)
    card = (f'<div class="card info noprint" id="calc"><b>🎯 Maqsad narxga yetish ehtimoli</b>'
            f'<p class="small muted">Hozirgi narx {usd(R["price"])}. Oʻzingiz xohlagan maqsad narxni yozing — Monte-Karlo ehtimollari darhol hisoblanadi.</p>'
            f'<div class="calcrow"><input id="tgt" inputmode="decimal" placeholder="masalan {R["price"] * 1.2:.0f}" aria-label="Maqsad narx">'
            f'<button class="btn" id="tgtgo" type="button">Hisoblash</button></div><div id="tgtout"></div></div>')
    return card, "<script>window.AKSIYA_G=" + json.dumps(G, separators=(",", ":")) + ";</script>"


def ticker_page(R: Dict[str, Any], st: app.State) -> str:
    t = R["ticker"]
    body = app.render_analysis_body(R, st)
    card, gjs = calc_card(R)
    marker = "<h2>Umumiy xulosa nimaga asoslangan</h2>"
    body = body.replace(marker, card + marker, 1) if marker in body else card + body
    fn = f"aksiya_{t}_{R['asof']:%Y-%m-%d}.html"
    tools = (f'<div class="tools noprint"><a class="btn alt" href="../stocks.html">← Barcha aksiyalar</a>'
             f'<a class="btn alt" href="{t}.html" download="{fn}">⬇️ Yuklab olish (HTML)</a></div>')
    name = R["fund"].get("shortName") if R["fund"].get("ok") else None
    ttl = f"{t}{' · ' + name if name else ''} — Aksiya AI"
    return page(ttl, f'<div class="wrapp"><div class="ai">{tools}{body}{DISCLAIMER}</div></div>', "stocks", depth=1, extra_js=gjs)


def summarize(R: Dict[str, Any], st: app.State) -> Dict[str, Any]:
    con, ai = R["conclusion"], R["ai"]
    reliable = [H for H, e in st.evals.items() if e.verdict in ("weak", "strong")]
    ai_pct = None
    if ai and reliable:
        v = [ai["pct"][H] for H in reliable if H in ai["pct"] and np.isfinite(ai["pct"][H])]
        ai_pct = float(np.mean(v)) if v else None
    fund = R["fund"]
    return dict(t=R["ticker"], name=(fund.get("shortName") or "") if fund.get("ok") else "", px=R["price"],
                c1m=R["chg"].get("1 oy"), label=con["label"], score=con["score"], ai_pct=ai_pct,
                wy=int(R["wy_code"]), rsi=float(R["last"].get("rsi14", np.nan)), uni=bool(R["in_universe"]))


# ======================================================================================
# 4b. YENGIL SAHIFALAR (cheklanmagan rejim): bitta sahifa, qisqa qator, parallel qurish
# ======================================================================================
LITE_BUILD_MINUTES = float(os.environ.get("LITE_BUILD_MINUTES", "50"))     # sahifalarni qurishga ajratilgan vaqt
LITE_DL_MINUTES = float(os.environ.get("LITE_DL_MINUTES", "70"))           # narx yuklashga ajratilgan vaqt
LITE_STOOQ_MINUTES = float(os.environ.get("LITE_STOOQ_MINUTES", "6"))      # Yahoo bermaganlar uchun Stooq zaxirasi
LITE_SIZE_MB = float(os.environ.get("LITE_SIZE_MB", "520"))                # yengil sahifalar jami hajmi chegarasi (GitHub Pages ~1 GB)
JOB_MINUTES = float(os.environ.get("AKSIYA_JOB_MINUTES", "105"))            # build_site.py umumiy vaqt chegarasi (workflow timeout dan kichik boʻlsin)
JOB_DEADLINE = 0.0                                                         # build() boshida belgilanadi (epoch soniya)
LITE_BATCH = 24
_LCTX: Dict[str, Any] = {}                                                 # fork orqali ishchi jarayonlarga meros


def _px(p: float) -> float:
    return round(float(p), 4 if p < 1 else 2)


def short_row(R: Dict[str, Any], full: bool, name: str = "") -> Dict[str, Any]:
    """Skaner jadvali uchun ixcham qator: xulosa yoʻnalishi va yuqoriga/pastga birinchi ikki daraja."""
    sh, con = R["short"], R["conclusion"]
    lv = lambda x: [_px(x["price"]), round(float(x["pct"]) * 100, 1)]
    c1 = R["chg"].get("1 oy")
    dv = R.get("dvol")
    return dict(t=R["ticker"], n=name or R.get("name") or "", px=_px(R["price"]), c1=(round(c1 * 100, 1) if c1 is not None else None),
                sc=round(float(con["score"]), 2), lab={"Salbiy": 0, "Ijobiy": 2}.get(con["label"], 1),
                up=[lv(x) for x in sh["up"][:2]], dn=[lv(x) for x in sh["down"][:2]],
                inv=(_px(sh["invalid"]["price"]) if sh.get("invalid") else None),
                dv=(round(dv / 1e6, 2) if dv is not None and np.isfinite(dv) else None), f=1 if full else 0, asof=f"{R['asof']:%Y-%m-%d}")


def lite_ticker_page(R: Dict[str, Any]) -> str:
    t = R["ticker"]
    body = app.render_analysis_body(R, app.LiteState())
    card, gjs = calc_card(R, lite=True)
    marker = "<h2>Umumiy xulosa nimaga asoslangan</h2>"
    body = body.replace(marker, card + marker, 1) if marker in body else card + body
    tools = '<div class="tools noprint"><a class="btn alt" href="../screener.html">← Skaner</a></div>'
    name = R.get("name")
    ttl = f"{t}{' · ' + name if name else ''} — Aksiya AI"
    return page(ttl, f'<div class="wrapp"><div class="ai">{tools}{body}{DISCLAIMER}</div></div>', "screener", depth=1, extra_js=gjs, css_link=True)


def _lite_batch(batch: List[str]) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]], int]:
    c = _LCTX
    rows: List[Dict[str, Any]] = []
    fails: List[Tuple[str, str]] = []
    nbytes = 0
    for t in batch:
        try:
            R = app.analyze_lite(t, c["data"][t], c["market"], name=c["names"].get(t), flag=c["flags"].get(t))
            html = lite_ticker_page(R)
            with open(os.path.join(c["out"], "t", f"{t}.html"), "w", encoding="utf-8") as f:
                f.write(html)
            nbytes += len(html.encode("utf-8"))
            rows.append(short_row(R, False, c["names"].get(t, "")))
        except Exception as e:                                # bitta tiker xatosi qolganlarini toʻxtatmasin
            fails.append((t, f"{type(e).__name__}: {str(e)[:100]}"))
    try:
        app.trim_memory()
    except Exception:
        pass
    return rows, fails, nbytes


def render_lite_pages(order: List[str], procs: int, deadline: float, size_cap: float) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]], float, List[str]]:
    """Sahifalarni parallel (fork) quradi. Qaytaradi: (qatorlar, xatolar, hajm_bayt, vaqt/hajm tufayli qurilmaganlar)."""
    batches = [order[i:i + LITE_BATCH] for i in range(0, len(order), LITE_BATCH)]
    rows: List[Dict[str, Any]] = []
    fails: List[Tuple[str, str]] = []
    total = 0.0
    skipped: List[str] = []
    t0 = time.time()
    ctx = None
    if procs > 1:
        try:
            import multiprocessing as mp
            ctx = mp.get_context("fork")
        except Exception:
            ctx = None
    if ctx is None:                                           # ketma-ket (sinov yoki fork yoʻq)
        for i, b in enumerate(batches):
            if time.time() > deadline or total > size_cap:
                skipped += [t for bb in batches[i:] for t in bb]
                break
            r, f, nb = _lite_batch(b)
            rows += r; fails += f; total += nb
            if (i + 1) % 10 == 0:
                app.log(f"  yengil sahifalar: {len(rows)}/{len(order)} ({time.time() - t0:.0f} s, {total / 1e6:.0f} MB)")
        return rows, fails, total, skipped
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=procs, mp_context=ctx) as ex:
        futs = {ex.submit(_lite_batch, b): b for b in batches}
        done = 0
        stop = False
        for fut in as_completed(futs):
            b = futs[fut]
            if fut.cancelled():
                skipped += b
                continue
            try:
                r, f, nb = fut.result()
            except Exception as e:
                fails += [(t, f"ishchi xatosi: {type(e).__name__}") for t in b]
                continue
            rows += r; fails += f; total += nb
            done += 1
            if done % 20 == 0:
                app.log(f"  yengil sahifalar: {len(rows)}/{len(order)} ({time.time() - t0:.0f} s, {total / 1e6:.0f} MB)")
            if not stop and (time.time() > deadline or total > size_cap):
                stop = True
                app.log(f"  yengil sahifalar: {'vaqt' if time.time() > deadline else 'hajm'} chegarasiga yetildi — qolganlari qurilmaydi")
                for g in futs:
                    g.cancel()
    return rows, fails, total, skipped


def synthetic_symbol_rows(n: int) -> List[Dict[str, Any]]:
    return [dict(sym=f"SY{i:04d}", name=f"Sintetik kompaniya {i} Inc. Common Stock", ex=("Q", "N", "A")[i % 3], test=False, st="N", etf=False) for i in range(n)]


def synthetic_lite_fetcher(t: str, start: str) -> Tuple[Optional[pd.DataFrame], str]:
    """Internetsiz sinov: har bir tiker uchun (tikerga bogʻliq) sintetik narx; baʼzilari qisqa tarixli yoki hajmsiz (chekka holatlar)."""
    h = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
    if h % 97 == 0:
        return None, "No data found, symbol may be delisted"
    d = synthetic_fetch_one(t)
    d = d.tail(630)
    if h % 11 == 0:
        d = d.tail(130 + h % 200)                              # yangi IPO: qisqa tarix
    if h % 53 == 0:
        d = d.copy(); d["Volume"] = 0.0
    return d, ""


def build_lite(out: str, full_set: set, market: pd.Series, synthetic: bool = False, lite_max: int = 0,
               synth_n: int = 0) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, str]]:
    """Toʻliq tahlildan tashqari barcha NYSE/NASDAQ aksiyalari uchun yengil sahifalar. Muvaffaqiyatsizlik butun saytni toʻxtatmaydi."""
    t0 = time.time()
    info: Dict[str, Any] = dict(enabled=True)
    cache_dir = app.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    # 1. roʻyxat
    if synthetic:
        raw, src = synthetic_symbol_rows(synth_n), "sintetik"
    else:
        raw, src = load_symbol_universe(os.path.join(cache_dir, "symdir.json"))
    cands = select_universe(raw, exclude=full_set)
    if lite_max and len(cands) > lite_max:
        cands = cands[:lite_max]
    info.update(source=src, raw=len(raw), candidates=len(cands))
    app.log(f"Yengil rejim: manba {src}; xom qator {len(raw)}, tanlangan aksiya {len(cands)}")
    if len(cands) < 50 and not synthetic:
        raise RuntimeError(f"aksiyalar roʻyxati olinmadi ({len(cands)} ta)")
    names = {c["t"]: c["name"] for c in cands}
    all_names = {}
    for r_ in raw:
        t_ = sym_to_yahoo(r_.get("sym", ""))
        if t_ and t_ not in all_names:
            all_names[t_] = clean_name(r_.get("name", ""))
    flags = {c["t"]: c["flag"] for c in cands if c.get("flag")}
    # 2. narx tarixi. Umumiy vaqt chegarasi (JOB_DEADLINE) ichida: avval sahifa qurishga zaxira ajratiladi, qolgani — yuklashga.
    tickers = [c["t"] for c in cands]
    tail_s = 240.0                                             # yakuniy fayllarni yozish uchun
    rem = (JOB_DEADLINE - time.time()) if (JOB_DEADLINE and not synthetic) else 1e9
    build_res = min(LITE_BUILD_MINUTES * 60.0, max(480.0, 0.30 * rem)) if rem < 1e8 else LITE_BUILD_MINUTES * 60.0
    stooq_s = LITE_STOOQ_MINUTES * 60.0
    dl_s = min(LITE_DL_MINUTES * 60.0, rem - build_res - stooq_s - tail_s) if rem < 1e8 else LITE_DL_MINUTES * 60.0
    info.update(dl_budget_min=(round(dl_s / 60, 1) if rem < 1e8 else None))
    if not synthetic and dl_s < 120.0:
        raise RuntimeError(f"umumiy vaqt chegarasida yengil bosqichga vaqt qolmadi (qolgan {rem / 60:.0f} daqiqa)")
    spy_last = market.index[-1]
    cache: Dict[str, pd.DataFrame] = {} if synthetic else load_price_cache()
    fresh = lambda t: t in cache and cache[t].index[-1] >= spy_last - pd.Timedelta(days=7)
    if cache:                                                   # chekllov boʻlsa har kuni bir xil tikerlar emas: keshda yoʻqlari birinchi, keshdagilari eskisidan boshlab
        tickers = ([t for t in tickers if t not in cache] + sorted((t for t in tickers if t in cache), key=lambda t: cache[t].index[-1]))
    if synthetic:
        data, status = download_lite(tickers, fetcher=synthetic_lite_fetcher, workers=4, budget_s=600.0, log_every=1000)
    else:
        app.log(f"Yengil rejim vaqt rejasi: yuklash <= {dl_s / 60:.0f} daq, Stooq <= {stooq_s / 60:.0f} daq, sahifalar <= {build_res / 60:.0f} daq "
                f"(umumiy qolgan {rem / 60:.0f} daq)")
        data, status = download_lite(tickers, budget_s=dl_s, workers=int(os.environ.get("LITE_DL_WORKERS", "8")))
        miss = [t for t, k in status.items() if k != "gone"]
        if miss and len(data) < 0.95 * (len(tickers) - sum(1 for k in status.values() if k == "gone")):
            left = (JOB_DEADLINE - time.time()) if JOB_DEADLINE else 1e9
            got = stooq_rescue(miss, data, status, budget_s=max(0.0, min(stooq_s, left - build_res - tail_s)))
            app.log(f"Stooq zaxirasi: {got} ta tiker olindi")
    stale = 0
    if not synthetic:
        for t in tickers:
            if t not in data and status.get(t) != "gone" and fresh(t):
                data[t] = cache[t]
                stale += 1
    keep: Dict[str, pd.DataFrame] = {}
    short, old = [], []
    for t, d in data.items():
        if d.index[-1] < spy_last - pd.Timedelta(days=10):
            old.append(t)                                      # savdosi toʻxtagan / delisting
        elif len(d) < app.LITE_MIN_BARS:
            short.append(t)
        else:
            keep[t] = d
    info.update(downloaded=len(data), stale_from_cache=stale, too_short=len(short), outdated=len(old), gone=sum(1 for k in status.values() if k == "gone"),
                dl_failed=sum(1 for k in status.values() if k in ("rate", "other", "time")), usable=len(keep),
                dl_minutes=round((time.time() - t0) / 60, 1))
    app.log(f"Yengil rejim: yuklandi {len(data)} (keshdan {stale}), yaroqli {len(keep)}, qisqa tarix {len(short)}, eskirgan {len(old)}, "
            f"'delisted' {info['gone']}, yuklanmagan {info['dl_failed']}")
    if not synthetic and keep:
        save_price_cache({t: d for t, d in keep.items()})
    if not keep:
        raise RuntimeError("birorta ham narx tarixi yuklanmadi")
    # 3. likvidlik boʻyicha tartib (eng likvidlari birinchi: hajm/vaqt chegarasida oxirgilari tushib qoladi)
    dv = {t: float((d["Close"] * d["Volume"]).tail(20).median()) for t, d in keep.items()}
    order = sorted(keep, key=lambda t: -(dv[t] if np.isfinite(dv[t]) else 0.0))
    # 4. sahifalar
    os.makedirs(os.path.join(out, "t"), exist_ok=True)
    _LCTX.clear()
    _LCTX.update(data=keep, market=market, names=names, flags=flags, out=out)
    procs = int(os.environ.get("LITE_PROCS", "0")) or (os.cpu_count() or 1)
    dl_build = time.time() + LITE_BUILD_MINUTES * 60.0
    if JOB_DEADLINE and not synthetic:
        dl_build = min(dl_build, JOB_DEADLINE - tail_s)
    rows, fails, nbytes, skipped = render_lite_pages(order, procs, dl_build, LITE_SIZE_MB * 1e6)
    _LCTX.clear()
    info.update(pages=len(rows), build_failed=len(fails), skipped=len(skipped), size_mb=round(nbytes / 1e6, 1), procs=procs,
                total_minutes=round((time.time() - t0) / 60, 1), fail_examples=fails[:8])
    app.log(f"Yengil sahifalar tayyor: {len(rows)} ta, {nbytes / 1e6:.0f} MB, xato {len(fails)}, qurilmagan {len(skipped)}; {info['total_minutes']} daqiqa ({procs} jarayon)")
    return rows, info, {**all_names, **names}


# ======================================================================================
# 5. UMUMIY SAHIFALAR
# ======================================================================================
def stocks_page(rows: List[Dict[str, Any]], failed: List[Tuple[str, str]], st: app.State) -> str:
    any_ai = any(r["ai_pct"] is not None for r in rows)
    trs = []
    for r in sorted(rows, key=lambda r: -r["score"]):
        cls = {"Ijobiy": "good", "Salbiy": "bad"}.get(r["label"], "")
        arrow = "▲" if r["label"] == "Ijobiy" else "▼" if r["label"] == "Salbiy" else "●"
        lab = "Neytral" if r["label"].startswith("Neytral") else r["label"]
        ai_cell = (f'<td class="n" data-v="{r["ai_pct"] * 100:.1f}">{r["ai_pct"] * 100:.0f}%</td>' if r["ai_pct"] is not None
                   else '<td class="n" data-v="">—</td>')
        extra = "" if r["uni"] or r["t"] == app.BENCH else ' <span class="tag small" title="AI universumiga kirmaydi">qoʻshimcha</span>'
        trs.append(
            f'<tr data-s="{esc((r["t"] + " " + r["name"]).lower())}" data-l="{esc(lab)}">'
            f'<td data-v="{r["t"]}"><a class="tl" href="t/{r["t"]}.html">{esc(r["t"])}</a>{extra}'
            f'<div class="small muted" style="white-space:normal">{esc(r["name"][:34])}</div></td>'
            f'<td class="n" data-v="{r["px"]:.4f}">{usd(r["px"])}</td>'
            f'<td class="n hm" data-v="{(r["c1m"] if r["c1m"] is not None else float("nan")) * 100:.2f}">{pc(r["c1m"], 1, True) if r["c1m"] is not None else "—"}</td>'
            f'<td data-v="{r["score"]:.3f}"><span class="tag {cls}">{arrow} {esc(lab)} {r["score"]:+.2f}</span></td>'
            f'{ai_cell}<td class="hm" data-v="{r["wy"]}">{WY_SHORT.get(r["wy"], "—")}</td>'
            f'<td class="n hm" data-v="{r["rsi"]:.1f}">{r["rsi"]:.0f}</td></tr>')
    head = ('<thead><tr><th class="s" data-k="0">Tiker</th><th class="s n" data-k="1">Narx</th><th class="s n hm" data-k="2">1 oy</th>'
            '<th class="s" data-k="3">Umumiy xulosa</th><th class="s n" data-k="4" title="Faqat tarixiy sinovdan oʻtgan ufqlar boʻyicha">AI persentil</th>'
            '<th class="s hm" data-k="5">Wyckoff</th><th class="s n hm" data-k="6">RSI</th></tr></thead>')
    ai_note = ("AI persentil — aksiyaning universum ichidagi oʻrni (100% = model eng yuqori baho bergan), faqat tarixiy sinovdan oʻtgan ufqlar boʻyicha."
               if any_ai else "AI persentil “—”: model hozir tarixiy sinovda ishonchli ustunlik koʻrsatmadi, shuning uchun AI bali koʻrsatilmaydi (“Sifat” sahifasiga qarang).")
    fl = ""
    if failed:
        fl = ('<p class="small muted">Tayyorlanmadi: ' + ", ".join(f"<b>{esc(t)}</b> ({esc(why[:80])})" for t, why in failed) + ".</p>")
    body = (f'<div class="wrapp"><div class="ai"><h2>Toʻliq tahlil qilingan aksiyalar ({len(rows)} ta): AI universumi va qoʻshimcha tikerlar</h2>'
            f'<p class="small muted">Tikerni bosing — toʻliq tahlil. Sarlavhani bosib saralash mumkin. “Umumiy xulosa” — <b>tavsiya emas</b>. '
            f'Boshqa barcha NYSE/NASDAQ aksiyalari uchun yengil tahlil bor: <a href="screener.html">Skaner →</a></p>'
            f'<details class="small muted"><summary>Ustunlar izohi</summary><p>Toʻliq tahlilda: grafik, Monte-Karlo, support/resistance, Wyckoff, baza va breakout maqsadlari, '
            f'Gann, sham, fundamental, yangilik, maqsad narx kalkulyatori. “Umumiy xulosa” — klassik qoidalar va (agar sinovdan oʻtgan boʻlsa) AI aralashmasi. {esc(ai_note)}</p></details>'
            f'<div class="flt"><input id="flt-q" placeholder="Qidirish: tiker yoki kompaniya nomi" aria-label="Qidirish">'
            f'<select id="flt-l"><option value="">Hammasi</option><option>Ijobiy</option><option>Neytral</option><option>Salbiy</option></select>'
            f'<span class="small muted" style="align-self:center">koʻrsatilmoqda: <b id="flt-n">{len(rows)} ta</b></span></div>'
            f'<div class="wrap"><table class="sortable">{head}<tbody>{"".join(trs)}</tbody></table></div>{fl}'
            f'<p class="small muted">Yengil tahlildagi aksiyaga <b>toʻliq tahlil</b> (AI ball, fundamental, yangilik) qoʻshish: GitHub → Actions → «Saytni qurish va joylash» → Run workflow (tikerlar maydoni). '
            f'Qoʻshilgan tikerlar extra_tickers.txt da saqlanadi va har kuni yangilanadi.</p>{DISCLAIMER}</div></div>')
    return page("Barcha aksiyalar — Aksiya AI", body, "stocks")


def index_page(st: app.State, built: Optional[set] = None, n_all: int = 0) -> str:
    html = app.render_reco(st)
    html = html.replace(CSS, "", 1)
    html = re.sub(r'<div class="lbl">Tahlil vaqti</div><b>[^<]*</b>',
                  f'<div class="lbl">Sayt qurilgan</div><b>{META["built_local"]} (Toshkent)</b>', html, count=1)
    html = re.sub(r'<div class="lbl">Keyingi yangilanish</div><b>[^<]*</b>',
                  '<div class="lbl">Yangilanish</div><b>har ish kuni</b>', html, count=1)
    picks, _ = app.reco_picks(st)
    def _lnk(m: "re.Match") -> str:                      # sahifasi qurilmagan tikerga havola bermaymiz (404 boʻlmasin)
        t = m.group(1)
        if built is not None and t not in built:
            return m.group(0)
        return f'<span class="tk"><a class="tl" href="t/{t}.html">{t}</a></span>'

    html = re.sub(r'<span class="tk">([A-Z0-9\-]+)</span>', _lnk, html)
    if len(picks) and (built is None or str(picks.index[0]) in built):
        t1 = str(picks.index[0])
        html = html.replace(f"<h2>Birinchi nomzodning chuqur tahlili: {esc(t1)}</h2>",
                            f'<h2>Birinchi nomzodning chuqur tahlili: {esc(t1)}</h2><p class="small"><a href="t/{t1}.html">'
                            f'Toʻliq sahifani ochish (maqsad narx kalkulyatori bilan) →</a></p>', 1)
    card = ""
    if n_all:
        card = (f'<div class="ai"><div class="card info"><b>🔎 Istalgan aksiya:</b> yuqoridagi qidiruvga tikerni yozing — {n_all:,} ta NYSE/NASDAQ aksiyasi uchun tayyor tahlil bor: '
                f'zonalar, trend chiziqlari, patternlar, Elliott, Fibonachchi va oxirida <b>«Qisqa xulosa»</b> (yuqoriga va pastga qaysi narxlargacha). '
                f'<a href="screener.html">Hammasini skaner qilish →</a></div></div>')
    return page("Aksiya AI — tavsiyalar", f'<div class="wrapp">{card}{html}</div>', "index")


def quality_page(st: app.State) -> str:
    return page("Model sifati — Aksiya AI", f'<div class="wrapp">{app.render_quality(st).replace(CSS, "", 1)}</div>', "quality")


def method_page(n_full: int = 0, n_lite: int = 0) -> str:
    md = app.METHODOLOGY_MD.replace("Tizim har ~12 soatda maʼlumotni yangilab, modelni qaytadan oʻqitadi va tekshiradi.",
                                    "Sayt har ish kuni maʼlumotni yangilab, modelni qaytadan oʻqitadi va tekshiradi.")
    body = (f'<div class="wrapp"><div class="ai">{md_to_html(md)}{md_to_html(static_note_md(n_full, n_lite))}{DISCLAIMER}</div></div>')
    return page("Usul — Aksiya AI", body, "method")


def site_base() -> str:
    """404 sahifasi har qanday chuqurlikdagi manzilda ochilishi mumkin: nisbiy havolalar buzilmasligi uchun <base> kerak."""
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if "/" not in repo:
        return ""
    owner, name = repo.split("/", 1)
    return "/" if name.lower() == f"{owner.lower()}.github.io" else f"/{name}/"


def not_found_page() -> str:
    b = site_base()
    body = ('<div class="wrapp"><div class="ai"><div class="card warn"><b>Sahifa topilmadi</b><p>Bu manzil sayt tarkibida yoʻq. '
            'Agar siz aksiya sahifasini qidirgan boʻlsangiz, u hali tayyorlanmagan boʻlishi mumkin: yuqoridagi qidiruvga tikerni yozing — '
            'tayyor boʻlmasa, sababi va majburan qoʻshish yoʻli koʻrsatiladi.</p></div></div></div>')
    return page("Topilmadi — Aksiya AI", body, "screener", head_extra=(f'<base href="{esc(b)}">' if b else ""), body_attr='data-nf="1"', css_link=bool(b))


# ======================================================================================
# 5b. SKANER (barcha aksiyalar) VA QIDIRUV MAʼLUMOTI
# ======================================================================================
SCREENER_JS = r"""(function(){
var D=window.AKSIYA_S||[],body=document.getElementById('sc-body');
if(!body){return;}
var q=document.getElementById('sc-q'),fl=document.getElementById('sc-l'),ft=document.getElementById('sc-t'),more=document.getElementById('sc-more'),cnt=document.getElementById('sc-n'),ths=document.querySelectorAll('#sc-tab th.s');
var sortK=0,asc=true,shown=100,F=[];
var ARR=['▼','●','▲'],CLS=['bad','','good'],LAB=['Salbiy','Neytral','Ijobiy'];
function esc(s){return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function fp(p){return p>=1000?p.toLocaleString('en-US',{maximumFractionDigits:0}):p>=1?p.toFixed(2):p.toFixed(4);}
function fc(c){return (c>=0?'+':'−')+Math.abs(c).toFixed(1)+'%';}
function lv(a){return a?('$'+fp(a[0])+' <span class="muted">('+fc(a[1])+')</span>'):'—';}
function two(a){if(!a||!a.length){return '\u2014';}
  var d='<span class="dm">'+lv(a[0])+(a.length>1?'<br><span class="small muted">maqsad: $'+fp(a[1][0])+' ('+fc(a[1][1])+')</span>':'')+'</span>';
  var m='<span class="mo">$'+fp(a[0][0])+' <span class="muted">'+fc(a[0][1])+'</span>'+(a.length>1?'<br><span class="small muted">\u2192 $'+fp(a[1][0])+'</span>':'')+'</span>';
  return d+m;}
function key(r,k){switch(k){case 0:return r[0];case 1:return r[2];case 2:return r[3];case 3:return r[4];case 4:return r[6].length?r[6][0][1]:null;case 5:return r[7].length?r[7][0][1]:null;}return null;}
function apply(){
  var s=(q.value||'').toLowerCase().trim(),l=fl.value,t=ft.value;
  F=D.filter(function(r){
    if(l!==''&&String(r[5])!==l){return false;}
    if(t!==''&&String(r[10])!==t){return false;}
    if(s&&r[0].toLowerCase().indexOf(s)<0&&String(r[1]||'').toLowerCase().indexOf(s)<0){return false;}
    return true;});
  F.sort(function(a,b){var x=key(a,sortK),y=key(b,sortK);
    if(x===null&&y===null){return 0;}if(x===null){return 1;}if(y===null){return -1;}
    if(x<y){return asc?-1:1;}if(x>y){return asc?1:-1;}return 0;});
  shown=100;render();
}
function render(){
  var h=[],n=Math.min(F.length,shown);
  for(var i=0;i<n;i++){var r=F[i],c=CLS[r[5]];
    h.push('<tr><td><a class="tl" href="t/'+r[0]+'.html">'+esc(r[0])+'</a>'+(r[10]?' <span class="tag small" title="Toʻliq tahlil: AI ball, fundamental, yangilik bor">toʻliq</span>':'')
      +'<br class="mo"><span class="small mo">$'+fp(r[2])+'</span>'
      +'<div class="small muted nm">'+esc(String(r[1]||'').slice(0,34))+'</div></td>'
      +'<td class="n hm">$'+fp(r[2])+'</td><td class="n hm">'+(r[3]==null?'\u2014':fc(r[3]))+'</td>'
      +'<td><span class="tag '+c+'">'+ARR[r[5]]+'<span class="dm"> '+LAB[r[5]]+'</span> '+(r[4]>=0?'+':'\u2212')+Math.abs(r[4]).toFixed(2)+'</span></td>'
      +'<td class="n">'+two(r[6])+'</td><td class="n">'+two(r[7])+'</td>'
      +'<td class="n hm">'+(r[8]==null?'\u2014':'$'+fp(r[8]))+'</td></tr>');}
  body.innerHTML=h.join('')||'<tr><td colspan="7" class="muted">Hech narsa topilmadi.</td></tr>';
  cnt.textContent=F.length+' ta topildi'+(F.length>n?' (dastlabki '+n+' ta koʻrsatilmoqda)':'');
  more.style.display=F.length>n?'':'none';
}
ths.forEach(function(th){th.addEventListener('click',function(){
  var k=parseInt(th.getAttribute('data-k'));asc=(k===sortK)?!asc:true;sortK=k;
  ths.forEach(function(x){x.classList.remove('asc','desc');});th.classList.add(asc?'asc':'desc');apply();});});
more.addEventListener('click',function(){shown+=100;render();});
q.addEventListener('input',apply);fl.addEventListener('change',apply);ft.addEventListener('change',apply);
apply();
})();
"""


def screener_page(n_all: int, n_full: int, n_lite: int, asof: str) -> str:
    head = ('<thead><tr><th class="s asc" data-k="0">Tiker</th><th class="s n hm" data-k="1">Narx</th><th class="s n hm" data-k="2">1 oy</th>'
            '<th class="s" data-k="3">Umumiy xulosa</th><th class="s n" data-k="4" title="Yuqorida birinchi toʻsiq va hisoblangan maqsad">Yuqoriga (toʻsiq → maqsad)</th>'
            '<th class="s n" data-k="5" title="Pastda birinchi tayanch va hisoblangan daraja">Pastga (tayanch → daraja)</th>'
            '<th class="n hm" title="Shu narxdan yuqoriga (ijobiy) yoki pastga (salbiy) yopilsa xulosa kuchini yoʻqotadi">Bekor qilish</th></tr></thead>')
    body = (f'<div class="wrapp"><div class="ai"><h2>Skaner: {n_all:,} ta aksiya</h2>'
            f'<p class="small muted">Har bir aksiya uchun hisoblangan <b>«Qisqa xulosa»</b>: umumiy yoʻnalish va yuqorida/pastda qaysi narx darajalarigacha borishi mumkinligi '
            f'(support/resistance, zonalar, trend kanali, patternlar, Elliott, Fibonachchi, baza maqsadlari ichidan konfluens boʻyicha tanlangan). '
            f'Bu — qoidalardan chiqqan <b>daraja xaritasi, tavsiya yoki bashorat emas</b>. Tikerni bossangiz — toʻliq sahifa (grafiklar va barcha hisob-kitoblar). '
            f'Narx sanasi: {esc(asof)}.</p>'
            f'<details class="small muted"><summary>Ikki xil tahlil turi</summary><p><b>Toʻliq</b> ({n_full} ta): AI ball, fundamental va yangilik bor. '
            f'<b>Yengil</b> ({n_lite:,} ta): faqat narx tarixidan hisoblangan texnik tahlil — AI, fundamental va yangilik yoʻq. '
            f'Yengil tahlildagi aksiya uchun toʻliq tahlil kerak boʻlsa, uning sahifasidagi havola orqali soʻrang.</p></details>'
            f'<div class="flt"><input id="sc-q" placeholder="Qidirish: tiker yoki kompaniya nomi" aria-label="Qidirish" autocomplete="off">'
            f'<select id="sc-l" aria-label="Xulosa"><option value="">Barcha xulosalar</option><option value="2">▲ Ijobiy</option><option value="1">● Neytral</option>'
            f'<option value="0">▼ Salbiy</option></select>'
            f'<select id="sc-t" aria-label="Tahlil turi"><option value="">Toʻliq va yengil</option><option value="1">Faqat toʻliq tahlil</option><option value="0">Faqat yengil tahlil</option></select></div>'
            f'<p class="small muted" id="sc-n">Yuklanmoqda…</p>'
            f'<div class="wrap"><table class="scr" id="sc-tab">{head}<tbody id="sc-body"></tbody></table></div>'
            f'<p><button class="btn alt" id="sc-more" type="button" style="display:none">Yana 100 ta koʻrsatish</button></p>{DISCLAIMER}</div></div>')
    return page("Skaner — Aksiya AI", body, "screener", extra_js='<script src="screener.js"></script>', tail_js=SCREENER_JS)


def write_search_data(out: str, rows: List[Dict[str, Any]]) -> None:
    """tickers.js (qidiruv takliflari: har sahifada kechiktirib yuklanadi) va screener.js (faqat skaner sahifasida)."""
    rows = sorted(rows, key=lambda r: r["t"])
    with open(os.path.join(out, "tickers.js"), "w", encoding="utf-8") as f:
        f.write("window.AKSIYA_T=" + json.dumps([[r["t"], r["n"], r["f"]] for r in rows], separators=(",", ":"), ensure_ascii=False) + ";")
    arr = [[r["t"], r["n"], r["px"], r["c1"], r["sc"], r["lab"], r["up"], r["dn"], r["inv"], r["dv"], r["f"]] for r in rows]
    with open(os.path.join(out, "screener.js"), "w", encoding="utf-8") as f:
        f.write("window.AKSIYA_S=" + json.dumps(arr, separators=(",", ":"), ensure_ascii=False) + ";")


# ======================================================================================
# 6. SINTETIK REJIM (internetsiz sinov)
# ======================================================================================
def synthetic_fetch_one(ticker: str, years: float = 8) -> pd.DataFrame:
    if ticker == "SPSC":
        return app.base_breakout_series()
    seed = int(hashlib.md5(ticker.encode()).hexdigest()[:6], 16)
    return app.make_synthetic_universe(n_stocks=1, days=1700, seed=seed)["S001"]


# ======================================================================================
# 7. ASOSIY QURISH
# ======================================================================================
META: Dict[str, Any] = {"built_local": "", "asof": ""}


def build(out: str, synthetic: bool = False, extra_env: str = "", max_tickers: int = 0, synth_n: int = 60,
          lite: bool = True, lite_max: int = 0, lite_synth_n: int = 0) -> int:
    global JOB_DEADLINE
    t_all = time.time()
    JOB_DEADLINE = t_all + JOB_MINUTES * 60.0
    extras = read_extra(EXTRA_FILE, extra_env)
    if synthetic:
        app.fetch_one = synthetic_fetch_one
        data = app.make_synthetic_universe(n_stocks=synth_n, days=1700, seed=7, alpha=0.0006)
    else:
        data = None
    app.log(f"Qoʻshimcha tikerlar ({len(extras)}): {', '.join(extras) or '—'}")
    st = app.build_state(data=data)
    uni = [t for t in st.data if t != app.BENCH]
    todo = list(dict.fromkeys(uni + extras + [app.BENCH]))
    if max_tickers:
        todo = list(dict.fromkeys(todo[:max_tickers] + [t for t in extras if t in todo][:5]))
    prefetch_info(todo, offline=synthetic)
    now = dt.datetime.now(dt.timezone.utc)
    META["built_local"] = now.astimezone(TZ_TASH).strftime("%Y-%m-%d %H:%M")
    META["asof"] = f"{st.asof:%Y-%m-%d}"
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    app.FULL_ANALYSIS_URL = f"https://github.com/{repo}/actions/workflows/{WORKFLOW_FILE}" if repo else ""

    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "t"))
    rows: List[Dict[str, Any]] = []
    srows: List[Dict[str, Any]] = []                     # skaner qatorlari (toʻliq tahlil)
    failed: List[Tuple[str, str]] = []
    t0 = time.time()
    for i, t in enumerate(todo, 1):
        try:
            R = app.analyze_ticker(t, st, None, use_cache=False)
            with open(os.path.join(out, "t", f"{t}.html"), "w", encoding="utf-8") as f:
                f.write(ticker_page(R, st))
            rows.append(summarize(R, st))
            srows.append(short_row(R, True, rows[-1]["name"]))
        except Exception as e:                           # bitta tiker butun qurilishni toʻxtatmasin
            failed.append((t, f"{type(e).__name__}: {str(e)[:100]}"))
            app.log(f"  {t}: oʻtkazib yuborildi — {type(e).__name__}: {str(e)[:100]}")
        if i % 10 == 0 or i == len(todo):
            app.log(f"  sahifalar: {i}/{len(todo)} ({time.time() - t0:.0f} s)")
        if i % 25 == 0:
            app.trim_memory()
    n_uni_ok = sum(1 for r in rows if r["uni"])
    if n_uni_ok < 0.6 * len(uni) and not max_tickers:
        app.log(f"XATO: universum sahifalarining faqat {n_uni_ok}/{len(uni)} tasi tayyor — sayt joylanmaydi.")
        return 2
    names = sorted(r["t"] for r in rows)

    # --- yengil bosqich: qolgan barcha NYSE/NASDAQ aksiyalari. Xato boʻlsa ham toʻliq sahifalar bilan sayt joylanadi.
    lite_rows: List[Dict[str, Any]] = []
    lite_info: Dict[str, Any] = dict(enabled=False)
    lite_names: Dict[str, str] = {}
    if lite and not max_tickers and (not synthetic or lite_synth_n > 0):
        try:
            lite_rows, lite_info, lite_names = build_lite(out, set(names), st.data[app.BENCH]["Close"], synthetic=synthetic,
                                                          lite_max=lite_max, synth_n=lite_synth_n)
        except Exception as e:
            lite_info = dict(enabled=True, error=f"{type(e).__name__}: {str(e)[:200]}")
            app.log(f"YENGIL BOSQICH XATOSI (sayt faqat toʻliq sahifalar bilan joylanadi): {lite_info['error']}")
    for r in srows:                                       # toʻliq sahifada kompaniya nomi boʻlmasa — birja roʻyxatidagi nom
        if not r["n"]:
            r["n"] = lite_names.get(r["t"], "")
    for r in rows:
        if not r["name"]:
            r["name"] = lite_names.get(r["t"], "")
    n_full, n_lite = len(rows), len(lite_rows)
    all_rows = srows + lite_rows
    n_all = len(all_rows)

    with open(os.path.join(out, "data.js"), "w", encoding="utf-8") as f:
        f.write("window.AKSIYA=" + json.dumps(dict(repo=repo, built=META["built_local"], n=dict(full=n_full, lite=n_lite, all=n_all)),
                                              separators=(",", ":")) + ";")
    with open(os.path.join(out, "site.js"), "w", encoding="utf-8") as f:
        f.write(SITE_JS)
    with open(os.path.join(out, "style.css"), "w", encoding="utf-8") as f:
        f.write(re.sub(r"</?style>", "", CSS + PAGE_CSS))
    open(os.path.join(out, ".nojekyll"), "w").close()
    write_search_data(out, all_rows)
    try:
        pwa_info = write_pwa(out)                         # ilova rejimi: manifest, belgilar, service worker (xato boʻlsa ham sayt qurilaveradi)
    except Exception as e:                                # noqa: BLE001
        pwa_info = dict(error=f"{type(e).__name__}: {str(e)[:120]}")
        app.log(f"  ILOVA REJIMI (PWA) fayllari yozilmadi: {pwa_info['error']}")
    writes = {"index.html": index_page(st, built=set(names), n_all=n_all), "stocks.html": stocks_page(rows, failed, st),
              "screener.html": screener_page(n_all, n_full, n_lite, META["asof"]),
              "quality.html": quality_page(st), "method.html": method_page(n_full, n_lite), "404.html": not_found_page(),
              "install.html": install_page()}
    for fn, html in writes.items():
        with open(os.path.join(out, fn), "w", encoding="utf-8") as f:
            f.write(html)
    ev = st.evals[st.primary_h]
    meta = dict(version=app.APP_VERSION, built_utc=now.isoformat(timespec="seconds"), asof=META["asof"], pages=n_all, full_pages=n_full, lite_pages=n_lite,
                universe=len(uni), extras=extras, failed=failed, primary_horizon=st.primary_h, primary_verdict=ev.verdict,
                verdicts={str(h): e.verdict for h, e in sorted(st.evals.items())}, synthetic=synthetic, lite=lite_info, pwa=pwa_info,
                build_minutes=round((time.time() - t_all) / 60, 1))
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    size = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(out) for fn in fns) / 1e6
    app.log(f"TAYYOR: {n_all} ta sahifa ({n_full} toʻliq + {n_lite} yengil), {size:.1f} MB, {meta['build_minutes']} daqiqa; asosiy ufq hukmi: {ev.verdict}")
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ and not synthetic:
        try:
            with open(summ, "a", encoding="utf-8") as f:
                f.write(f"### Sayt qurildi\n\n- Sahifalar: **{n_all}** ta ({n_full} toʻliq + {n_lite} yengil; {size:.0f} MB), maʼlumot sanasi **{META['asof']}**\n"
                        f"- Model hukmi ({HORIZON_UZ.get(st.primary_h)}): **{ev.verdict}**; ufqlar: {meta['verdicts']}\n"
                        + (f"- Toʻliq tahlilda tayyorlanmadi: {failed}\n" if failed else "")
                        + (f"- Yengil bosqich: {json.dumps({k: v for k, v in lite_info.items() if k != 'fail_examples'}, ensure_ascii=False)}\n" if lite_info.get("enabled") else ""))
        except Exception:
            pass
    return 0


# ======================================================================================
# 9. SELFTEST (internetsiz): yengil rejim mantiqi
# ======================================================================================
_ST_NASDAQ = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
AACG|ATA Creativity Global - American Depositary Shares, each representing two common shares|G|N|D|100|N|N
ACAHU|Atlantic Coastal Acquisition Corp. II Units, each consisting of one Class A ordinary share and one-half of one redeemable warrant|G|N|N|100|N|N
ACAHW|Atlantic Coastal Acquisition Corp. II Warrants, each whole warrant exercisable for one Class A ordinary share|G|N|N|100|N|N
ZZZZT|NASDAQ TEST STOCK|G|Y|N|100|N|N
QQQ|Invesco QQQ Trust, Series 1|G|N|N|100|Y|N
QQEW|First Trust NASDAQ-100 Equal Weighted Index Fund|G|N|N|100|Y|N
GOOGL|Alphabet Inc. - Class A Common Stock|Q|N|N|100|N|N
HOLO|MicroCloud Hologram Inc. - Class A Ordinary Shares|S|N|N|100|N|N
FUNDX|Some Closed End Fund Shares of Beneficial Interest|G|N|N|100|N|N
File Creation Time: 1122202509:30|||||||
"""
_ST_OTHER = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
A|Agilent Technologies, Inc. Common Stock|N|A|N|100|N|A
BRK.B|Berkshire Hathaway Inc. Class B Common Stock New|N|BRK.B|N|100|N|BRK$B
BAC$L|Bank of America Corporation Depositary Shares, each representing a 1/1200th interest in a share of Non-Cumulative Preferred Stock, Series L|N|BAC PRL|N|100|N|BAC$L
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
XYZETF|Random Sector ETF|P|XYZETF|Y|100|N|XYZETF
TSM|Taiwan Semiconductor Manufacturing Company Ltd. American Depositary Shares|N|TSM|N|100|N|TSM
MOG.A|Moog Inc. Class A Common Stock|N|MOG.A|N|100|N|MOG$A
ABCD.WS|ABCD Acquisition Corp Warrants|N|ABCD WS|N|100|N|ABCD.WS
AMEX1|Small Amex Co Common Stock|A|AMEX1|N|100|N|AMEX1
BATS1|Some BATS thing|Z|BATS1|N|100|N|BATS1
ZTEST|Test Co Common Stock|N|ZTEST|N|100|Y|ZTEST
File Creation Time: 1122202509:30|||||||
"""


def selftest(verbose: bool = True) -> bool:
    """Yengil rejim mantiqini internetsiz tekshiradi: tiker roʻyxatini oʻqish va saralash, yuklash mantiqi (cheklov, yopilgan tikerlar,
    vaqt byudjeti, tez toʻxtash). Haqiqiy Yahoo/nasdaqtrader javoblari bu yerda sinalmaydi — ular faqat GitHub Actions ishida koʻrinadi."""
    import threading
    from collections import Counter
    ok_all = True

    def rep(name: str, ok: bool, msg: str) -> None:
        nonlocal ok_all
        ok_all &= bool(ok)
        if verbose:
            print(("  [OK]   " if ok else "  [XATO] ") + name + " — " + msg, flush=True)

    # 1. tiker roʻyxati
    rows = parse_nasdaqlisted(_ST_NASDAQ) + parse_otherlisted(_ST_OTHER)
    sel = {r["t"]: r for r in select_universe(rows)}
    want_in = ["AAPL", "AACG", "GOOGL", "HOLO", "A", "BRK-B", "TSM", "MOG-A", "AMEX1", "SPY", "QQQ"]
    want_out = ["ACAHU", "ACAHW", "ZZZZT", "QQEW", "FUNDX", "BAC$L", "XYZETF", "ABCD-WS", "BATS1", "ZTEST"]
    miss = [t for t in want_in if t not in sel]
    extra = [t for t in want_out if t in sel]
    rep("Tiker roʻyxati: oddiy aksiyalar (ADR, klass A/B) kiradi; varrant, birlik, test, fond, imtiyozli aksiya kirmaydi", not miss and not extra,
        f"{len(sel)} ta tanlandi" + (f"; yoʻq: {miss}" if miss else "") + (f"; ortiqcha: {extra}" if extra else ""))
    ex = {r["t"] for r in select_universe(rows, exclude={"AAPL", "GOOGL"})}
    rep("Toʻliq tahlildagi tikerlar yengil roʻyxatdan chiqariladi", "AAPL" not in ex and "GOOGL" not in ex and "HOLO" in ex, f"{len(ex)} ta qoldi")
    syms = {raw: sym_to_yahoo(raw) for raw in ("BRK.B", "BF.B", "BAC$L", "GOOGL", "XYZ.WS", "")}
    rep("Belgilarni Yahoo koʻrinishiga oʻtkazish (BRK.B → BRK-B)", syms["BRK.B"] == "BRK-B" and syms["BF.B"] == "BF-B" and syms["GOOGL"] == "GOOGL" and not syms["BAC$L"] and not syms[""],
        str({k: v for k, v in syms.items() if k}))

    # 2. yuklash mantiqi (soxta yuklovchi bilan, uxlashlar tezlatilgan)
    real_sleep = time.sleep
    time.sleep = lambda x: real_sleep(min(x, 5) / 200.0)
    try:
        n = 300
        idx = pd.bdate_range(end="2026-11-20", periods=n)
        c = 50 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, n)))
        df0 = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1e6}, index=idx)
        lock, calls = threading.Lock(), {}

        def fetcher(kind: str):
            def f(t: str, start: Any):
                with lock:
                    calls[t] = calls.get(t, 0) + 1
                    k = calls[t]
                h = sum(map(ord, t))
                if kind == "mixed":
                    if h % 17 == 0:
                        return None, "YFTzMissingError('$X: possibly delisted; no timezone found')"
                    if h % 13 == 0 and k <= 2:
                        return None, "YFRateLimitError('Too Many Requests. Rate limited. Try after a while.')"
                    if h % 19 == 0:
                        return None, "Timeout: read timed out"
                    if h % 23 == 0 and k == 1:
                        raise RuntimeError("tasodifiy xato")
                    return df0.copy(), ""
                if kind == "down":
                    return None, "ConnectionError: boom"
                if kind == "slow":
                    real_sleep(0.05)
                    return df0.copy(), ""
                return None, "Too Many Requests"
            return f

        tick = [f"T{i:04d}" for i in range(400)]
        data, status = download_lite(tick, fetcher=fetcher("mixed"), workers=8, budget_s=60, log_every=10 ** 9)
        gone = sum(1 for t in tick if sum(map(ord, t)) % 17 == 0)
        rep("Yuklash: aralash holat (cheklov, yopilgan tiker, uzilish, tasodifiy xato) — hech biri yoʻqolmaydi",
            len(data) + len(status) == len(tick) and sum(1 for v in status.values() if v == "gone") == gone and len(data) > 0.6 * len(tick),
            f"olindi {len(data)}, holatlar {dict(Counter(status.values()))}")
        calls.clear()
        t0 = time.time()
        data, status = download_lite(tick, fetcher=fetcher("down"), workers=8, budget_s=60, log_every=10 ** 9)
        rep("Yuklash: internet butunlay yoʻq boʻlsa tez toʻxtaydi (soatlab osilib qolmaydi)", len(data) == 0 and time.time() - t0 < 30 and sum(calls.values()) < 2 * len(tick),
            f"{sum(calls.values())} ta soʻrovdan keyin toʻxtadi")
        calls.clear()
        data, status = download_lite(tick[:120], fetcher=fetcher("rate"), workers=8, budget_s=30, log_every=10 ** 9)
        rep("Yuklash: hamma soʻrov cheklansa — uzoq kutmasdan tugaydi", len(data) == 0, f"holatlar {dict(Counter(status.values()))}")
        data, status = download_lite(tick, fetcher=fetcher("slow"), workers=4, budget_s=1.0, log_every=10 ** 9)
        rep("Yuklash: vaqt byudjeti ushlab turiladi", 0 < len(data) < len(tick) and set(status.values()) <= {"time"}, f"1 soniyada {len(data)} ta olindi, qolgani 'vaqt tugadi'")
    finally:
        time.sleep = real_sleep
    # 3. sintetik sayt: toʻliq + yengil sahifalar, qidiruv/skaner maʼlumoti
    import tempfile
    out = tempfile.mkdtemp(prefix="aksiya_st_")
    try:
        code = build(out, synthetic=True, extra_env="", max_tickers=0, synth_n=40, lite=True, lite_max=0, lite_synth_n=24)
        need = ["index.html", "screener.html", "tickers.js", "screener.js", "data.js", "site.js", "style.css", "404.html"]
        have = [f for f in need if os.path.exists(os.path.join(out, f))]
        pages = len([f for f in os.listdir(os.path.join(out, "t"))]) if os.path.isdir(os.path.join(out, "t")) else 0
        rep("Sintetik sayt qurilishi (toʻliq + yengil sahifalar, qidiruv va skaner)", code == 0 and len(have) == len(need) and pages >= 30,
            f"chiqish kodi {code}, sahifalar {pages}, fayllar {len(have)}/{len(need)}")
        # ilova rejimi (PWA): manifest, belgilar, service worker, har sahifada ulanish, «Ilova» sahifasi
        try:
            from PIL import Image
            mf = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
            icons_ok = all(os.path.exists(os.path.join(out, i["src"])) for i in mf["icons"])
            for i in mf["icons"]:
                if i["src"].endswith(".png"):
                    icons_ok &= Image.open(os.path.join(out, i["src"])).size == tuple(int(x) for x in i["sizes"].split("x"))
            htmls = [os.path.join(out, f) for f in os.listdir(out) if f.endswith(".html")] + [os.path.join(out, "t", f) for f in os.listdir(os.path.join(out, "t"))]
            linked = sum(1 for f in htmls if 'rel="manifest"' in open(f, encoding="utf-8").read() and "apple-touch-icon" in open(f, encoding="utf-8").read())
            sw = open(os.path.join(out, "sw.js"), encoding="utf-8").read()
            inst = open(os.path.join(out, "install.html"), encoding="utf-8").read()
            rep("Ilova rejimi (PWA): manifest, belgilar (oʻlchami), service worker, har sahifada ulanish, «Ilova» sahifasi",
                mf.get("display") == "standalone" and mf.get("start_url") == "./index.html" and mf.get("scope") == "./" and icons_ok and linked == len(htmls)
                and "__STAMP__" not in sw and "fetch" in sw and 'id="pwa-install"' in inst and "📲 Ilova" in inst,
                f"{len(mf['icons'])} belgi, {linked}/{len(htmls)} sahifa manifest bilan ulangan")
        except Exception as e:                                       # noqa: BLE001
            rep("Ilova rejimi (PWA)", False, f"{type(e).__name__}: {e}")
        # Pillow ishlamasa — SVG zaxira, sayt baribir qurilaveradi
        global _icon_png
        orig_icon, tmp_icons = _icon_png, tempfile.mkdtemp(prefix="aksiya_ic_")
        try:
            def _boom(*a: Any, **k: Any) -> Any:
                raise RuntimeError("Pillow yoʻq (sinov)")
            _icon_png = _boom
            ic = make_icons(tmp_icons)
            mfz = pwa_manifest(ic)
            rep("PWA belgilar: Pillow ishlamasa SVG zaxira ishlatiladi (manifest yaroqli)", ic[0]["src"].endswith(".svg") and os.path.exists(os.path.join(tmp_icons, "icon.svg"))
                and mfz["icons"] == ic and mfz["shortcuts"], f"{ic[0]['src']}")
        finally:
            _icon_png = orig_icon
            shutil_rm = __import__("shutil").rmtree
            shutil_rm(tmp_icons, ignore_errors=True)
    except Exception as e:                                          # noqa: BLE001
        rep("Sintetik sayt qurilishi", False, f"{type(e).__name__}: {e}")
    finally:
        import shutil
        shutil.rmtree(out, ignore_errors=True)
    return ok_all


def main() -> None:
    ap = argparse.ArgumentParser(description="Aksiya AI statik sayt quruvchi v%s" % app.APP_VERSION)
    ap.add_argument("--out", default="site")
    ap.add_argument("--synthetic", action="store_true", help="internetsiz sinov (sintetik bozor)")
    ap.add_argument("--extra", default="", help="qoʻshimcha tikerlar (vergul bilan); EXTRA_TICKERS muhit oʻzgaruvchisi ham oʻqiladi")
    ap.add_argument("--max-tickers", type=int, default=0, help="sinov uchun: faqat birinchi N ta sahifa (yengil bosqich oʻtkazib yuboriladi)")
    ap.add_argument("--synth-n", type=int, default=60)
    ap.add_argument("--no-lite", action="store_true", help="yengil bosqichni (barcha NYSE/NASDAQ aksiyalari) oʻtkazib yuborish")
    ap.add_argument("--lite-max", type=int, default=0, help="sinov uchun: yengil sahifalar soni chegarasi")
    ap.add_argument("--lite-synth-n", type=int, default=0, help="sintetik rejimda yengil sahifalar soni (0 — oʻtkazib yuboriladi)")
    ap.add_argument("--save-extras", default=None, metavar="TIKERLAR", help="tikerlarni extra_tickers.txt ga qoʻshadi va chiqadi")
    ap.add_argument("--selftest", action="store_true", help="yengil rejim mantiqini internetsiz tekshirish")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.save_extras is not None:
        added = save_extras(a.save_extras)
        print("Qoʻshildi:", ", ".join(added) if added else "— (yangi tiker yoʻq yoki limit toʻldi)")
        return
    env_txt = " ".join(x for x in (a.extra, os.environ.get("EXTRA_TICKERS", "")) if x)
    sys.exit(build(a.out, synthetic=a.synthetic, extra_env=env_txt, max_tickers=a.max_tickers, synth_n=a.synth_n,
                   lite=not a.no_lite, lite_max=a.lite_max, lite_synth_n=a.lite_synth_n))


if __name__ == "__main__":
    main()
