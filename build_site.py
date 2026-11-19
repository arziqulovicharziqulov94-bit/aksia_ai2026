# -*- coding: utf-8 -*-
"""
AKSIYA AI — statik sayt quruvchi (GitHub Pages uchun: bepul va doim ochiq)
==========================================================================
Dvigatel (app.py) bilan bir marta hisoblab, har bir aksiya uchun tayyor HTML sahifa yozadi.
Sayt serversiz ishlaydi: telefon va kompyuter brauzerida darhol ochiladi; GitHub Actions uni har kuni yangilaydi.

    python build_site.py --out site                      # real maʼlumot (Yahoo; zaxira Stooq)
    python build_site.py --out site --synthetic          # internetsiz sinov (sintetik bozor)
    python build_site.py --save-extras "PLTR, SOFI"      # tikerlarni extra_tickers.txt ga qoʻshish

Cheklov (halol): sayt faqat OLDINDAN tayyorlangan aksiyalarni koʻrsatadi (universum + extra_tickers.txt).
Yangi tiker qoʻshish — GitHub Actions orqali (5–20 daqiqa) yoki Colab daftarida (cheksiz tiker).
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
from typing import Any, Dict, List, Optional, Tuple

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
table.sortable th.s{cursor:pointer;user-select:none}
table.sortable th.s:after{content:" ↕";opacity:.35}
table.sortable th.s.asc:after{content:" ▲";opacity:1}
table.sortable th.s.desc:after{content:" ▼";opacity:1}
.flt{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
.flt input,.flt select{padding:8px 10px;border:1px solid #c9c8c0;border-radius:8px;font-size:16px;background:#fff;min-width:0}
.flt input{flex:1;min-width:150px}
.calcrow{display:flex;gap:8px;margin:8px 0}
.calcrow input{padding:9px 11px;border:1px solid #c9c8c0;border-radius:8px;font-size:16px;flex:1;min-width:0;max-width:260px}
.ai a.tl{font-weight:700;text-decoration:none}
@media (max-width:640px){.top{position:static}.srch{margin:0;width:100%}.srch input{flex:1;width:auto}.nav a{padding:5px 8px;font-size:.84rem}.wrapp{padding:4px 8px 26px}}
@media print{.top,footer,.noprint,#srchmsg{display:none!important}}
</style>"""

SITE_JS = r"""(function(){
var A=window.AKSIYA||{tickers:[],repo:""};
function norm(s){return (s||'').toUpperCase().replace(/\s+/g,'').replace(/\./g,'-').replace(/[^A-Z0-9\-\^=]/g,'');}
function base(){var s=document.currentScript&&document.currentScript.src||'';return s.replace(/site\.js(\?.*)?$/,'');}
var B=base();
window.goT=function(f){
  var T=norm(f.q.value), box=document.getElementById('srchmsg');
  if(!T){return false;}
  if(A.tickers.indexOf(T)>=0){location.href=B+'t/'+T+'.html';return false;}
  var wf=A.repo?('https://github.com/'+A.repo+'/actions/workflows/build-site.yml'):'';
  var msg='<div class="card"><b>'+T+'</b> tayyor tahlillar roʻyxatida yoʻq. Bu sayt faqat oldindan tayyorlangan '+A.tickers.length+' ta aksiyani koʻrsatadi.<br>'
   +'<b>Qoʻshish:</b> GitHub → Actions → «Saytni qurish va joylash» → <i>Run workflow</i> → tikerlar maydoniga <b>'+T+'</b> yozing '
   +'(≈10–20 daqiqadan keyin saytda paydo boʻladi va keyingi kunlarda ham yangilanib turadi)'
   +(wf?' — <a href="'+wf+'" target="_blank" rel="noopener">shu yerda</a>':'')
   +'. Yoki cheksiz tiker uchun Colab daftaridan foydalaning (README).</div>';
  if(box){box.innerHTML=msg;}
  return false;
};
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
})();
"""


def search_form() -> str:
    return ('<form class="srch noprint" onsubmit="return goT(this)" autocomplete="off">'
            '<input name="q" placeholder="Tiker: AAPL, NVDA, AMD…" list="tl" autocapitalize="characters" '
            'autocorrect="off" spellcheck="false" aria-label="Tiker"><button type="submit">Tahlil</button></form>')


def page(title: str, body: str, active: str, depth: int = 0, extra_js: str = "") -> str:
    p = "../" * depth
    tabs = [("index", "🏆 Tavsiyalar"), ("stocks", "🔍 Aksiyalar"), ("quality", "📊 Sifat"), ("method", "ℹ️ Usul")]
    nav = "".join(f'<a href="{p}{k}.html"{" class=on" if k == active else ""}>{lab}</a>' for k, lab in tabs)
    foot = (f'<footer>Aksiya AI v{app.APP_VERSION} · sayt qurilgan: {META["built_local"]} (Toshkent vaqti) · narx maʼlumoti sanasi: '
            f'{META["asof"]} · manba: Yahoo Finance. Bu — taʼlim va tadqiqot vositasi, moliyaviy maslahat EMAS. '
            f'Sahifa sayt qurilgandagi holatni koʻrsatadi (real vaqtda yangilanmaydi).</footer>')
    return ('<!doctype html><html lang="uz"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta name="color-scheme" content="light"><meta name="robots" content="noindex,nofollow">'
            f'<title>{esc(title)}</title>{CSS}{PAGE_CSS}</head><body>'
            f'<header class="top"><div class="in"><a class="brand" href="{p}index.html">📈 Aksiya AI</a>'
            f'<nav class="nav">{nav}</nav>{search_form()}</div><div id="srchmsg"></div></header>'
            f'{body}{foot}<script src="{p}data.js"></script>{extra_js}<script src="{p}site.js"></script></body></html>')


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


STATIC_NOTE_MD = """
## Bu sayt qanday ishlaydi (statik variant)

* Sayt **serversiz**: har bir aksiya uchun sahifa oldindan hisoblab qoʻyilgan, shuning uchun telefon va kompyuterda **darhol** ochiladi va doim ishlaydi (GitHub Pages).
* GitHub Actions saytni **har ish kuni** AQSh bozori yopilgandan keyin qaytadan quradi: maʼlumot yangilanadi, model qayta oʻqitiladi va halol tekshiruvdan oʻtkaziladi.
* Sahifada koʻrinadigan narx — **sayt qurilgan paytdagi** yopilish narxi (real vaqt emas).
* **Cheklov:** faqat oldindan tayyorlangan aksiyalar koʻrinadi (universum + `extra_tickers.txt`). Roʻyxatda yoʻq tikerni qoʻshish uchun GitHub → Actions → *Run workflow* → tikerni yozing (≈10–20 daqiqa). Cheksiz, istalgan tiker uchun Colab daftaridan foydalaning.
* **Maqsad narx** kalkulyatori aksiya sahifasida brauzerning oʻzida ishlaydi (Monte-Karlo ehtimollar jadvalidan interpolyatsiya; aniqlik ≈ ±0.2 foiz punkti).
"""


# ======================================================================================
# 4. BITTA AKSIYA SAHIFASI
# ======================================================================================
def calc_card(R: Dict[str, Any]) -> Tuple[str, str]:
    G = app.mc_barrier_grid(R["df"])
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
    body = (f'<div class="wrapp"><div class="ai"><h2>Barcha aksiyalar ({len(rows)} ta)</h2>'
            f'<p class="small muted">Tikerni bosing — toʻliq tahlil. Sarlavhani bosib saralash mumkin. “Umumiy xulosa” — <b>tavsiya emas</b>.</p>'
            f'<details class="small muted"><summary>Ustunlar izohi</summary><p>Toʻliq tahlilda: grafik, Monte-Karlo, support/resistance, Wyckoff, baza va breakout maqsadlari, '
            f'Gann, sham, fundamental, yangilik, maqsad narx kalkulyatori. “Umumiy xulosa” — klassik qoidalar va (agar sinovdan oʻtgan boʻlsa) AI aralashmasi. {esc(ai_note)}</p></details>'
            f'<div class="flt"><input id="flt-q" placeholder="Qidirish: tiker yoki kompaniya nomi" aria-label="Qidirish">'
            f'<select id="flt-l"><option value="">Hammasi</option><option>Ijobiy</option><option>Neytral</option><option>Salbiy</option></select>'
            f'<span class="small muted" style="align-self:center">koʻrsatilmoqda: <b id="flt-n">{len(rows)} ta</b></span></div>'
            f'<div class="wrap"><table class="sortable">{head}<tbody>{"".join(trs)}</tbody></table></div>{fl}'
            f'<p class="small muted">Roʻyxatda yoʻq tikerni qoʻshish: GitHub → Actions → «Saytni qurish va joylash» → Run workflow (tikerlar maydoni). '
            f'Qoʻshilgan tikerlar extra_tickers.txt da saqlanadi va har kuni yangilanadi.</p>{DISCLAIMER}</div></div>')
    return page("Barcha aksiyalar — Aksiya AI", body, "stocks")


def index_page(st: app.State, built: Optional[set] = None) -> str:
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
    return page("Aksiya AI — tavsiyalar", f'<div class="wrapp">{html}</div>', "index")


def quality_page(st: app.State) -> str:
    return page("Model sifati — Aksiya AI", f'<div class="wrapp">{app.render_quality(st).replace(CSS, "", 1)}</div>', "quality")


def method_page() -> str:
    md = app.METHODOLOGY_MD.replace("Tizim har ~12 soatda maʼlumotni yangilab, modelni qaytadan oʻqitadi va tekshiradi.",
                                    "Sayt har ish kuni maʼlumotni yangilab, modelni qaytadan oʻqitadi va tekshiradi.")
    body = (f'<div class="wrapp"><div class="ai">{md_to_html(md)}{md_to_html(STATIC_NOTE_MD)}{DISCLAIMER}</div></div>')
    return page("Usul — Aksiya AI", body, "method")


def not_found_page() -> str:
    body = ('<div class="wrapp"><div class="ai"><div class="card warn"><b>Sahifa topilmadi</b><p>Bu manzil sayt tarkibida yoʻq. '
            'Agar siz aksiya sahifasini qidirgan boʻlsangiz, u hali tayyorlanmagan boʻlishi mumkin: yuqoridagi qidiruvga tikerni yozing — '
            'tayyor boʻlmasa, qoʻshish yoʻli koʻrsatiladi.</p></div></div></div>')
    return page("Topilmadi — Aksiya AI", body, "stocks")


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


def build(out: str, synthetic: bool = False, extra_env: str = "", max_tickers: int = 0, synth_n: int = 60) -> int:
    t_all = time.time()
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

    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "t"))
    rows: List[Dict[str, Any]] = []
    failed: List[Tuple[str, str]] = []
    t0 = time.time()
    for i, t in enumerate(todo, 1):
        try:
            R = app.analyze_ticker(t, st, None, use_cache=False)
            with open(os.path.join(out, "t", f"{t}.html"), "w", encoding="utf-8") as f:
                f.write(ticker_page(R, st))
            rows.append(summarize(R, st))
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
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    with open(os.path.join(out, "data.js"), "w", encoding="utf-8") as f:
        f.write("window.AKSIYA=" + json.dumps(dict(tickers=names, repo=repo, built=META["built_local"]), separators=(",", ":")) + ";")
    with open(os.path.join(out, "site.js"), "w", encoding="utf-8") as f:
        f.write(SITE_JS)
    open(os.path.join(out, ".nojekyll"), "w").close()
    writes = {"index.html": index_page(st, built=set(names)), "stocks.html": stocks_page(rows, failed, st),
              "quality.html": quality_page(st), "method.html": method_page(), "404.html": not_found_page()}
    for fn, html in writes.items():
        with open(os.path.join(out, fn), "w", encoding="utf-8") as f:
            f.write(html)
    ev = st.evals[st.primary_h]
    meta = dict(version=app.APP_VERSION, built_utc=now.isoformat(timespec="seconds"), asof=META["asof"], pages=len(rows),
                universe=len(uni), extras=extras, failed=failed, primary_horizon=st.primary_h, primary_verdict=ev.verdict,
                verdicts={str(h): e.verdict for h, e in sorted(st.evals.items())}, synthetic=synthetic,
                build_minutes=round((time.time() - t_all) / 60, 1))
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    size = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(out) for fn in fns) / 1e6
    app.log(f"TAYYOR: {len(rows)} ta sahifa, {size:.1f} MB, {meta['build_minutes']} daqiqa; asosiy ufq hukmi: {ev.verdict}")
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ and not synthetic:
        try:
            with open(summ, "a", encoding="utf-8") as f:
                f.write(f"### Sayt qurildi\n\n- Sahifalar: **{len(rows)}** ta ({size:.0f} MB), maʼlumot sanasi **{META['asof']}**\n"
                        f"- Model hukmi ({HORIZON_UZ.get(st.primary_h)}): **{ev.verdict}**; ufqlar: {meta['verdicts']}\n"
                        + (f"- Tayyorlanmadi: {failed}\n" if failed else ""))
        except Exception:
            pass
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Aksiya AI statik sayt quruvchi v%s" % app.APP_VERSION)
    ap.add_argument("--out", default="site")
    ap.add_argument("--synthetic", action="store_true", help="internetsiz sinov (sintetik bozor)")
    ap.add_argument("--extra", default="", help="qoʻshimcha tikerlar (vergul bilan); EXTRA_TICKERS muhit oʻzgaruvchisi ham oʻqiladi")
    ap.add_argument("--max-tickers", type=int, default=0, help="sinov uchun: faqat birinchi N ta sahifa")
    ap.add_argument("--synth-n", type=int, default=60)
    ap.add_argument("--save-extras", default=None, metavar="TIKERLAR", help="tikerlarni extra_tickers.txt ga qoʻshadi va chiqadi")
    a = ap.parse_args()
    if a.save_extras is not None:
        added = save_extras(a.save_extras)
        print("Qoʻshildi:", ", ".join(added) if added else "— (yangi tiker yoʻq yoki limit toʻldi)")
        return
    env_txt = " ".join(x for x in (a.extra, os.environ.get("EXTRA_TICKERS", "")) if x)
    sys.exit(build(a.out, synthetic=a.synthetic, extra_env=env_txt, max_tickers=a.max_tickers, synth_n=a.synth_n))


if __name__ == "__main__":
    main()
