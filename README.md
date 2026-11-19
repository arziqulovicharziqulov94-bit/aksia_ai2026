# 📈 Aksiya AI — bepul, doim ochiq sayt (GitHub Pages)

AQSh aksiyalarini (NYSE/NASDAQ) tahlil qiluvchi va **o'z modelini o'zi halol sinaydigan** tizim. Sayt **serversiz**:
har kuni avtomatik quriladi va GitHub Pages'da turadi — telefon va kompyuterda **darhol** ochiladi, hech narsani ishga tushirish shart emas.
**Narxi: 0 so'm** (GitHub bepul hisobi yetarli).

> ⚠️ Bu — ta'lim va tadqiqot vositasi, **moliyaviy maslahat emas**. Hech qanday model kelajakni kafolatlamaydi.
> Sayt "Sifat" bo'limida modelning tarixiy ustunligi **isbotlanganmi yoki yo'qmi** — shuni ochiq ko'rsatadi.

## Saytda nima bor

| Bo'lim | Nima qiladi |
|---|---|
| 🏆 **Tavsiyalar** | Bozorni o'zi skaner qiladi (~170 yirik aksiya), mezonni siz tanlamaysiz: eng istiqbolli nomzodlarni va birinchi nomzodning **chuqur tahlilini** ko'rsatadi |
| 🔍 **Aksiyalar** | Barcha tayyor aksiyalar jadvali (saralash, qidirish, filtr). Tikerni bossangiz — to'liq tahlil sahifasi |
| 📊 **Sifat** | Model haqiqatan ustunlik beryaptimi: halol (purged walk-forward) sinov, backtest, kalibrovka |
| ℹ️ **Usul** | Hamma narsa qanday hisoblanishi va cheklovlar |

**Har bir aksiya sahifasida:** narx grafigi (shamlar, SMA, support/resistance, Gann, Wyckoff Spring/Upthrust, **baza va maqsad darajalari**),
ko'p ufqli prognoz (5 kundan 6 oygacha) va Monte-Karlo narx diapazoni, **maqsad narxga yetish ehtimoli kalkulyatori** (istalgan narxni yozing),
baza (konsolidatsiya) + breakout va "measured move" maqsadlari, Wyckoff, sham modellari, fundamental ko'rsatkichlar, yangilik kayfiyati,
bitta umumiy xulosa va **yuklab olish** tugmasi (HTML fayl).

## Bir martalik o'rnatish (≈15 daqiqa, kompyuterdan qulayroq)

1. [github.com](https://github.com) da bepul ro'yxatdan o'ting.
2. **New repository** → nomi `aksiya-ai` → **Public** (⚠️ bepul hisobda GitHub Pages faqat ochiq repozitoriy uchun ishlaydi) → **Create repository**.
3. Shu zip'ni oching. Repozitoriy sahifasida **uploading an existing file** → papka ichidagi **hamma fayllarni** (`.github` papkasi bilan!) oynaga sudrab tashlang → **Commit changes**.
   *Tekshirish:* repozitoriyda `.github/workflows/build-site.yml` fayli ko'rinishi kerak. Ko'rinmasa: **Add file → Create new file** → nomga `.github/workflows/build-site.yml` yozing, fayl matnini joylang → Commit.
4. **Settings → Pages → Build and deployment → Source:** **GitHub Actions** ni tanlang.
5. **Actions** yorlig'i → (so'rasa: *I understand my workflows, go ahead and enable them*) → chapdan **"Saytni qurish va joylash"** → **Run workflow**.
6. Taxminan 10–25 daqiqadan keyin ish yashil ✓ bo'ladi *(bu — taxmin, o'lchanmagan: real internetdagi birinchi ishga tushirishni men tekshira olmadim)*.
   Sayt manzili: `https://<GitHub-loginingiz>.github.io/aksiya-ai/` (deploy qadamida ham ko'rinadi).
7. Telefonda: manzilni oching → brauzer menyusi → **Bosh ekranga qo'shish**. Endi u ilova kabi ochiladi.

## Kundalik ishlatish

* **Yangilanish:** sayt har ish kuni AQSh bozori yopilgandan keyin (22:30 UTC = Toshkent vaqti bilan ertasi kuni 03:30) o'zi qayta quriladi. Sahifa quyi qismida "sayt qurilgan" vaqti va narx ma'lumoti sanasi ko'rinadi. (GitHub jadvalli ishlarni ba'zan kechiktiradi.)
* **Yangi aksiya qo'shish (telefondan ham):** Actions → "Saytni qurish va joylash" → **Run workflow** → *tickers* maydoniga yozing (masalan `PLTR, SOFI`). Ular `extra_tickers.txt` ga saqlanadi va keyingi kunlarda ham yangilanib turadi.
  Yoki shu faylni GitHub'ning o'zida tahrirlang (✏️ → Commit) — sayt avtomatik qayta quriladi. Eng ko'pi 60 ta qo'shimcha tiker.
* **Tayyor bo'lmagan tikerni qidirsangiz**, sayt buni aytadi va qanday qo'shishni ko'rsatadi.
* ⚠️ **60 kun qoidasi:** GitHub ochiq repozitoriyda 60 kun hech qanday faollik bo'lmasa, jadvalli ishlarni o'zi o'chirib qo'yadi
  (GitHub hujjatlari). Bunday bo'lsa: Actions → workflow'ni qayta yoqing (*Enable*) yoki istalgan kichik o'zgarish kiriting.

## Cheksiz tiker kerakmi? — Colab

Bu sayt faqat **oldindan tayyorlangan** aksiyalarni ko'rsatadi (statik sayt shunday ishlaydi). Istalgan tikerni shu zahoti tahlil qilish uchun
`aksiya-ai.ipynb` ni [Google Colab](https://colab.research.google.com) da oching (Runtime → Run all) — bepul, lekin har safar qo'lda ishga tushiriladi.
Repozitoriyga yuklagan bo'lsangiz: `https://colab.research.google.com/github/<login>/aksiya-ai/blob/main/aksiya-ai.ipynb`.
Doimiy, har doim yoqilgan va istalgan tikerni tahlil qiladigan server (Render Starter) pulli — alohida `aksiya-ai-render.zip` paketida.

## Muammo bo'lsa

Actions → so'nggi ish → **"Ma'lumot manbalarini tekshirish"** qadami jadval chiqaradi:

| Holat | Ma'nosi |
|---|---|
| Yahoo (ommaviy yuklash) ✅ | Hammasi joyida |
| Yahoo ❌, Stooq ✅ | Tizim zaxira manbaga o'tadi — ishlaydi, lekin sekinroq |
| Ikkalasi ❌ | Narx manbasi yo'q: ish to'xtaydi, **eski sayt joyida qoladi** (buzilmaydi). Keyinroq qayta ishga tushiring |
| Fundamental/yangilik ❌ | Ixtiyoriy: tahlil shu bo'limlarsiz ishlayveradi (sahifada "olinmadi" deb yoziladi) |

*Halol ogohlantirish:* Yahoo bulutli serverlarning IP manzillariga ba'zan cheklov qo'yadi. GitHub serverlarida bu qanchalik tez-tez bo'lishini men
oldindan tekshira olmadim (mening muhitimda Yahoo ham, Stooq ham bloklangan). Shuning uchun birinchi ishga tushirishda yuqoridagi jadvalni ko'ring.

Kompyuteringizda o'zingiz sinash: `pip install -r requirements.txt` → `python app.py --check-data` (manbalar), `python build_site.py --out site` (sayt; `--synthetic` — internetsiz sinov).

## Halol cheklovlar

* **Mukammal AI yo'q.** Bozor raqobatli; ochiq ma'lumotdagi ustunlik odatda juda kichik yoki yo'q. Sayt buni "Sifat" bo'limida ko'rsatadi, "dalil yo'q" bo'lsa AI bali umumiy xulosaga qo'shilmaydi.
* **Wyckoff, Gann, sham, baza/breakout va maqsad darajalari — mexanik qoidalar**, ularning bashorat kuchi bu yerda tekshirilmagan (faqat AI bloki sinovdan o'tadi).
  Signalsiz sintetik tasodifiy bozorda baza qoidasi oynalarning ~49% ida "baza", ~5% ida "hajm bilan breakout" topadi — baza topilishi o'z-o'zidan signal emas.
* **Real ma'lumotda** to'liq ishga tushirish (174 aksiya) men tomonimdan tekshirilmagan: bu muhitda Yahoo/Stooq bloklangan. Dasturiy tekshiruvlar (kelajakka qarash yo'qligi, halol validatsiya, hisoblash mexanikasi, sayt, qidiruv, kalkulyator, telefon va kompyuter ko'rinishi) sintetik ma'lumotda o'tgan.
* **Survivorship bias:** universum — bugungi yirik kompaniyalar. **Kichik/illikvid aksiyalar** (masalan SPSC) AI universumiga kirmaydi: ular uchun faqat klassik bloklar va Monte-Karlo ko'rsatiladi.
* Narx — sayt qurilgan paytdagi yopilish narxi (real vaqt emas).

## Fayllar

| Fayl | Vazifasi |
|---|---|
| `app.py` | Dvigatel: ma'lumot, ML va halol sinov, tahlil (v4.1) |
| `build_site.py` | Statik saytni quruvchi |
| `extra_tickers.txt` | Qo'shimcha tikerlar ro'yxati |
| `.github/workflows/build-site.yml` | Avtomatik qurish va joylash |
| `requirements.txt` | Kutubxonalar |
| `aksiya-ai.ipynb` | Colab daftari (istalgan tiker uchun) |
