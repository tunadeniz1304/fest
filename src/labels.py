# -*- coding: utf-8 -*-
"""
Sartname ve FTR teslim dokumaninda tanimlanan TUM gecerli etiketler ve sema.

KRITIK KURALLAR (teslim dokumani Bolum 2-5):
  - Tum etiketler ASCII-safe ve KUCUK harf (Turkce karakter YASAK).
  - JSON anahtarlari birebir: "confidence_score" (asla "score" / "guven_skoru").
  - Plaka birlesik format (34ABC123) veya regex'e uygun normalize.

Bu dosya tek dogruluk kaynagidir; pipeline ve validator buradan okur.
"""
import re

# --- Arac bilgisi alanlari (Tablo 1) ---
ARAC_TIPLERI = ["sedan", "suv", "hatchback", "pickup", "minibus", "panelvan", "kamyon"]
ARAC_RENKLERI = ["beyaz", "siyah", "gri", "kirmizi", "mavi", "sari", "yesil", "turuncu", "kahverengi"]

# --- Tespit kategorileri ve gecerli etiketler (Tablo 2) ---
# NOT: Cikti ornek JSON'da kategori adi "sofor_eylemi" olarak gecer (Bolum 4.1).
KATEGORI_ETIKETLERI = {
    "sofor_eylemi": [
        "arkaya_bakma", "esneme", "sigara_icme", "su_icme",
        "telefonla_konusma", "slalom", "etrafa_bakinma", "emniyet_kemeri_ihlali",
    ],
    "nesneler": ["teknocan", "bilgisayar"],
    "yolcular": ["arka_koltuk_1", "arka_koltuk_2", "on_koltuk"],
}

GECERLI_KATEGORILER = set(KATEGORI_ETIKETLERI.keys())

# --- Plaka regex (teslim dokumani Tablo 1) ---
# Birlesik (bosluksuz) plaka kontrolu icin kullanilir.
PLAKA_REGEX = re.compile(
    r"^(0[1-9]|[1-7][0-9]|8[01])"
    r"((\s?[A-Z]\s?)(\d{4,5})"
    r"|(\s?[A-Z]{2}\s?)(\d{3,4})"
    r"|(\s?[A-Z]{3}\s?)(\d{2,3}))$"
)

# OCR sonrasi gevsek temizlik icin: TR plakada gecmeyen harfler (Q, W, X yok)
GECERSIZ_PLAKA_HARFLERI = set("QWX")


def plaka_gecerli_mi(plaka):
    """Birlesik plaka stringi TR regex'ine uyuyor mu?"""
    if not plaka:
        return False
    return bool(PLAKA_REGEX.match(plaka.strip()))


def etiket_gecerli_mi(kategori, etiket):
    return kategori in KATEGORI_ETIKETLERI and etiket in KATEGORI_ETIKETLERI[kategori]


def validate_results(data):
    """
    results.json icerigini sartnameye gore programatik dogrular.
    Hata listesi dondurur (bos liste = gecerli). Teslimden once main.py cagirir.
    """
    errors = []

    if not isinstance(data, dict):
        return ["Kok nesne dict olmali."]

    # video_id
    if "video_id" not in data or not isinstance(data["video_id"], str):
        errors.append("video_id eksik veya string degil.")

    # arac_bilgisi
    ab = data.get("arac_bilgisi")
    if not isinstance(ab, dict):
        errors.append("arac_bilgisi eksik veya dict degil.")
    else:
        if ab.get("tip") not in ARAC_TIPLERI + [None, ""]:
            errors.append(f"arac_bilgisi.tip gecersiz: {ab.get('tip')}")
        if ab.get("renk") not in ARAC_RENKLERI + [None, ""]:
            errors.append(f"arac_bilgisi.renk gecersiz: {ab.get('renk')}")
        cs = ab.get("confidence_score")
        if not isinstance(cs, (int, float)) or not (0.0 <= float(cs) <= 1.0):
            errors.append(f"arac_bilgisi.confidence_score 0-1 araliginda float olmali: {cs}")
        # plaka bos veya regex'e uygun olmali (tespit edilemediyse bos string)
        plaka = ab.get("plaka", "")
        if plaka and not plaka_gecerli_mi(plaka):
            errors.append(f"arac_bilgisi.plaka regex'e uymuyor: {plaka}")

    # tespitler
    tespitler = data.get("tespitler")
    if not isinstance(tespitler, list):
        errors.append("tespitler eksik veya list degil.")
    else:
        for i, t in enumerate(tespitler):
            if not isinstance(t, dict):
                errors.append(f"tespitler[{i}] dict degil.")
                continue
            if not isinstance(t.get("zaman_saniye"), (int, float)):
                errors.append(f"tespitler[{i}].zaman_saniye float olmali.")
            kat = t.get("kategori")
            etk = t.get("etiket")
            if kat not in GECERLI_KATEGORILER:
                errors.append(f"tespitler[{i}].kategori gecersiz: {kat}")
            elif not etiket_gecerli_mi(kat, etk):
                errors.append(f"tespitler[{i}].etiket '{etk}' kategori '{kat}' icin gecersiz.")
            cs = t.get("confidence_score")
            if not isinstance(cs, (int, float)) or not (0.0 <= float(cs) <= 1.0):
                errors.append(f"tespitler[{i}].confidence_score 0-1 float olmali: {cs}")

    return errors
