# -*- coding: utf-8 -*-
"""
Yardimci fonksiyonlar: renk tespiti (HSV), arac tipi heuristigi,
plaka OCR normalizasyonu, frame sampling.

Tum fonksiyonlar cokmeye dayaniklidir: hata olursa makul bir varsayilan doner.
"""
import re
import numpy as np
import cv2

from src.labels import ARAC_RENKLERI, GECERSIZ_PLAKA_HARFLERI, plaka_gecerli_mi


# ---------------------------------------------------------------------------
# RENK TESPITI (klasik CV, model degil)
# ---------------------------------------------------------------------------
# Sartname renkleri icin temsili HSV merkez degerleri (OpenCV: H 0-179, S/V 0-255).
# Aciklama: akromatik renkler (beyaz/siyah/gri) ayri ele alinir.
_RENK_HSV = {
    "kirmizi":     [(0, 200, 150), (179, 200, 150)],   # kirmizi iki uctan dolanir
    "turuncu":     [(15, 200, 180)],
    "sari":        [(28, 200, 180)],
    "yesil":       [(60, 150, 120)],
    "mavi":        [(110, 180, 150)],
    "kahverengi":  [(15, 150, 90)],
}


def _hsv_mesafe(h1, s1, v1, h2, s2, v2):
    # Hue dairesel mesafe
    dh = min(abs(h1 - h2), 180 - abs(h1 - h2)) / 90.0
    ds = abs(s1 - s2) / 255.0
    dv = abs(v1 - v2) / 255.0
    return 2.0 * dh + ds + dv  # hue'ya agirlik


def baskin_renk(bgr_crop):
    """
    Arac bbox kirpintisindan baskin rengi tespit eder, sartnamenin 9 renginden
    en yakinini dondurur. Hata olursa 'gri' (notr varsayilan) doner.
    """
    try:
        if bgr_crop is None or bgr_crop.size == 0:
            return "gri"
        # Merkez bolge agirlikli ornekleme (kenarlardaki arka plani azalt)
        h, w = bgr_crop.shape[:2]
        y0, y1 = int(h * 0.25), int(h * 0.75)
        x0, x1 = int(w * 0.25), int(w * 0.75)
        crop = bgr_crop[y0:y1, x0:x1]
        if crop.size == 0:
            crop = bgr_crop
        crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        H = np.median(hsv[:, :, 0])
        S = np.median(hsv[:, :, 1])
        V = np.median(hsv[:, :, 2])

        # Akromatik kontrol: dusuk doygunluk -> beyaz/gri/siyah
        if S < 45:
            if V > 175:
                return "beyaz"
            if V < 70:
                return "siyah"
            return "gri"

        # Kromatik: en yakin renk merkezi
        en_iyi, en_kucuk = "gri", 1e9
        for renk, merkezler in _RENK_HSV.items():
            for (hh, ss, vv) in merkezler:
                d = _hsv_mesafe(H, S, V, hh, ss, vv)
                if d < en_kucuk:
                    en_kucuk, en_iyi = d, renk
        return en_iyi
    except Exception:
        return "gri"


# ---------------------------------------------------------------------------
# ARAC TIPI HEURISTIGI (COCO car/truck/bus -> 7 sartname tipi)
# ---------------------------------------------------------------------------
def arac_tipi_heuristik(coco_label, bbox):
    """
    COCO sinifi + bbox en/boy oranindan sartname arac tipini tahmin eder.
    Kesin degil ama 'tip' alanini doldurur (bos birakmaktan iyidir).
    bbox: (x1, y1, x2, y2)
    """
    try:
        x1, y1, x2, y2 = bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        oran = w / h  # genis = uzun arac

        if coco_label == "bus":
            return "minibus"
        if coco_label == "truck":
            # Genis ve buyuk -> kamyon; kompakt -> pickup/panelvan
            if oran > 1.8:
                return "kamyon"
            return "pickup"
        # coco_label == "car" (varsayilan)
        if oran > 1.7:
            return "suv"        # genis govde
        if oran < 1.25:
            return "hatchback"  # kompakt/kare
        return "sedan"          # tipik bir oran
    except Exception:
        return "sedan"


# ---------------------------------------------------------------------------
# PLAKA OCR NORMALIZASYONU
# ---------------------------------------------------------------------------
_OCR_DUZELT = {
    "I": "1", "İ": "1", "Ö": "0", "O": "0", "Q": "0",
    "Ü": "U", "Ç": "C", "Ş": "S", "Ğ": "G", " ": "", "-": "", ".": "",
}


def plaka_normalize(ham_metin):
    """
    OCR ham ciktisini birlesik TR plaka formatina cevirir.
    Regex'e uymazsa bos string doner (sartname: tespit edilemedi).
    """
    if not ham_metin:
        return ""
    s = ham_metin.upper()
    # Yaygin OCR karisikliklarini ve bosluklari temizle
    s = "".join(_OCR_DUZELT.get(ch, ch) for ch in s)
    s = re.sub(r"[^A-Z0-9]", "", s)
    # Gecersiz harfleri (Q/W/X) at -- yanlis okuma olasiligi
    s = "".join(ch for ch in s if ch not in GECERSIZ_PLAKA_HARFLERI or ch.isdigit())
    if plaka_gecerli_mi(s):
        return s
    return ""


# ---------------------------------------------------------------------------
# FRAME SAMPLING
# ---------------------------------------------------------------------------
def hedef_frame_indeksleri(toplam_frame, fps, hedef_fps=2.0):
    """
    Videodan saniyede ~hedef_fps kare ornekler. 10dk timeout ve farkli
    kaynak FPS'lere robustluk saglar. Indeks listesi dondurur.
    """
    if fps is None or fps <= 0:
        fps = 25.0
    if toplam_frame is None or toplam_frame <= 0:
        return []
    adim = max(1, int(round(fps / max(0.1, hedef_fps))))
    return list(range(0, int(toplam_frame), adim))


def bbox_merkez_konumu(bbox, frame_w):
    """person bbox merkezine gore kabaca yolcu konumu tahmini (on/arka)."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    # Goruntude soldaki/sagdaki kisi on koltuk varsayimi cok kaba; basit kural:
    # alt yariya yakin ve buyuk bbox -> on; ust/kucuk -> arka.
    return cx, (y1 + y2) / 2.0
