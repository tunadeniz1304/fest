# -*- coding: utf-8 -*-
"""
Arac rengi - VCoR fine-tune YOLOv8m-cls (top1 %88, GUVENILIR skor).

Izole: weights/color.pt varsa egitilmis model, yoksa cagiran taraf HSV'ye duser.
VCoR 15 ingilizce sinif -> bizim 9 ASCII-safe Turkce renk.
"""
import os

_DOCKER_W = "/app/weights"
_LOCAL_W = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
WEIGHTS_DIR = _DOCKER_W if os.path.isdir(_DOCKER_W) else _LOCAL_W
WEIGHTS = os.path.join(WEIGHTS_DIR, "color.pt")

# VCoR -> bizim 9 renk (beyaz/siyah/gri/kirmizi/mavi/sari/yesil/turuncu/kahverengi)
VCOR_ESLEME = {
    "beige": "sari", "black": "siyah", "blue": "mavi", "brown": "kahverengi",
    "gold": "sari", "green": "yesil", "grey": "gri", "orange": "turuncu",
    "pink": "kirmizi", "purple": "mavi", "red": "kirmizi", "silver": "gri",
    "tan": "kahverengi", "white": "beyaz", "yellow": "sari",
}

_MODEL = None
_YUKLENDI = False


def _model_yukle():
    global _MODEL, _YUKLENDI
    if _YUKLENDI:
        return _MODEL
    _YUKLENDI = True
    if not os.path.exists(WEIGHTS):
        return None
    try:
        from ultralytics import YOLO
        _MODEL = YOLO(WEIGHTS)
    except Exception as e:
        print(f"[Renk] model yuklenemedi: {e}")
        _MODEL = None
    return _MODEL


def renk_tahmin(bgr_crop):
    """
    Arac crop'undan rengi dondurur. Model yoksa/hata olursa None (cagiran HSV'ye duser).
    """
    if bgr_crop is None or getattr(bgr_crop, "size", 0) == 0:
        return None
    m = _model_yukle()
    if m is None:
        return None
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        r = m.predict(source=bgr_crop, device=dev, verbose=False)[0]
        if r.probs is None:
            return None
        top1 = int(r.probs.top1)
        ham = m.names.get(top1)
        return VCOR_ESLEME.get(ham)
    except Exception as e:
        print(f"[Renk] tahmin atlandi: {e}")
        return None
