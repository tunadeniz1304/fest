# -*- coding: utf-8 -*-
"""
Plaka bolgesi tespiti - License Plate fine-tune YOLOv8m (mAP %97, GUVENILIR).

Izole: weights/plate.pt varsa egitilmis model ile arac crop'unda plaka bbox'u
bulur (OCR oncesi); yoksa cagiran taraf klasik plaka_bolgesi_bul'a duser.
Cikti: en yuksek conf'lu plaka crop'u (bgr) veya None.
"""
import os

_DOCKER_W = "/app/weights"
_LOCAL_W = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
WEIGHTS_DIR = _DOCKER_W if os.path.isdir(_DOCKER_W) else _LOCAL_W
WEIGHTS = os.path.join(WEIGHTS_DIR, "plate.pt")

CONF_ESIK = 0.35

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
        print(f"[Plaka] model yuklenemedi: {e}")
        _MODEL = None
    return _MODEL


def plaka_crop_bul(bgr_crop):
    """
    Arac crop'unda plaka bolgesini bulur, plaka bgr crop'unu dondurur.
    Model yoksa/plaka yoksa/hata olursa None (cagiran klasik yonteme duser).
    """
    if bgr_crop is None or getattr(bgr_crop, "size", 0) == 0:
        return None
    m = _model_yukle()
    if m is None:
        return None
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        r = m.predict(source=bgr_crop, conf=CONF_ESIK, device=dev, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None
        # En yuksek conf'lu plaka kutusu
        i = int(r.boxes.conf.argmax())
        x1, y1, x2, y2 = (int(v) for v in r.boxes.xyxy[i].tolist())
        h, w = bgr_crop.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return bgr_crop[y1:y2, x1:x2]
    except Exception as e:
        print(f"[Plaka] tespit atlandi: {e}")
        return None
