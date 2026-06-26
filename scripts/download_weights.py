# -*- coding: utf-8 -*-
"""
Agirliklari BUILD ONCESI (internet aciken) indirir ve weights/ klasorune yerlestirir.
Bu agirliklar Docker imajina gomulecek; calisma aninda internet KAPALI olacak.

Calistir:  python scripts/download_weights.py

Indirilenler:
  weights/best_model.pt   -> YOLO COCO agirligi (yolov8m)
  weights/easyocr/        -> EasyOCR detection + recognition modelleri (offline)
"""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEIGHTS = os.path.join(ROOT, "weights")
EASYOCR_DIR = os.path.join(WEIGHTS, "easyocr")

os.makedirs(WEIGHTS, exist_ok=True)
os.makedirs(EASYOCR_DIR, exist_ok=True)


def indir_yolo():
    """yolov8m COCO agirligini indirip best_model.pt olarak kaydeder."""
    hedef = os.path.join(WEIGHTS, "best_model.pt")
    if os.path.exists(hedef):
        print(f"[YOLO] Zaten var: {hedef}")
        return
    print("[YOLO] yolov8m.pt indiriliyor (ultralytics)...")
    from ultralytics import YOLO
    # Bu cagri agirligi calisma dizinine indirir
    m = YOLO("yolov8m.pt")
    # Indirilen .pt dosyasini bul
    kaynak = None
    for aday in ["yolov8m.pt", os.path.join(os.getcwd(), "yolov8m.pt")]:
        if os.path.exists(aday):
            kaynak = aday
            break
    # ultralytics ckpt path'i
    if kaynak is None:
        try:
            kaynak = m.ckpt_path
        except Exception:
            pass
    if kaynak and os.path.exists(kaynak):
        shutil.copy(kaynak, hedef)
        print(f"[YOLO] Kaydedildi: {hedef}")
    else:
        raise RuntimeError("yolov8m.pt indirilemedi/bulunamadi.")


def indir_easyocr():
    """EasyOCR EN modellerini weights/easyocr/ icine indirir."""
    print("[OCR] EasyOCR modelleri indiriliyor...")
    import easyocr
    # Bu cagri detection+recognition modellerini model_storage_directory'ye indirir
    easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=EASYOCR_DIR,
        download_enabled=True,
    )
    dosyalar = os.listdir(EASYOCR_DIR)
    print(f"[OCR] Indirilen modeller: {dosyalar}")
    if not dosyalar:
        raise RuntimeError("EasyOCR modelleri indirilemedi.")


if __name__ == "__main__":
    indir_yolo()
    indir_easyocr()
    print("\nTUM AGIRLIKLAR HAZIR. weights/ icerigi:")
    for kok, _, fs in os.walk(WEIGHTS):
        for f in fs:
            p = os.path.join(kok, f)
            print(f"  {os.path.relpath(p, ROOT)}  ({os.path.getsize(p)//1024} KB)")
