# -*- coding: utf-8 -*-
"""
Ana cikarim (inference) pipeline.

Mimari:
  video -> frame sampling -> YOLO (COCO) tespit
        -> arac (car/truck/bus): tip heuristigi + renk (HSV) + plaka (kirp+OCR)
        -> kolay etiketler: cell phone->telefonla_konusma, bottle->su_icme,
                            laptop->bilgisayar, person->yolcular
        -> results.json semasi

Tasarim ilkeleri:
  - Hicbir sey cokmez: her adim try/except, tespit yoksa bos liste.
  - Offline: agirliklar lokal path'ten, otomatik indirme kapali.
  - Sure korumasi: 9 dk wall-clock guard (10 dk timeout'tan once eldekini yaz).
  - ASCII/kucuk harf etiketler labels.py'den; uydurma yok.

NOT: Anti-cheat -> ortam/hostname/IP/env tespiti YOK. Akis deterministik.
"""
import os
import time

import numpy as np
import cv2

from src.labels import KATEGORI_ETIKETLERI
from src.utils import (
    baskin_renk,
    arac_tipi_heuristik,
    plaka_normalize,
    hedef_frame_indeksleri,
)

# --- Sabitler ---
WEIGHTS_DIR = "/app/weights"
YOLO_WEIGHTS = os.path.join(WEIGHTS_DIR, "best_model.pt")   # COCO yolo agirligi
PLATE_WEIGHTS = os.path.join(WEIGHTS_DIR, "plate.pt")        # opsiyonel plaka detektoru
EASYOCR_DIR = os.path.join(WEIGHTS_DIR, "easyocr")           # offline OCR modelleri

HEDEF_FPS = 5.0          # saniyede ~5 kare ornekle
SURE_GUARD_SN = 9 * 60   # 9 dk: bu sure dolarsa eldekini yaz
CONF_ESIK = 0.35         # YOLO minimum guven
# Ayni etiket icin yakin saniyelerde tekrar tespitleri bastir (gurultu azalt)
DEDUP_PENCERE_SN = 1.5

# COCO sinif adi -> sartname (kategori, etiket) eslemesi
COCO_KOLAY = {
    "cell phone": ("sofor_eylemi", "telefonla_konusma"),
    "bottle":     ("sofor_eylemi", "su_icme"),
    "laptop":     ("nesneler", "bilgisayar"),
}
COCO_ARAC = {"car", "truck", "bus"}


def _lazy_yolo(path):
    """YOLO'yu lokal agirliktan yukler; cuda varsa cuda, yoksa cpu."""
    from ultralytics import YOLO
    import torch
    model = YOLO(path)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return model, dev


def _lazy_ocr():
    """EasyOCR'i offline (download_enabled=False) yukler. Yoksa None doner."""
    try:
        import easyocr
        import torch
        gpu = torch.cuda.is_available()
        reader = easyocr.Reader(
            ["en"],
            gpu=gpu,
            model_storage_directory=EASYOCR_DIR if os.path.isdir(EASYOCR_DIR) else None,
            download_enabled=False,
        )
        return reader
    except Exception as e:
        print(f"[OCR] EasyOCR yuklenemedi, plaka OCR atlanacak: {e}")
        return None


def _plaka_oku(reader, arac_crop):
    """Arac kirpintisinda plaka okumaya calisir. Bulamazsa '' doner."""
    if reader is None or arac_crop is None or arac_crop.size == 0:
        return "", 0.0
    try:
        sonuc = reader.readtext(arac_crop, detail=1, paragraph=False)
        en_iyi_metin, en_iyi_conf = "", 0.0
        for item in sonuc:
            # easyocr: (bbox, text, conf)
            metin = item[1] if len(item) > 1 else ""
            conf = float(item[2]) if len(item) > 2 else 0.0
            norm = plaka_normalize(metin)
            if norm and conf >= en_iyi_conf:
                en_iyi_metin, en_iyi_conf = norm, conf
        return en_iyi_metin, en_iyi_conf
    except Exception:
        return "", 0.0


def _yolcu_konumu(bbox, frame_w):
    """person bbox'indan kaba yolcu konumu (on_koltuk / arka_koltuk_1 / arka_koltuk_2)."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    alan = (x2 - x1) * (y2 - y1)
    # Buyuk ve merkeze yakin -> on koltuk; kucuk ve kenarda -> arka
    if alan > 0.20 * frame_w * frame_w:
        return "on_koltuk"
    return "arka_koltuk_1" if cx < frame_w / 2 else "arka_koltuk_2"


def run_inference(video_path, weights_path=YOLO_WEIGHTS):
    """
    Video uzerinde cikarim yapar, sartname semasinda dict dondurur.
    Hata olsa bile gecerli (bos da olsa) bir sema doner -- asla cokmez.
    """
    video_id = os.path.basename(video_path) if video_path else "video.mp4"
    bos_sonuc = {
        "video_id": video_id,
        "arac_bilgisi": {"tip": "", "plaka": "", "renk": "", "confidence_score": 0.0},
        "tespitler": [],
    }

    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap or not cap.isOpened():
            print(f"[Inference] Video acilamadi: {video_path}")
            return bos_sonuc

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        toplam = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        indeksler = set(hedef_frame_indeksleri(toplam, fps, HEDEF_FPS))
        print(f"[Inference] fps={fps:.1f} toplam_frame={toplam} ornek={len(indeksler)}")

        model, dev = _lazy_yolo(weights_path)
        names = model.names  # {id: 'car', ...}
        reader = _lazy_ocr()

        baslangic = time.time()
        tespitler = []
        son_etiket_zaman = {}   # dedup icin
        # En guvenilir arac tespiti (tum video icindeki en yuksek conf arac)
        en_iyi_arac = {"conf": 0.0, "tip": "", "renk": "", "plaka": "", "plaka_conf": 0.0}

        idx = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            if idx not in indeksler:
                continue

            # Sure korumasi: 9 dk dolduysa eldekini dondur
            if time.time() - baslangic > SURE_GUARD_SN:
                print("[Inference] Sure guard tetiklendi, eldeki sonuclar yaziliyor.")
                break

            zaman_sn = round(idx / fps, 2)

            try:
                res = model(frame, device=dev, half=(dev == "cuda"),
                            conf=CONF_ESIK, verbose=False)[0]
            except Exception as e:
                print(f"[Inference] frame {idx} YOLO hatasi: {e}")
                continue

            if res.boxes is None:
                continue

            for box in res.boxes:
                try:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    coco_ad = names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                    bbox = (x1, y1, x2, y2)
                except Exception:
                    continue

                # --- Arac ---
                if coco_ad in COCO_ARAC:
                    crop = frame[int(max(0, y1)):int(y2), int(max(0, x1)):int(x2)]
                    renk = baskin_renk(crop)
                    tip = arac_tipi_heuristik(coco_ad, bbox)
                    if conf > en_iyi_arac["conf"]:
                        plaka, p_conf = _plaka_oku(reader, crop)
                        en_iyi_arac.update(
                            conf=conf, tip=tip, renk=renk,
                            plaka=plaka, plaka_conf=p_conf,
                        )
                    continue

                # --- Kolay COCO etiketleri ---
                if coco_ad in COCO_KOLAY:
                    kat, etk = COCO_KOLAY[coco_ad]
                    anahtar = (kat, etk)
                    if zaman_sn - son_etiket_zaman.get(anahtar, -99) >= DEDUP_PENCERE_SN:
                        tespitler.append({
                            "zaman_saniye": zaman_sn,
                            "kategori": kat,
                            "etiket": etk,
                            "confidence_score": round(conf, 2),
                        })
                        son_etiket_zaman[anahtar] = zaman_sn
                    continue

                # --- Yolcular (person) ---
                if coco_ad == "person":
                    etk = _yolcu_konumu(bbox, frame_w)
                    anahtar = ("yolcular", etk)
                    if zaman_sn - son_etiket_zaman.get(anahtar, -99) >= DEDUP_PENCERE_SN:
                        tespitler.append({
                            "zaman_saniye": zaman_sn,
                            "kategori": "yolcular",
                            "etiket": etk,
                            "confidence_score": round(conf, 2),
                        })
                        son_etiket_zaman[anahtar] = zaman_sn
                    continue

        arac_bilgisi = {
            "tip": en_iyi_arac["tip"],
            "plaka": en_iyi_arac["plaka"],
            "renk": en_iyi_arac["renk"],
            "confidence_score": round(en_iyi_arac["conf"], 2),
        }
        return {
            "video_id": video_id,
            "arac_bilgisi": arac_bilgisi,
            "tespitler": tespitler,
        }

    except Exception as e:
        print(f"[Inference] Beklenmeyen hata, bos sema donuluyor: {e}")
        return bos_sonuc
    finally:
        if cap is not None:
            cap.release()
