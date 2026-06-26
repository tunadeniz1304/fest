# -*- coding: utf-8 -*-
"""
Track-tabanli cikarim pipeline: ByteTrack + temporal voting.

Mimari:
  video -> model.track (ByteTrack, vid_stride) -> her track_id icin tahminleri biriktir
        -> track bitince: cls/tip/renk/koltuk cogunluk oyu, plaka karakter oylama,
           eylem frame-orani esigi -> results.json semasi

Tasarim ilkeleri:
  - Hicbir sey cokmez: her adim try/except, hata olursa bos gecerli sema.
  - Offline: agirliklar lokal path; otomatik indirme kapali.
  - Sure korumasi: 9 dk wall-clock guard.
  - Temporal voting: tek-frame yanlis pozitifleri track boyunca eler (precision^),
    track buffer kacan frame'leri koprular (recall^).

NOT: Anti-cheat -> ortam/hostname/IP/env tespiti YOK. Akis deterministik.
"""
import os
import time
from collections import defaultdict

import cv2

from src.labels import esik_al
from src.aggregate import (
    cogunluk_oyu, plaka_karakter_oylama, eylem_karari, tespit_dedup, slalom_var_mi,
)
from src.utils import baskin_renk, arac_tipi_heuristik, plaka_normalize

# Docker'da /app/weights; yerelde proje koku/weights. Hangisi varsa onu kullan.
_DOCKER_W = "/app/weights"
_LOCAL_W = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
WEIGHTS_DIR = _DOCKER_W if os.path.isdir(_DOCKER_W) else _LOCAL_W
YOLO_WEIGHTS = os.path.join(WEIGHTS_DIR, "best_model.pt")
TRACKER_CFG = os.path.join(WEIGHTS_DIR, "bytetrack.yaml")
EASYOCR_DIR = os.path.join(WEIGHTS_DIR, "easyocr")

VID_STRIDE = 5
SURE_GUARD_SN = 9 * 60
COCO_ARAC = {"car", "truck", "bus"}
COCO_KOLAY = {
    "cell phone": ("sofor_eylemi", "telefonla_konusma"),
    "bottle": ("sofor_eylemi", "su_icme"),
    "laptop": ("nesneler", "bilgisayar"),
}


def _bos_sonuc(video_id):
    return {
        "video_id": video_id,
        "arac_bilgisi": {"tip": "", "plaka": "", "renk": "", "confidence_score": 0.0},
        "tespitler": [],
    }


def _lazy_ocr():
    """EasyOCR'i offline (download_enabled=False) yukler. Yoksa None."""
    try:
        import easyocr
        import torch
        return easyocr.Reader(
            ["en"], gpu=torch.cuda.is_available(),
            model_storage_directory=EASYOCR_DIR if os.path.isdir(EASYOCR_DIR) else None,
            download_enabled=False,
        )
    except Exception as e:
        print(f"[OCR] EasyOCR yuklenemedi, plaka OCR atlanacak: {e}")
        return None


def run_inference(video_path, weights_path=YOLO_WEIGHTS):
    video_id = os.path.basename(video_path) if video_path else "video.mp4"

    # Video acilabilir mi? (cokmeden erken don)
    cap = cv2.VideoCapture(video_path)
    if not cap or not cap.isOpened():
        if cap:
            cap.release()
        print(f"[Inference] Video acilamadi: {video_path}")
        return _bos_sonuc(video_id)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    cap.release()

    try:
        from ultralytics import YOLO
        import torch
        model = YOLO(weights_path)
        names = model.names
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        reader = _lazy_ocr()
    except Exception as e:
        print(f"[Inference] Model yuklenemedi: {e}")
        return _bos_sonuc(video_id)

    # track_id -> birikim
    arac_tip = defaultdict(list)      # [(tip, conf)]
    arac_renk = defaultdict(list)     # [(renk, conf)]
    arac_plaka = defaultdict(list)    # [(plaka, conf)]
    arac_conf = defaultdict(list)     # [conf]
    arac_merkez_x = defaultdict(list)  # [cx] -> slalom (trajectory) icin
    arac_ilk_zaman = {}               # tid -> ilk gorulme saniyesi
    eylem_say = defaultdict(lambda: defaultdict(int))    # tid -> {(kat,etk): frame}
    eylem_conf = defaultdict(lambda: defaultdict(list))
    eylem_zaman = defaultdict(dict)   # tid -> {(kat,etk): ilk_saniye}
    track_len = defaultdict(int)
    yolcu_oy = defaultdict(list)      # tid -> [(etk, conf)]
    yolcu_zaman = {}

    baslangic = time.time()
    tracker = TRACKER_CFG if os.path.exists(TRACKER_CFG) else "bytetrack.yaml"
    try:
        akis = model.track(
            source=video_path, tracker=tracker, persist=True, stream=True,
            conf=0.20, vid_stride=VID_STRIDE, device=dev,
            half=(dev == "cuda"), verbose=False,
        )
        f_idx = 0
        for r in akis:
            if time.time() - baslangic > SURE_GUARD_SN:
                print("[Inference] Sure guard tetiklendi.")
                break
            f_idx += VID_STRIDE
            zaman_sn = round(f_idx / fps, 2)
            if r.boxes is None or r.boxes.id is None:
                continue
            ids = r.boxes.id.int().cpu().tolist()
            clss = r.boxes.cls.int().cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()
            xyxy = r.boxes.xyxy.cpu().tolist()
            frame = r.orig_img
            for tid, c, cf, box in zip(ids, clss, confs, xyxy):
                coco_ad = names.get(c, str(c))
                if cf < esik_al(coco_ad):
                    continue
                track_len[tid] += 1
                x1, y1, x2, y2 = box
                if coco_ad in COCO_ARAC:
                    crop = frame[int(max(0, y1)):int(y2), int(max(0, x1)):int(x2)]
                    arac_tip[tid].append((arac_tipi_heuristik(coco_ad, (x1, y1, x2, y2)), cf))
                    arac_renk[tid].append((baskin_renk(crop), cf))
                    arac_conf[tid].append(cf)
                    arac_merkez_x[tid].append((x1 + x2) / 2.0)
                    arac_ilk_zaman.setdefault(tid, zaman_sn)
                    if reader is not None and crop.size > 0:
                        try:
                            for it in reader.readtext(crop, detail=1, paragraph=False):
                                norm = plaka_normalize(it[1])
                                if norm:
                                    arac_plaka[tid].append((norm, float(it[2])))
                        except Exception:
                            pass
                elif coco_ad in COCO_KOLAY:
                    anahtar = COCO_KOLAY[coco_ad]
                    eylem_say[tid][anahtar] += 1
                    eylem_conf[tid][anahtar].append(cf)
                    eylem_zaman[tid].setdefault(anahtar, zaman_sn)
                elif coco_ad == "person":
                    alan = (x2 - x1) * (y2 - y1)
                    etk = "on_koltuk" if alan > 0.20 * frame_w * frame_w else (
                        "arka_koltuk_1" if (x1 + x2) / 2 < frame_w / 2 else "arka_koltuk_2")
                    yolcu_oy[tid].append((etk, cf))
                    yolcu_zaman.setdefault(tid, zaman_sn)
    except Exception as e:
        print(f"[Inference] Track akisi hatasi: {e}")

    # --- Nihai karar: en guvenilir aracdan arac_bilgisi ---
    en_iyi_tid, en_iyi_conf = None, -1.0
    for tid, cs in arac_conf.items():
        ort = sum(cs) / len(cs)
        if ort > en_iyi_conf:
            en_iyi_conf, en_iyi_tid = ort, tid
    arac_bilgisi = {"tip": "", "plaka": "", "renk": "", "confidence_score": 0.0}
    if en_iyi_tid is not None:
        tip, _ = cogunluk_oyu(arac_tip[en_iyi_tid], min_count=1, min_ratio=0.0)
        renk, _ = cogunluk_oyu(arac_renk[en_iyi_tid], min_count=1, min_ratio=0.0)
        plaka, _ = plaka_karakter_oylama(arac_plaka[en_iyi_tid])
        arac_bilgisi = {
            "tip": tip or "", "plaka": plaka or "", "renk": renk or "",
            "confidence_score": round(en_iyi_conf, 2),
        }

    # --- Tespitler: eylem-frame-orani esigi (anlik flash'lari eler) ---
    tespitler = []
    for tid, anahtarlar in eylem_say.items():
        for anahtar, sayi in anahtarlar.items():
            confs = eylem_conf[tid][anahtar]
            ort = sum(confs) / len(confs) if confs else 0.0
            raporla, conf = eylem_karari(sayi, track_len[tid], ort)
            if raporla:
                kat, etk = anahtar
                tespitler.append({
                    "zaman_saniye": eylem_zaman[tid][anahtar],
                    "kategori": kat, "etiket": etk, "confidence_score": conf,
                })
    for tid, oylar in yolcu_oy.items():
        etk, conf = cogunluk_oyu(oylar, min_count=2, min_ratio=0.4)
        if etk:
            tespitler.append({
                "zaman_saniye": yolcu_zaman[tid],
                "kategori": "yolcular", "etiket": etk, "confidence_score": conf,
            })

    # --- Slalom: arac trajectory'sinin yatay zig-zag'indan (egitimsiz) ---
    for tid, xs in arac_merkez_x.items():
        var, conf = slalom_var_mi(xs, frame_w)
        if var:
            tespitler.append({
                "zaman_saniye": arac_ilk_zaman.get(tid, 0.0),
                "kategori": "sofor_eylemi", "etiket": "slalom",
                "confidence_score": conf,
            })

    # Mukerrer (kategori, etiket) tespitlerini birlestir (precision)
    tespitler = tespit_dedup(tespitler)
    return {"video_id": video_id, "arac_bilgisi": arac_bilgisi, "tespitler": tespitler}
