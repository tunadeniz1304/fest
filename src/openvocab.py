# -*- coding: utf-8 -*-
"""
teknocan icin YOLO-World open-vocabulary tespit (izole, opsiyonel inovasyon katmani).

teknocan COCO'da yok -> sifir egitimle tespit icin open-vocab detektor.
Deep research YOLOE onerdi; ultralytics 8.2.103'te YOLOE yok, YOLOWorld var
(ayni prompt-tabanli open-vocab yaklasimi). Text prompt ile calisir, egitimsiz.

IZOLE: agirlik yoksa veya hata olursa [] doner; ana pipeline'i ASLA etkilemez.
"""
import os

# Docker'da /app/weights; yerelde proje koku/weights
_DOCKER_W = "/app/weights"
_LOCAL_W = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
WEIGHTS_DIR = _DOCKER_W if os.path.isdir(_DOCKER_W) else _LOCAL_W
# Sinif embedding'leri GOMULU model (offline). download_weights.py uretir;
# set_classes() runtime'da cagrilmaz -> CLIP metin kodlayici indirme gerekmez.
WORLD_WEIGHTS = os.path.join(WEIGHTS_DIR, "yolo_world_teknocan.pt")
# Geri donus: gomulu model yoksa ham model + runtime set_classes (CLIP imajda olmali)
WORLD_WEIGHTS_RAW = os.path.join(WEIGHTS_DIR, "yolo_world.pt")

# teknocan ~ markali icecek kutusu. Open-vocab text prompt'lari.
PROMPTLAR = ["can", "beverage can", "soda can", "tin can"]
SURE_STRIDE = 15
CONF_ESIK = 0.30


def teknocan_tespit(video_path, vid_stride=SURE_STRIDE):
    """
    teknocan tespitlerini dondurur. Agirlik/video/hata yoksa [] (ana akisi etkilemez).
    Her tespit: {zaman_saniye, kategori:"nesneler", etiket:"teknocan", confidence_score}
    """
    if not os.path.exists(video_path):
        return []
    try:
        from ultralytics import YOLOWorld
        import torch
        import cv2

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # Tercih: gomulu-embedding model (CLIP indirmesiz). Yoksa ham + set_classes.
        if os.path.exists(WORLD_WEIGHTS):
            model = YOLOWorld(WORLD_WEIGHTS)
        elif os.path.exists(WORLD_WEIGHTS_RAW):
            model = YOLOWorld(WORLD_WEIGHTS_RAW)
            model.set_classes(PROMPTLAR)
        else:
            return []

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        sonuc = []
        gorulen = False
        f_idx = 0
        for r in model.predict(source=video_path, stream=True, conf=CONF_ESIK,
                               vid_stride=vid_stride, device=dev, verbose=False):
            f_idx += vid_stride
            if gorulen:
                break
            if r.boxes is None or len(r.boxes) == 0:
                continue
            # Ilk guclu tespiti raporla (tek teknocan yeterli, dedup pipeline'da)
            cf = float(r.boxes.conf.max())
            sonuc.append({
                "zaman_saniye": round(f_idx / fps, 2),
                "kategori": "nesneler",
                "etiket": "teknocan",
                "confidence_score": round(cf, 2),
            })
            gorulen = True
        return sonuc
    except Exception as e:
        print(f"[OpenVocab] teknocan tespiti atlandi: {e}")
        return []
