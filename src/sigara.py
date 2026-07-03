# -*- coding: utf-8 -*-
"""
sigara_icme tespiti (IZOLE, opsiyonel katman).

driver_smoking (Roboflow, CC BY 4.0, 312 gorsel) ile fine-tune edilmis YOLOv8n
agirligi (weights/sigara.pt). teknocan/DMS gibi izole: agirlik/video/hata yoksa
[] doner, ana pipeline'i ASLA etkilemez.

Anti-cheat: saf gorsel tespit, ortam kontrolu yok.
"""
import os

# Docker'da /app/weights; yerelde proje koku/weights
_DOCKER_W = "/app/weights"
_LOCAL_W = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
WEIGHTS_DIR = _DOCKER_W if os.path.isdir(_DOCKER_W) else _LOCAL_W
SIGARA_WEIGHTS = os.path.join(WEIGHTS_DIR, "sigara.pt")

VID_STRIDE = 8        # sigara saniyeler surer; seyrek tarama yeterli + hizli
CONF_ESIK = 0.35
MIN_FRAME = 2         # anlik tek-frame yanlis pozitifi ele (precision)


def sigara_tespit(video_path, vid_stride=VID_STRIDE):
    """
    sigara_icme tespitlerini dondurur. Agirlik/video/hata yoksa [].
    En az MIN_FRAME karede gorulurse tek (en yuksek conf) tespit raporlanir.
    Cikti: [{zaman_saniye, kategori:"sofor_eylemi", etiket:"sigara_icme", confidence_score}]
    """
    if not os.path.exists(SIGARA_WEIGHTS) or not os.path.exists(video_path):
        return []
    try:
        from ultralytics import YOLO
        import torch
        import cv2

        model = YOLO(SIGARA_WEIGHTS)
        dev = "cuda" if torch.cuda.is_available() else "cpu"

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        gorulen = 0
        en_iyi_conf = 0.0
        ilk_zaman = None
        f_idx = 0
        for r in model.predict(source=video_path, stream=True, conf=CONF_ESIK,
                               vid_stride=vid_stride, device=dev, verbose=False):
            f_idx += vid_stride
            if r.boxes is None or len(r.boxes) == 0:
                continue
            cf = float(r.boxes.conf.max())
            gorulen += 1
            if ilk_zaman is None:
                ilk_zaman = round(f_idx / fps, 2)
            en_iyi_conf = max(en_iyi_conf, cf)

        if gorulen >= MIN_FRAME:
            return [{
                "zaman_saniye": ilk_zaman if ilk_zaman is not None else 0.0,
                "kategori": "sofor_eylemi",
                "etiket": "sigara_icme",
                "confidence_score": round(en_iyi_conf, 2),
            }]
        return []
    except Exception as e:
        print(f"[Sigara] katman atlandi: {e}")
        return []
