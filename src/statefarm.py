# -*- coding: utf-8 -*-
"""
State Farm surucu davranisi - YEDEK katman (classification, YOLOv8m-cls).

dms_actions (detection) birincil; bu classification modeli yedek/teyit. Ayni
davranis dedup ile birlestirilir. IZOLE: agirlik/video/hata yoksa [].

UYARI (durustluk): %99.9 top1 ANCAK State Farm'da ayni surucunun kareleri
train+val'e dagilmis (klasik leakage). Yedek amacli; tek basina guvenilmez.
Bu yuzden YUKSEK esik + cok-frame sarti.
"""
import os

_DOCKER_W = "/app/weights"
_LOCAL_W = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
WEIGHTS_DIR = _DOCKER_W if os.path.isdir(_DOCKER_W) else _LOCAL_W
WEIGHTS = os.path.join(WEIGHTS_DIR, "statefarm.pt")

# State Farm standart sinif tanimlari -> bizim etiket (None = raporlanmaz)
SF_ESLEME = {
    "c0": None,                    # safe driving
    "c1": "telefonla_konusma",     # texting - right
    "c2": "telefonla_konusma",     # talking on phone - right
    "c3": "telefonla_konusma",     # texting - left
    "c4": "telefonla_konusma",     # talking on phone - left
    "c5": None,                    # operating radio
    "c6": "su_icme",               # drinking
    "c7": "arkaya_uzanma",         # reaching behind
    "c8": None,                    # hair and makeup
    "c9": None,                    # talking to passenger
}

VID_STRIDE = 10
CONF_ESIK = 0.6      # sismis model -> cok yuksek esik
MIN_FRAME = 4        # yedek katman: en az 4 karede gorulmeli


def statefarm_tespit(video_path, vid_stride=VID_STRIDE):
    """State Farm classification ile davranis (yedek). Agirlik/video/hata yoksa []."""
    if not os.path.exists(WEIGHTS) or not os.path.exists(video_path):
        return []
    try:
        from ultralytics import YOLO
        import torch
        import cv2

        model = YOLO(WEIGHTS)
        isimler = model.names
        dev = "cuda" if torch.cuda.is_available() else "cpu"

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        toplam = {}  # etiket -> [sayac, ilk_zaman, max_conf]
        f_idx = 0
        for r in model.predict(source=video_path, stream=True, vid_stride=vid_stride,
                               device=dev, verbose=False):
            f_idx += vid_stride
            if r.probs is None:
                continue
            conf = float(r.probs.top1conf)
            if conf < CONF_ESIK:
                continue
            ham = isimler.get(int(r.probs.top1))
            etiket = SF_ESLEME.get(ham)
            if etiket is None:
                continue
            zaman = round(f_idx / fps, 2)
            if etiket not in toplam:
                toplam[etiket] = [0, zaman, 0.0]
            toplam[etiket][0] += 1
            toplam[etiket][2] = max(toplam[etiket][2], conf)

        sonuc = []
        for etiket, (sayac, ilk_zaman, mx) in toplam.items():
            if sayac >= MIN_FRAME:
                sonuc.append({
                    "zaman_saniye": ilk_zaman,
                    "kategori": "sofor_eylemi",
                    "etiket": etiket,
                    "confidence_score": round(mx, 2),
                })
        return sonuc
    except Exception as e:
        print(f"[StateFarm] katman atlandi: {e}")
        return []
