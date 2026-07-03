# -*- coding: utf-8 -*-
"""
Surucu davranisi tespiti (IZOLE) - driver_monitoring_v4 fine-tune YOLOv8m.

7 sinif egitildi; bunlardan SADECE yarisma semasiyla ortusen davranislari
raporlariz (safe_driving/radio/makyaj raporlanmaz - bizim etiket setinde yok).
teknocan/sigara/dms gibi izole: agirlik/video/hata yoksa [] doner.

UYARI (durustluk): bu model validasyonda %99.5 mAP verdi ANCAK veri seti ayni
cekimlerin komsu karelerinden olusuyor (near-duplicate); gercek videoda daha
dusuk beklenir. track-level voting ile yanlis pozitif azaltilir.
"""
import os

_DOCKER_W = "/app/weights"
_LOCAL_W = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
WEIGHTS_DIR = _DOCKER_W if os.path.isdir(_DOCKER_W) else _LOCAL_W
WEIGHTS = os.path.join(WEIGHTS_DIR, "dms_actions.pt")

# driver_monitoring_v4 sinif adi -> bizim ASCII-safe etiket (None = raporlanmaz)
SINIF_ESLEME = {
    "Drinking_water": "su_icme",
    "Hair_and _Makeup": None,
    "Reaching_behind": "arkaya_uzanma",
    "Talking_on_phone": "telefonla_konusma",
    "Texting_on_phone": "telefonla_konusma",
    "operating_radio": None,
    "safe_driving": None,
}

VID_STRIDE = 8
CONF_ESIK = 0.45      # sismis model -> yuksek esik (precision)
MIN_FRAME = 3         # en az 3 karede gorulmeli (track-voting benzeri, yanlis poz. ele)


def dms_actions_tespit(video_path, vid_stride=VID_STRIDE):
    """
    Surucu davranislarini dondurur. Agirlik/video/hata yoksa [].
    Her etiket icin en az MIN_FRAME karede gorulurse tek (ilk zaman + max conf) tespit.
    """
    if not os.path.exists(WEIGHTS) or not os.path.exists(video_path):
        return []
    try:
        from ultralytics import YOLO
        import torch
        import cv2

        model = YOLO(WEIGHTS)
        isimler = model.names   # {id: 'Drinking_water', ...}
        dev = "cuda" if torch.cuda.is_available() else "cpu"

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        # etiket -> [sayac, ilk_zaman, max_conf]
        toplam = {}
        f_idx = 0
        for r in model.predict(source=video_path, stream=True, conf=CONF_ESIK,
                               vid_stride=vid_stride, device=dev, verbose=False):
            f_idx += vid_stride
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for cls_id, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                ham = isimler.get(int(cls_id))
                etiket = SINIF_ESLEME.get(ham)
                if etiket is None:
                    continue
                zaman = round(f_idx / fps, 2)
                if etiket not in toplam:
                    toplam[etiket] = [0, zaman, 0.0]
                toplam[etiket][0] += 1
                toplam[etiket][2] = max(toplam[etiket][2], float(conf))

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
        print(f"[DMS-Actions] katman atlandi: {e}")
        return []
