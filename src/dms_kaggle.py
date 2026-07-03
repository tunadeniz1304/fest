# -*- coding: utf-8 -*-
"""
Kaggle DMS (habbas11, Apache 2.0) fine-tune YOLOv8m katmani (IZOLE).

5 sinif: Open Eye, Closed Eye, Cigarette, Phone, Seatbelt.
NEGATIF-DENGELI egitildi (State Farm c0 + video kareleri %31 background) ->
"interior=ihlal" ezberi YOK, temiz videoda yanlis-pozitif vermez.

Bizim sema etiketlerine esleme:
  Cigarette  -> sigara_icme
  Seatbelt   -> emniyet_kemeri_ihlali   (NOT: bu modelde "Seatbelt" = kemer TAKILI;
               ihlal "kemer YOK" oldugu icin asagida ACIKLAMA + parametre var)
  Phone      -> telefonla_konusma       (COCO da veriyor; dedup birlestirir)
  Closed Eye -> (yorgunluk sinyali; su an raporlanmaz, MediaPipe EAR birincil)

IZOLE: agirlik/video/hata yoksa [] doner, ana pipeline'i ASLA etkilemez.
Anti-cheat: saf gorsel tespit, ortam kontrolu yok.
"""
import os

_DOCKER_W = "/app/weights"
_LOCAL_W = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
WEIGHTS_DIR = _DOCKER_W if os.path.isdir(_DOCKER_W) else _LOCAL_W
WEIGHTS = os.path.join(WEIGHTS_DIR, "sigara_kemer.pt")

# Kaggle DMS sinif adi -> bizim etiket (None = raporlanmaz)
# NOT: "Seatbelt" modelde kemer TAKILI demek. Yarisma "emniyet_kemeri_ihlali" istiyor.
# Guvenli yaklasim: Seatbelt'i su an raporlamiyoruz (ihlal = yokluk; tek-sinifla
# "ihlal" demek yanlis olur). Sigara/telefon dogrudan eslenir.
SINIF_ESLEME = {
    "Cigarette": "sigara_icme",
    "Phone": "telefonla_konusma",
    "Seatbelt": None,          # "takili" demek; ihlal degil -> raporlama
    "Open Eye": None,
    "Closed Eye": None,        # yorgunluk MediaPipe EAR'dan gelir
}

VID_STRIDE = 8
# Gercek videoda Cigarette ara-sira yanlis-pozitif verdigi icin yuksek esik +
# cok-frame sarti (precision oncelik). Gercek dashcam'de calisir; sentetik domain-gap.
CONF_ESIK = 0.50
MIN_FRAME = 5      # en az 5 karede gorulmeli (yanlis-pozitif sertce ele)


def dms_kaggle_tespit(video_path, vid_stride=VID_STRIDE):
    """
    Kaggle DMS modeliyle sigara/telefon tespiti. Agirlik/video/hata yoksa [].
    Her etiket icin en az MIN_FRAME karede gorulurse tek (ilk zaman + max conf) tespit.
    """
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

        toplam = {}   # etiket -> [sayac, ilk_zaman, max_conf]
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
        print(f"[DMS-Kaggle] katman atlandi: {e}")
        return []
