# -*- coding: utf-8 -*-
"""
DMS katmani: MediaPipe Face Mesh ile surucu davranisi tespiti (IZOLE, opsiyonel).

Deep research'in 1 numarali onerisi: ticari DMS'lerin (Cipia/Seeing Machines)
iki-katmanli mimarisini acik kaynakla replike. Katman-1 = MediaPipe 468 landmark
(egitimsiz, offline, CPU'da gercek zamanli), katman-2 = src.dms_logic (MAR/yaw).

Tespit edilen etiketler (su an baseline'da recall=0 olanlar):
  - esneme        (MAR esik-ustu sureklilik)
  - arkaya_bakma  (buyuk yaw)
  - etrafa_bakinma(orta yaw)

IZOLE: MediaPipe yoksa / video yoksa / hata -> [] (ana pipeline'i ASLA etkilemez).
Anti-cheat: saf gorsel analiz, ortam tespiti yok.
"""
import os

from src.dms_logic import (
    mar_hesapla, yaw_davranisi, esneme_karari, MAR_ESIK, YAW_MIN_ARDISIK,
)

# MediaPipe Face Mesh landmark indeksleri
_AGIZ_UST = 13       # ust dudak ic orta
_AGIZ_ALT = 14       # alt dudak ic orta
_AGIZ_SOL = 61       # sol agiz kosesi
_AGIZ_SAG = 291      # sag agiz kosesi

VID_STRIDE = 8   # DMS davranislari (esneme/bakma) saniyeler surer; 8 stride yeterli + hizli

# solvePnP icin 3D yuz model noktalari (genel insan yuzu, mm benzeri olcek) ve
# karsilik gelen MediaPipe landmark indeksleri. Basit landmark-geometri yan
# kameralarda sasiriyordu; PnP gercek 3D oryantasyon -> kamera acisina dayanikli.
import numpy as _np
_MODEL_3D = _np.array([
    [0.0, 0.0, 0.0],          # 1   burun ucu
    [0.0, -330.0, -65.0],     # 199 cene
    [-225.0, 170.0, -135.0],  # 33  sol goz dis kose
    [225.0, 170.0, -135.0],   # 263 sag goz dis kose
    [-150.0, -150.0, -125.0], # 61  sol agiz kosesi
    [150.0, -150.0, -125.0],  # 291 sag agiz kosesi
], dtype=_np.float64)
_PNP_IDX = [1, 199, 33, 263, 61, 291]


def _yaw_tahmin(lm, w, h):
    """
    solvePnP ile yaw (kafa yatay donusu, derece). 6 landmark -> 3D-2D PnP ->
    Rodrigues -> Euler. Landmark-geometriden cok daha dayanikli (kamera acisi).
    Hata olursa 0.0 (= duz; yanlis-pozitif uretmez).
    """
    try:
        import cv2
        p2d = _np.array([[lm[i].x * w, lm[i].y * h] for i in _PNP_IDX], dtype=_np.float64)
        cam = _np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=_np.float64)
        ok, rvec, _ = cv2.solvePnP(_MODEL_3D, p2d, cam, _np.zeros((4, 1)),
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return 0.0
        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return float(angles[1])   # y ekseni = yaw
    except Exception:
        return 0.0


def dms_tespit(video_path, vid_stride=VID_STRIDE):
    """
    Surucu davranislarini (esneme, arkaya_bakma, etrafa_bakinma) tespit eder.
    MediaPipe/video/hata yoksa [] (ana akisi etkilemez).
    Her tespit: {zaman_saniye, kategori:"sofor_eylemi", etiket, confidence_score}
    """
    if not os.path.exists(video_path):
        return []
    try:
        import cv2
        import mediapipe as mp

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=False, min_detection_confidence=0.5,
        )
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        esneme_ardisik = 0
        esneme_zaman = None
        # Yaw: "herhangi bir bakma" uzerinden ARDISIK say (arkaya<->etrafa zikzagi
        # serisi bozmasin - kafa cevirme kesintisiz bakmadir). Seri bitince etiket
        # = seride gorulen EN BUYUK aci (arkaya > etrafa) -> en siddetli davranis.
        bakma_ardisik = 0     # mevcut kesintisiz bakma serisi
        bakma_seri_max_aci = 0.0   # mevcut seride gorulen en buyuk |yaw|
        bakma_seri_zaman = None    # mevcut seri baslangic zamani
        bakma_seri_confs = []      # mevcut seri guvenleri
        # En iyi (en uzun) bakma serisi sonucu
        en_iyi = {"len": 0, "aci": 0.0, "zaman": None, "confs": []}
        tespitler = []
        idx = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            if idx % vid_stride != 0:
                continue
            zaman_sn = round(idx / fps, 2)
            f_h, f_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if not res.multi_face_landmarks:
                esneme_ardisik = 0
                continue
            lm = res.multi_face_landmarks[0].landmark

            # --- Esneme (MAR sureklilik) ---
            mar = mar_hesapla(
                ust=lm[_AGIZ_UST].y, alt=lm[_AGIZ_ALT].y,
                sol=lm[_AGIZ_SOL].x, sag=lm[_AGIZ_SAG].x,
            )
            if mar >= MAR_ESIK:
                if esneme_ardisik == 0:
                    esneme_zaman = zaman_sn
                esneme_ardisik += 1
            else:
                esneme_ardisik = 0

            # --- Yaw davranisi (arkaya/etrafa bakma) - ARDISIK sureklilik ---
            yaw = _yaw_tahmin(lm, f_w, f_h)
            etk = yaw_davranisi(yaw)   # None / etrafa_bakinma / arkaya_bakma
            if etk:   # bu karede bir tur bakma var -> seriyi surdur
                if bakma_ardisik == 0:
                    bakma_seri_zaman = zaman_sn
                    bakma_seri_max_aci = 0.0
                    bakma_seri_confs = []
                bakma_ardisik += 1
                bakma_seri_max_aci = max(bakma_seri_max_aci, abs(yaw))
                bakma_seri_confs.append(min(1.0, abs(yaw) / 90.0))
                if bakma_ardisik > en_iyi["len"]:
                    en_iyi = {"len": bakma_ardisik, "aci": bakma_seri_max_aci,
                              "zaman": bakma_seri_zaman, "confs": list(bakma_seri_confs)}
            else:     # bakma yok -> seri bitti
                bakma_ardisik = 0

        cap.release()
        face_mesh.close()

        # Karar: esneme
        if esneme_karari(esneme_ardisik) or esneme_ardisik >= 8:
            tespitler.append({
                "zaman_saniye": esneme_zaman if esneme_zaman is not None else 0.0,
                "kategori": "sofor_eylemi", "etiket": "esneme",
                "confidence_score": 0.7,
            })
        # Karar: bakma (en uzun kesintisiz seri esigi gecerse). Etiket = serinin
        # EN BUYUK acisina gore (arkaya esigi astiysa arkaya_bakma, yoksa etrafa).
        if en_iyi["len"] >= YAW_MIN_ARDISIK:
            etk = yaw_davranisi(en_iyi["aci"]) or "etrafa_bakinma"
            confs = en_iyi["confs"] or [0.5]
            tespitler.append({
                "zaman_saniye": en_iyi["zaman"] if en_iyi["zaman"] is not None else 0.0,
                "kategori": "sofor_eylemi", "etiket": etk,
                "confidence_score": round(0.5 + 0.4 * (sum(confs) / len(confs)), 2),
            })
        return tespitler
    except Exception as e:
        print(f"[DMS] katman atlandi: {e}")
        return []
