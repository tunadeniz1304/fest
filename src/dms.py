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
    mar_hesapla, yaw_davranisi, esneme_karari, MAR_ESIK,
)

# MediaPipe Face Mesh landmark indeksleri
_AGIZ_UST = 13       # ust dudak ic orta
_AGIZ_ALT = 14       # alt dudak ic orta
_AGIZ_SOL = 61       # sol agiz kosesi
_AGIZ_SAG = 291      # sag agiz kosesi
_BURUN = 1           # burun ucu
_YUZ_SOL = 234       # sol yuz kenari
_YUZ_SAG = 454       # sag yuz kenari

VID_STRIDE = 8   # DMS davranislari (esneme/bakma) saniyeler surer; 8 stride yeterli + hizli


def _yaw_tahmin(lm):
    """
    Basit yaw tahmini: burnun, yuz sol/sag kenarlarina yatay uzakliklarinin
    oranindan kafa donusu (derece ~ -90..90). 6DRepNet yerine landmark-geometri
    (ekstra agirlik/model gerektirmez, offline).
    """
    try:
        sol_d = abs(lm[_BURUN].x - lm[_YUZ_SOL].x)
        sag_d = abs(lm[_YUZ_SAG].x - lm[_BURUN].x)
        toplam = sol_d + sag_d
        if toplam <= 1e-6:
            return 0.0
        # oran 0.5 -> duz; 0'a/1'e yaklasinca yana doner
        oran = sol_d / toplam
        return (oran - 0.5) * 180.0   # -90..+90 yaklasik
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
        yaw_say = {}          # etiket -> frame sayisi
        yaw_zaman = {}        # etiket -> ilk zaman
        yaw_conf = {}         # etiket -> [orantili guven]
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

            # --- Yaw davranisi (arkaya/etrafa bakma) ---
            yaw = _yaw_tahmin(lm)
            etk = yaw_davranisi(yaw)
            if etk:
                yaw_say[etk] = yaw_say.get(etk, 0) + 1
                yaw_zaman.setdefault(etk, zaman_sn)
                yaw_conf.setdefault(etk, []).append(min(1.0, abs(yaw) / 90.0))

        cap.release()
        face_mesh.close()

        # Karar: esneme
        if esneme_karari(esneme_ardisik) or esneme_ardisik >= 8:
            tespitler.append({
                "zaman_saniye": esneme_zaman if esneme_zaman is not None else 0.0,
                "kategori": "sofor_eylemi", "etiket": "esneme",
                "confidence_score": 0.7,
            })
        # Karar: yaw davranislari (min sureklilik + ortalama guven)
        for etk, sayi in yaw_say.items():
            if sayi >= 3:   # anlik bakisi ele
                confs = yaw_conf.get(etk, [0.5])
                tespitler.append({
                    "zaman_saniye": yaw_zaman[etk],
                    "kategori": "sofor_eylemi", "etiket": etk,
                    "confidence_score": round(0.5 + 0.4 * (sum(confs) / len(confs)), 2),
                })
        return tespitler
    except Exception as e:
        print(f"[DMS] katman atlandi: {e}")
        return []
