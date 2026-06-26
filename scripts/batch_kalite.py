# -*- coding: utf-8 -*-
"""
data/samples/ icindeki TUM videolari pipeline'dan gecirip toplu kalite raporu uretir.
Her video: cozunurluk, sure, tespit edilen etiketler (kategori bazli), sema gecerliligi.
Sonucu data/output/kalite_raporu.json + konsol ozeti olarak yazar.
"""
import os
import sys
import json
import time
import glob

sys.path.insert(0, os.path.abspath("."))
from src.predict import run_inference
from src.labels import validate_results
import cv2

SAMPLES = "data/samples"
WEIGHTS = "weights/best_model.pt"
os.makedirs("data/output", exist_ok=True)

videolar = sorted(glob.glob(os.path.join(SAMPLES, "*.mp4")))
print(f"Toplam {len(videolar)} video bulundu.\n")

rapor = []
for i, v in enumerate(videolar, 1):
    ad = os.path.basename(v)
    cap = cv2.VideoCapture(v)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    sure_video = round(fc / fps, 1) if fps else 0

    print(f"[{i}/{len(videolar)}] {ad}  ({w}x{h}, {sure_video}s)")
    t0 = time.time()
    try:
        sonuc = run_inference(v, WEIGHTS)
        islem_sure = time.time() - t0
        hatalar = validate_results(sonuc)
        from collections import Counter
        etiketler = Counter((t["kategori"], t["etiket"]) for t in sonuc["tespitler"])
        ab = sonuc["arac_bilgisi"]
        ozet = {
            "video": ad, "cozunurluk": f"{w}x{h}", "video_sure_sn": sure_video,
            "islem_sure_sn": round(islem_sure, 1),
            "arac": f"{ab['tip']}/{ab['renk']}/{ab['plaka'] or '-'} ({ab['confidence_score']})",
            "etiketler": {f"{k[0]}:{k[1]}": n for k, n in etiketler.items()},
            "tespit_sayisi": len(sonuc["tespitler"]),
            "sema_hatasi": len(hatalar),
        }
        print(f"      arac: {ozet['arac']}")
        print(f"      etiketler: {ozet['etiketler'] or '(yok)'}")
        print(f"      sure: {ozet['islem_sure_sn']}s | sema: {'OK' if not hatalar else 'HATA'}\n")
        rapor.append(ozet)
    except Exception as e:
        print(f"      HATA: {e}\n")
        rapor.append({"video": ad, "hata": str(e)})

with open("data/output/kalite_raporu.json", "w", encoding="utf-8") as f:
    json.dump(rapor, f, ensure_ascii=False, indent=2)

# Ozet tablo
print("=" * 60)
print("OZET: Hangi etiketler hangi videolarda cikti")
tum_etiket = Counter()
for r in rapor:
    for e, n in r.get("etiketler", {}).items():
        tum_etiket[e] += 1
for e, n in tum_etiket.most_common():
    print(f"  {e}: {n} videoda")
print(f"\nToplam islenen: {len([r for r in rapor if 'hata' not in r])}/{len(videolar)}")
print("Rapor: data/output/kalite_raporu.json")
