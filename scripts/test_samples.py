# -*- coding: utf-8 -*-
"""
data/samples/ icindeki GERCEK videolari (zip degil) sistemden gecirir.
Bunlar Pexels gercek videolari - hem in-cabin (phone) hem trafik (vehicles).
Gercek-dunya davranisini gormek icin (Gemini sentetik degil).
"""
import os, sys, time, glob
sys.path.insert(0, os.path.abspath("."))
from src.predict import run_inference

KLASOR = "data/samples"
WEIGHTS = "weights/best_model.pt"

videolar = sorted(glob.glob(os.path.join(KLASOR, "*.mp4")))
print(f"{len(videolar)} gercek video test edilecek\n" + "="*60)

for v in videolar:
    ad = os.path.basename(v)
    print(f"\n>>> {ad}")
    t0 = time.time()
    try:
        r = run_inference(v, WEIGHTS)
        dt = time.time() - t0
        ab = r["arac_bilgisi"]
        print(f"    sure: {dt:.0f}sn | arac: tip={ab['tip'] or '-'} renk={ab['renk'] or '-'} plaka={ab['plaka'] or '-'} ({ab['confidence_score']})")
        if r["tespitler"]:
            for t in r["tespitler"]:
                print(f"      [{t['kategori']}] {t['etiket']} @ {t['zaman_saniye']}sn (conf {t['confidence_score']})")
        else:
            print("      (tespit yok)")
    except Exception as e:
        import traceback
        print(f"    HATA: {e}"); traceback.print_exc()

print("\n" + "="*60 + "\nBITTI")
