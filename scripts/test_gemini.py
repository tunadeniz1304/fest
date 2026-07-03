# -*- coding: utf-8 -*-
"""
geminitest/ videolarini sistemden gecirip tespit raporu uretir.
Beklenen davranis (overfit/yanlis-pozitif kontrolu):
  - goodmax: TEMIZ olmali (ihlal yok veya cok az)
  - badmax: telefon/sigara/esneme/bakma DOLU olmali
  - kemeri: sigara (+ kemer - modelimizde yok)
  - telefon+su: telefon + su_icme
"""
import os, sys, time, glob
sys.path.insert(0, os.path.abspath("."))
from src.predict import run_inference

KLASOR = "data/samples/geminitest"
WEIGHTS = "weights/best_model.pt"

videolar = sorted(glob.glob(os.path.join(KLASOR, "*.mp4")))
print(f"{len(videolar)} video test edilecek\n" + "="*60)

for v in videolar:
    ad = os.path.basename(v)
    print(f"\n>>> {ad}")
    t0 = time.time()
    try:
        r = run_inference(v, WEIGHTS)
        dt = time.time() - t0
        ab = r["arac_bilgisi"]
        print(f"    sure: {dt:.0f}sn | arac: tip={ab['tip'] or '-'} renk={ab['renk'] or '-'} plaka={ab['plaka'] or '-'}")
        if r["tespitler"]:
            for t in r["tespitler"]:
                print(f"      [{t['kategori']}] {t['etiket']} @ {t['zaman_saniye']}sn (conf {t['confidence_score']})")
        else:
            print("      (tespit yok - TEMIZ)")
    except Exception as e:
        import traceback
        print(f"    HATA: {e}"); traceback.print_exc()

print("\n" + "="*60)
print("DEGERLENDIRME: goodmax temiz mi? badmax dolu mu? yanlis-pozitif var mi?")
