# -*- coding: utf-8 -*-
"""Pipeline kalite testi: bir video uzerinde detayli cikti + sure + sema raporu."""
import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath("."))
from src.predict import run_inference
from src.labels import validate_results

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "data/input/test_hareket.mp4"
WEIGHTS = "weights/best_model.pt"

print(f"\n{'='*60}\nTEST VIDEO: {VIDEO}\n{'='*60}")
t0 = time.time()
sonuc = run_inference(VIDEO, WEIGHTS)
sure = time.time() - t0

hatalar = validate_results(sonuc)

print(f"\n--- ARAC BILGISI ---")
ab = sonuc["arac_bilgisi"]
print(f"  tip:   {ab['tip'] or '(yok)'}")
print(f"  renk:  {ab['renk'] or '(yok)'}")
print(f"  plaka: {ab['plaka'] or '(yok)'}")
print(f"  conf:  {ab['confidence_score']}")

print(f"\n--- TESPITLER ({len(sonuc['tespitler'])}) ---")
# kategori bazli grupla
from collections import Counter
kat_say = Counter((t["kategori"], t["etiket"]) for t in sonuc["tespitler"])
for (kat, etk), n in sorted(kat_say.items()):
    confs = [t["confidence_score"] for t in sonuc["tespitler"]
             if t["kategori"] == kat and t["etiket"] == etk]
    print(f"  [{kat}] {etk}: {n}x  (conf: {max(confs):.2f})")

print(f"\n--- KALITE METRIKLERI ---")
print(f"  Sure: {sure:.1f}s  (limit 600s/10dk -> {'OK' if sure < 600 else 'ASILDI'})")
print(f"  Sema hatasi: {len(hatalar)}  ({'GECERLI' if not hatalar else 'HATALI'})")
for h in hatalar:
    print(f"    ! {h}")
print(f"  Toplam tespit: {len(sonuc['tespitler'])}")
print(f"  Benzersiz etiket turu: {len(kat_say)}")

os.makedirs("data/output", exist_ok=True)
with open("data/output/results.json", "w", encoding="utf-8") as f:
    json.dump(sonuc, f, ensure_ascii=True, indent=2)
print(f"\n  -> data/output/results.json yazildi")
