# -*- coding: utf-8 -*-
"""Pipeline'i gercek agirlikla kostur, sema dogrula, sure olc."""
import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath("."))
from src.predict import run_inference
from src.labels import validate_results

VIDEO = "data/input/video.mp4"
WEIGHTS = "weights/best_model.pt"
os.makedirs("data/output", exist_ok=True)

t0 = time.time()
sonuc = run_inference(VIDEO, WEIGHTS)
sure = time.time() - t0

hatalar = validate_results(sonuc)
with open("data/output/results.json", "w", encoding="utf-8") as f:
    json.dump(sonuc, f, ensure_ascii=True, indent=2)

print(f"Sure: {sure:.1f}s | Tespit: {len(sonuc['tespitler'])} | Sema hatasi: {len(hatalar)}")
print("Arac:", sonuc["arac_bilgisi"])
for h in hatalar:
    print("  HATA:", h)
sys.exit(0 if not hatalar else 1)
