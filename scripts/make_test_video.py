# -*- coding: utf-8 -*-
"""Ornek video yoksa sentetik ~10sn test videosu uretir (pipeline cokme testi)."""
import os
import cv2
import numpy as np

OUT = "data/input/video.mp4"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 480))
for i in range(250):  # 10 sn @ 25fps
    f = np.full((480, 640, 3), 60, np.uint8)
    cv2.rectangle(f, (200 + i % 50, 200), (400 + i % 50, 350), (200, 200, 200), -1)
    vw.write(f)
vw.release()
print(f"Sentetik video yazildi: {OUT}")
