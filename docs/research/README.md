# 📚 Deep-Research Notları

Bu klasör, modelin mimari kararlarını dayandırdığımız deep-research raporlarını içerir. Her doküman,
belirli bir alt problemi (kafa-pozu, veri seti seçimi, leakage) açık kaynak literatürü ve pratik
ROI/efor dengesiyle inceler. Kararların gerekçesi ana [README](../../README.md) içinde özetlenmiştir.

| Doküman | Konu | Ana çıktı |
|---|---|---|
| [`01_cv-iyilestirme-genel.md`](01_cv-iyilestirme-genel.md) | Bilgisayarlı görü iyileştirme yol haritası | En yüksek ROI: MediaPipe DMS + hazır YOLO ağırlıkları + hafif renk CNN |
| [`02_veri-setleri-rehberi.md`](02_veri-setleri-rehberi.md) | Açık kaynak veri setleri ve lisanslar | DMS için Roboflow/State Farm; plaka için YOLO+EasyOCR; lisans uyarıları |
| [`03_head-pose-gaze-dual-angle.md`](03_head-pose-gaze-dual-angle.md) | Ön + profil kamera için kafa-pozu / bakış | solvePnP yerine 6DRepNet; gaze için L2CS/MediaPipe Iris |
| [`04_yolov8-veri-agirliklari.md`](04_yolov8-veri-agirliklari.md) | Sürücü davranışı YOLOv8 veri/ağırlık | Sigara için dengeli (560/560) set + negatif kareler; kemer için CC BY setleri |
| [`05_leakage-free-distraction.md`](05_leakage-free-distraction.md) | Leakage'sız dikkat dağınıklığı tespiti | Frame-level leakage'i klip-ayrık split + ROI crop ile çöz; dürüst yeniden değerlendir |

> Bu notlar araştırma anındaki bulguları yansıtır; bir kaynak/lisans/ağırlık önerisini uygulamadan
> önce güncelliğini doğrulayın.
