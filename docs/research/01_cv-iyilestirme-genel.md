# TEKNOFEST 2026 "5G & Yapay Zeka ile Akıllı Yol Güvenliği" — Bilgisayarlı Görü İyileştirme Araştırma Raporu

## TL;DR
- **En yüksek ROI (2 günde uygulanabilir):** (1) DMS'i **MediaPipe Face Mesh + EAR/MAR/PERCLOS + 6DRepNet head-pose** ile kur (yawning / looking around / looking backward / distraction); (2) **seatbelt + plaka tespiti** için açık-kaynak fine-tuned YOLO ağırlıklarını embed et; (3) vehicle color/type için **VCoR'da eğitilmiş hafif bir CNN sınıflandırıcı** ile HSV heuristic'i değiştir. Üçü de offline, T4'te gerçek zamanlı ve jüriye "ticari DMS mimarisini açık-kaynakla replike ediyoruz" diye anlatılabilir.
- **Riskli / ertele:** Video action recognition (SlowFast/X3D/TSM) ve lane/slalom deep learning (UFLD) — teknik olarak feasible ama 2 günlük takvimde offline ağırlık + T4 latency + entegrasyon riski yüksek. Bunları final tura bırak; slalom'u şimdilik ByteTrack trajectory'sinin lateral varyansından hesapla.
- **Hiçbir benchmark uydurma:** Aşağıdaki tüm sayılar kaynaklı ve ilgili makalenin **kendi test setine** ait. Senin verinde (in-cabin açı, TR plaka, gece, hava koşulları) domain gap nedeniyle DÜŞER — jüriye "bizim sistemimiz %97 yapar" diye sunma.

## Key Findings
- Ticari DMS pazarı dar: Colin Barnden (Principal Analyst, Semicast Research, EE Times) — *"More than 30 DMS companies are targeting the automotive sector, but just three—Cipia, Seeing Machines and Smart Eye—appear able to thrive."* Bu üçü iki katmanlı mimari kullanır: (1) face/landmark/head-pose tespiti, (2) bu sinyalleri fizyolojik duruma (drowsiness/distraction) çeviren ikinci katman. Bunu açık kaynakla taklit edebilirsin ve bu, jüri için güçlü bir "innovation" hikâyesidir.
- EAR/MAR/PERCLOS + head pose klasik, güvenilir ve **eğitim gerektirmez**; MediaPipe Face Mesh CPU'da bile gerçek zamanlı çalışır → 2 günde kesinlikle feasible ve en yüksek innovation/effort oranı bu.
- Seatbelt ve TR plaka için hazır fine-tuned YOLO ağırlıkları + açık datasetler mevcut; embed etmek COCO-only baseline'a göre büyük precision/recall artışı sağlar.
- Aspect-ratio heuristic'i sedan/suv/hatchback ayrımında çok zayıf; hafif bir CNN sınıflandırıcı devreye almak en kolay "kazanılmış" doğruluk artışıdır.
- Video action recognition single-frame'e göre daha doğru ama T4 + 10dk + offline + 8GB kısıtında 2 güne sığmaz.

## Details

### Topic 1 — Driver Monitoring Systems (DMS)

**Ticari mimari (jüri sunumu için):** Sadece üç firma (Cipia, Seeing Machines, Smart Eye) ticari olarak öne çıkıyor (Barnden/Semicast, EE Times). Seeing Machines kendi **Fovio** çipini ve **Occula** NPU'sunu kullanıyor; GM Super Cruise ile Mercedes S-Class/EQS'e DMS yazılımı sağlıyor. Cipia (eski adıyla Eyesight Technologies — otomotiv varlıkları Haziran 2025'te HARMAN International tarafından satın alındı) **iki katmanlı AI** kullanıyor: katman-1 yüz tespiti + head pose + göz açıklığı + gaze; katman-2 bunları drowsiness/distraction'a çeviriyor; ayrıca seatbelt, telefon ve sigara da tespit ediyor. Donanım tarafında ileri sistemler EE Times'a göre *"operate at 940 nm, with a frame rate of 60 frames per second... an alternating, strobing illumination pattern operating at 60 Hz"* kullanır. Sunumda "biz bu iki-katmanlı mimariyi açık-kaynakla kuruyoruz" demek innovation puanı kazandırır.

**Açık kaynak teknikler:**
- **EAR (Eye Aspect Ratio):** Soukupová & Čech (2016). Göz açıkken tipik 0.28–0.35, blink'te keskin düşer. Pratik eşik ~0.22–0.25.
- **MAR (Mouth Aspect Ratio):** Yawning için; MAR eşiği üstünde 15+ ardışık frame = doğrulanmış yawn.
- **PERCLOS:** NHTSA standardı; 60 saniyelik kayan pencerede gözün %80+ kapalı olduğu frame oranı. >%15 = drowsy.
- **Head pose:** **6DRepNet** (Hempel, Abdelrahman & Al-Hamadi, ICIP 2022, arXiv 2202.12555). Makale: *"Experiments on the public AFLW2000 and BIWI datasets demonstrate that our proposed method significantly outperforms other state-of-the-art methods by up to 20%."* `pip install SixDRepNet`. Looking backward / looking around için head **yaw** açısı; microsleep nod için **pitch** kullanılır.
- **MediaPipe Face Mesh:** 468 (iris ile 478) 3D landmark, tek RGB kameradan, CPU'da gerçek zamanlı (mobil GPU'da 50–1000 FPS aralığı raporlanıyor). EAR/MAR/gaze için landmark indeksleri hazır (örn. 33/263 dış göz köşeleri, 13/14 üst/alt dudak).

**Benchmark (kaynaklı, dikkatli yorumla):**
- Ecuador DMS çalışması (PMC12899127): MobileNetV2 (RAF-DB) + MediaPipe 468 landmark ile **distraction %100, yawning %85.19, eye closure %88.89 accuracy** — fakat **sadece 27 katılımcı, controlled ortam**; gerçek dünyada düşer.
- **DMD (Driver Monitoring Dataset, Vicomtech, ECCV 2020 Workshops):** 37 sürücü, 41 saat RGB/depth/IR, 3 kamera (face/body/hands), distraction + gaze allocation + drowsiness + hands-wheel annotation'ları. dBehaviourMD alt-seti 13 distraction aktivitesi içeriyor. Offline test/validasyon için ideal.
- **SynDD1/SynDD2 (Iowa State, NVIDIA ortaklı):** distracted behavior + gaze zone, dashboard/rearview/window kameraları, "with/without appearance blocks" (şapka/gözlük) varyasyonları.

**Feasibility:** ✅ **FEASIBLE (2 gün).** MediaPipe + 6DRepNet offline çalışır, ağırlıklar küçük (6DRepNet ~tens of MB), T4'te hızlı. EAR/MAR/PERCLOS saf geometri → eğitim yok. **Sistemin en yüksek innovation/effort oranı burada.**

**Linkler:** 6DRepNet → github.com/thohemp/6DRepNet (DOI 10.1109/ICIP46576.2022.9897219) | 6DRepNet360 (TIP 2024) → github.com/thohemp/6DRepNet360 | MediaPipe → github.com/google-ai-edge/mediapipe | DMD → github.com/Vicomtech/DMD-Driver-Monitoring-Dataset

### Topic 2 — Seatbelt Violation Detection

**Yaklaşım:** İki aşamalı en iyi: önce windshield/ROI tespit, sonra ROI içinde seatbelt sınıflandırma (gürültüyü düşürür).

**Benchmark (kaynaklı):**
- YOLOv7, overhead traffic surveillance (ResearchGate 399255010): görülmemiş test setinde **mAP@50 %97.46, F1 %95.37** (gündüz+gece, data augmentation + anchor optimization).
- DW-YOLOv8 (KorkanaRahul): windshield (YOLOv5) → seatbelt (depthwise-conv YOLOv8) iki aşamalı, Hindistan trafiği için hafifletilmiş.
- "Real time seatbelt detection using YOLO" (ResearchGate 369406683): bildirilen precision %94.1, recall %98.2, mAP %98.8 (not: bu sayılar araç tespit bileşenine ait; seatbelt sınıfı için ayrı doğrulanmalı).

**Datasetler:** Roboflow seatbelt-detection (seatbelttraining): **3.489 image**; College "seatbeltYolov5"; fyp "Seatbelt dataset": **900 image**. Hepsi YOLOv5/v8/v11 formatında export edilebilir.

**Feasibility:** ✅ **FEASIBLE (2 gün)** ama uyar: bu datasetler çoğunlukla **dıştan/üstten (overhead) trafik kamerası açısı**. TEKNOFEST in-cabin açısıysa domain gap olur. Yine de fine-tuned ağırlık embed etmek, COCO'da olmayan seatbelt sınıfı için baseline'dan kesinlikle iyidir.

**Linkler:** github.com/KorkanaRahul/Seatbelt-Detection-Using-DWYOLOv8-Model | universe.roboflow.com/seatbelttraining-7yh0f/seatbelt-detection-lb1ec

### Topic 3 — Turkish License Plate Recognition (TR ALPR)

**Datasetler (kaynaklı):**
- Roboflow "License Plates of Vehicles in Turkey" (kemalkilicaslan): **3.501 image**.
- Roboflow "Turkish Number Plates" (plakatanima): **2.246 image**, CC BY 4.0, pre-trained model + API.
- Roboflow "TR PLAKA DATASET" (MEWI): **1.426 image**.
- Kaggle "Synthetic Turkish License Plates" (tustunkok): **100.000 sentetik image (~2 GB)** — OCR pre-training için çok değerli.
- Kaggle "Turkish License Plate Dataset" (smaildurcan): tek-sınıf, YOLO formatı (kesin görsel sayısı Kaggle sayfasında doğrulanmalı).

**Modeller & Benchmark (kaynaklı):**
- **utkuatasoy/License-Plate-Recognition-System:** 52.201 image hybrid dataset, **YOLOv11x detection mAP@50 0.98466, mAP@50-95 0.71605** (20 epoch); best Precision(B) 0.97384, best Recall(B) 0.96591. OCR = EasyOCR. (Not: TR datasetlerini de içeren hybrid set.)
- **Semihocakli/turkish-plate-recognition-w-yolov8-onnx-to-engine-cpp:** YOLOv8 + ONNX→TensorRT, C++, TR plaka, gerçek-zaman optimize.
- **Akademik (Kilic & Aydin, TR plaka, deep learning):** car detection **%97**, plate localization **%98**, character recognition **%90**.
- **TR karakter segmentasyon (Üstünkök et al.):** letter-digit ifade üretimi **%99.28 accuracy**, blob segmentasyon %96.12; "yalnızca Türk plakalarında çalışır."
- **Benchmarking ALPR (arXiv 2203.14298):** sentetik TR plaka (1.000 test), LPRNet (10k TR plakada retrain) **%88.6**, Tesseract **%93.3** mean accuracy.

**OCR seçimi — EasyOCR vs PaddleOCR vs fast-plate-ocr:**
- **fast-plate-ocr (ankandrew) — ÖNERİLEN.** `global-plates-mobile-vit-v2-model` (MobileViT-2 backbone, 65+ ülke, 85k+ plaka) **plate_acc %93.3**; `european-plates-mobile-vit-v2-model` **%92.5**; `argentinian-plates-cnn-model` %94.05. v2 modellerin region head'i 114.000+ örnekli held-out sette **>0.99 val_region_macro_f1** raporluyor. CCT modelleri RTX 3090'da **0.3–0.7 ms** (cct-xs-v2 ~2.144 plate/sn). ONNX, tamamen offline. EasyOCR'dan hem hızlı hem de TR formatına regex + region head ile uyarlanabilir. `fast-alpr` çerçevesi detektör + OCR'ı tek pakette birleştirir (varsayılan detektör: yolo-v9-t-384-license-plate-end2end).
- **PaddleOCR:** detection branch'i kapatıp recognition'a custom TR karakter listesi + angle classification verince rotated plakalarda EasyOCR'dan iyi (TED Üniversitesi ADA447 projesi). Genel literatür: Google Vision en iyi, PaddleOCR "en güçlü açık-kaynak alternatif."
- **EasyOCR (mevcut baseline):** bir IEEE çalışmasında plaka datasetinde **>%95** vs Tesseract %90. CRNN tabanlı, 80+ dil. Düşük çözünürlük/yansıma/açıda preprocessing'e bağımlı.

**İyileştirme teknikleri:**
- **Perspective correction (4-point / homography):** "Vehicle License Plate Detection and Perspective Rectification" çalışması perspective rectification ile **%3 recognition accuracy** artışı bildiriyor. OpenCV `getPerspectiveTransform` + plaka köşe tespiti. ✅ ucuz, hızlı, feasible.
- **Super-resolution:** MDPI (Mathematics 13/10/1673) SwinFIR + perceptual loss (Swin Transformer + DISTS) ile düşük çözünürlüklü plakada OCR accuracy **%85.14** (%9.75 iyileşme). LCDNet (arXiv 2408.15103) layout-aware focal loss. Ama SR modeli embed + T4 latency riski → orta. **Sadece düşük-res plakalarda koşullu uygula.**

**Feasibility:** ✅ **FEASIBLE:** fast-plate-ocr'a geçiş + perspective correction (2 gün). ⚠️ **RİSKLİ:** super-resolution (latency + 8GB image bütçesi).

**Linkler:** github.com/ankandrew/fast-plate-ocr | github.com/ankandrew/fast-alpr | github.com/utkuatasoy/License-Plate-Recognition-System | arxiv.org/abs/2408.15103

### Topic 4 — Fine-Grained Vehicle Classification

**Heuristic'in problemi:** Aspect-ratio ile sedan/suv/hatchback ayrımı çok zayıf — bu sınıflar benzer en-boy oranına sahip. Hafif bir CNN sınıflandırıcı baseline'dan belirgin biçimde iyi.

**Benchmark (kaynaklı):**
- **EfficientNetV2S:** Stanford Cars **%89.8**, VehicleID **%99.2** (make/model) — lightweight ve T4'e uygun.
- **Stanford Cars genel SOTA:** fine-tuned CNN'lerle %92–97 top-1 (Krause et al. %97.10 manuel bbox / %92.14 otomatik bbox; Sighthound %93.6; ABNet ResNet-50 %94.6).
- **VCoR (Vehicle Color Recognition):** "Veri-Car" (arXiv 2411.06864) Table 3 — Multi-Similarity color modeli Kaggle VCoR (15 renk sınıfı) üzerinde **Precision@1 %90.55**. Senin 9 renk sınıfına doğrudan uyarlanabilir.
- VMMRdb + Stanford Cars birleşik transfer learning (Pells31 repo): donmuş ImageNet backbone + son katman fine-tune.

**Strateji:** YOLOv8m araç bbox'ını kes → crop'u hafif EfficientNet/MobileNet sınıflandırıcıya ver (type + color, iki ayrı baş veya iki model). **Renk: HSV yerine CNN.** VCoR çalışması gece sahnelerin en çok hatayı ürettiğini gösteriyor; HSV gece/gölgede çöküyor, öğrenilmiş özellikler daha dayanıklı.

**Feasibility:** ✅ **FEASIBLE (2 gün)** color için (VCoR ağırlığı bul/eğit, model küçük). ⚠️ **ORTA RİSK** type için: 7 TEKNOFEST sınıfı (minibus/panelvan/pickup) Stanford Cars'ta yok → kendi class-mapping/küçük eğitim gerekebilir. Stanford Cars'ı type'a değil make/model'e göre kategorize ettiği için, body-type'a remap eden bir ara katman ya da COCO + VMMRdb karması gerekir.

**Linkler:** VCoR → Kaggle "Vehicle Color Recognition (VCoR) Dataset" | github.com/Pells31/Vehicle-Make-and-Model-Recognition | github.com/morrisfl/stanford_cars_refined (CLIP ConvNeXt-B ile color-refined Stanford Cars)

### Topic 5 — Lane / Slalom (Lane Departure) Detection

**Klasik vs DL:**
- **Klasik (Hough + Canny):** ucuz, eğitim yok; ama dazzle/gölge/gece'de kötü, eğrilerde başarısız.
- **Ultra-Fast-Lane-Detection (UFLD, ECCV 2020, cfzd):** TuSimple accuracy **~%95.9** (ResNet-18), CULane F1 ~%72; row-anchor classification yaklaşımıyla ResNet-18'de 300+ FPS. UFLD-v2 (TPAMI 2022) daha iyi. TensorRT portları mevcut (KopiSoftware/TRT_Ultra_Fast_Lane_Detect).
- **LaneNet:** embedding + semantic segmentation (TuSimple yarışma kazananı SCNN ile birlikte referans).

**Slalom mantığı:** Lane çizgilerini tespit et → aracın lane merkezine göre lateral pozisyonunu zamanla takip et → weaving/oscillation paterni = slalom. Bu, lane detection üstüne ek bir temporal/sinyal-işleme mantığı gerektirir ve ego-motion'dan ayrıştırma ister.

**Feasibility:** ⚠️ **RİSKLİ → muhtemelen ERTELE.** UFLD offline çalışır ve T4'te hızlıdır, ama: (1) in-vehicle kameradan lane net görünmeyebilir; (2) slalom mantığı ego-motion'dan ayrıştırma gerektirir; (3) 2 günde entegrasyon + tuning zor. **Önerilen kısa yol:** Roadside kamera senaryosunda aracın trajectory'sini zaten ByteTrack'tan alıyorsun — slalom'u track'in **lateral pozisyon varyansı / sıfır-geçiş (zig-zag) sayısından** çıkarmak çok daha basit, eğitimsiz ve daha az riskli. Bunu önce dene; UFLD'yi final tura sakla.

**Linkler:** github.com/cfzd/Ultra-Fast-Lane-Detection | github.com/cfzd/Ultra-Fast-Lane-Detection-v2

### Topic 6 — Video Action Recognition

**Single-frame vs temporal:** Mevcut yaklaşımın (COCO object + action-frame-ratio + track-level majority vote) aslında zaten hafif bir temporal voting. Gerçek action recognition (drinking, talking, yawning) hareketi modellediği için single-frame'den daha doğru olabilir — ama maliyeti yüksek.

**Benchmark (kaynaklı):**
- **State Farm Distracted Driver:** CNN/genetik ensemble **%99.75–%99.8 accuracy** — ⚠️ bu rakamlar **şişirilmiş**: aynı sürücünün ardışık frame'leri train/test'e karışıyor (literatür açıkça eleştiriyor; IET ITS 2023, Wang et al.). **Cross-driver split'te %85–92'ye düşer.**
- **AUC Distracted Driver:** genetik ağırlıklı 6-model ensemble **%96.37** (arXiv 2107.13355); MobileNetV2-tiny lightweight versiyon AUC'de orijinalden %1.63 daha yüksek, %78 parametre.
- **X3D (Feichtenhofer, CVPR 2020, arXiv 2004.04730):** X3D-M Kinetics-400 **top-1 %74.6, top-5 %91.7, sadece 4.73 GFLOPs**; makale: *"X3D-M is comparable to SlowFast 4×16, R50... while having 4.7× fewer FLOPs and 9.1× fewer parameters."* X3D-XS %68.6 @ 0.60 GFLOPs.
- **SlowFast R101 16×8:** Kinetics %78.7 ama **215.6 GFLOPs** (T4 için ağır).
- **TSM (Temporal Shift Module, arXiv 2109.13227):** X3D'den 3.4× hızlı; ir-CSN-152'ye yakın accuracy ama daha hızlı → edge için en iyi accuracy-speed trade-off.

**Strateji:** Temporal'a geçeceksen **X3D-M veya TSM** seç (SlowFast'tan kaçın — çok ağır). State Farm/AUC'de pretrain + driver crop. Ama T4'te her track için video-clip inference + 10 dk limit ciddi bütçe yer.

**Feasibility:** ⚠️ **RİSKLİ → ERTELE (final tur).** 2 gün için asıl quick win driver action'larında: MediaPipe (yawning/distraction geometrik) + COCO object (phone/bottle/laptop) + 6DRepNet head pose + iyi temporal voting. 3D-CNN/video-transformer entegrasyonu 2 güne sığmaz ve T4 latency + 8GB image riski yüksek.

**Linkler:** github.com/facebookresearch/SlowFast (PySlowFast — X3D, SlowFast dahil) | State Farm → Kaggle "State Farm Distracted Driver Detection" | AUC Distracted Driver dataset (Eraqi/Abouelnaga)

---

## Birleşik Öncelik Tablosu — En Yüksek Etkili 5 İyileştirme

| # | İyileştirme | Etki (Impact) | Efor (2 gün) | Risk | Neden / Benchmark dayanağı |
|---|------------|---------------|--------------|------|----------------------------|
| **1** | **DMS katmanı:** MediaPipe Face Mesh + EAR/MAR/PERCLOS + 6DRepNet head-pose (yawning, looking around/backward, distraction) | **ÇOK YÜKSEK** — 4+ davranış sınıfını tek seferde, COCO'nun yapamadığı şekilde kapsar | Düşük-Orta | **Düşük** — offline, eğitimsiz geometri + küçük ağırlık | Ecuador DMS: distraction %100/yawning %85.19/eye-closure %88.89 (controlled); 6DRepNet AFLW2000/BIWI'de SOTA +%20; ticari DMS'lerin (Cipia/Seeing Machines) iki-katmanlı mimarisi |
| **2** | **TR ALPR upgrade:** fast-plate-ocr (global-plates-mobile-vit-v2, plate_acc %93.3) + 4-point perspective correction | **YÜKSEK** — plaka recall/karakter doğruluğu artar; EasyOCR'dan hızlı | Düşük | **Düşük** — ONNX, offline, TR regex+region | fast-plate-ocr %93.3 plate_acc, 0.3–0.7ms; perspective rectification +%3 accuracy |
| **3** | **Seatbelt fine-tuned YOLO** ağırlığı embed (windshield→seatbelt iki aşama) | **YÜKSEK** — COCO'da seatbelt yok; sıfırdan kazanılan sınıf | Düşük | **Orta** — overhead dataset açısı vs in-cabin domain gap | YOLOv7 overhead: mAP@50 %97.46, F1 %95.37; Roboflow 3.489-image dataset |
| **4** | **Vehicle color CNN** (VCoR Prec@1 %90.55) ile HSV heuristic'i değiştir; type için EfficientNet-lite | **ORTA-YÜKSEK** — HSV gece/gölgede çöküyor; renk 9-sınıf doğruluğu artar | Düşük (color) / Orta (type) | **Düşük** (color) / **Orta** (type sınıf mapping) | VCoR color Prec@1 %90.55; EfficientNetV2S Stanford Cars %89.8 |
| **5** | **Slalom = ByteTrack trajectory lateral varyansı** (deep lane detection yerine) | **ORTA** — slalom sınıfını eğitimsiz, ucuz kazanır | Düşük | **Düşük** — mevcut tracker çıktısı; ego-motion'a dikkat | UFLD TuSimple %95.9 ama in-cabin/entegrasyon riski → trajectory-tabanlı kısa yol önerilir |

**Erteleme eşikleri (final tura bırak):** Video action recognition (X3D-M/TSM) — ancak baseline kararlı ve T4 latency bütçen 10 dk'da rahatsa. Lane DL (UFLD) — ancak roadside kamera + lane net görünüyorsa. Plate super-resolution — ancak düşük-res plaka recall'un ölçülebilir biçimde düşükse, **koşullu** (sadece küçük bbox'larda tetiklenen) pipeline olarak.

**Innovation anlatımı (jüri için):** "Ticari DMS'lerin (Cipia/HARMAN, Seeing Machines) iki-katmanlı mimarisini açık kaynakla replike ettik: katman-1 landmark + head-pose (MediaPipe + 6DRepNet), katman-2 fizyolojik durum çıkarımı (EAR/MAR/PERCLOS eşikleri + zamansal füzyon). Plaka için perspective-rectification + region-aware OCR (fast-plate-ocr), renk için HSV yerine öğrenilmiş özellikler kullandık. Slalom'u ayrı bir ağ yerine mevcut ByteTrack trajectory'sinden türettik — düşük maliyetli, açıklanabilir bir sinyal."

## Recommendations (öncelik sıralı, 2 gün)

**Gün 1**
1. **DMS katmanını kur** (MediaPipe Face Mesh + EAR/MAR/PERCLOS + 6DRepNet). Driver crop'u ByteTrack track'inden al; yawning, looking around/backward, distraction bunu kullanır. → En yüksek impact.
2. **fast-plate-ocr entegrasyonu + perspective correction** (OpenCV 4-point). EasyOCR'ı değiştir ya da ensemble yap; TR regex + region filtresi.
3. **Seatbelt fine-tuned YOLO ağırlığını embed et** (Roboflow seatbelt dataset'inde eğitilmiş).

**Gün 2**
4. **Vehicle color CNN** (VCoR'da eğitilmiş hafif model) → HSV heuristic'i değiştir. Type için EfficientNet-lite + TEKNOFEST sınıf mapping (mümkünse).
5. **teknocan:** YOLOE prompt-free zaten iyi seçim; birkaç örnek görselle **visual prompt (SAVPE)** kullanırsan accuracy artar (YOLOE T4'te ölçülmüş, prompt-free LRPC ile çalışır).
6. **Slalom:** ByteTrack trajectory'sinin lateral pozisyon varyansı / zig-zag sayısından hesapla; deep lane detection'a girme.

## Caveats
- **Benchmark'lar domain-specific:** Yukarıdaki tüm accuracy/mAP rakamları ilgili makalelerin kendi test setlerinde. Senin TEKNOFEST verinde (in-cabin açı, TR plaka, gece, kar/yağmur) bu rakamlar DÜŞER. Hiçbirini "bizim sistemimiz %X yapar" diye sunma; "literatürde bu yöntem şu sette %X bildirmiş" de.
- **State Farm %99 şişirme:** Random split yüzünden; cross-driver'da %85–92. Jüriye dürüst ol, bu farkındalık puan kazandırır.
- **Seatbelt / lane datasetleri açı uyumsuzluğu:** Çoğu overhead/dıştan; in-cabin domain gap riski yüksek.
- **Offline kısıt:** MediaPipe, fast-plate-ocr (ONNX), 6DRepNet, YOLOE, VCoR/EfficientNet ağırlıklarının HEPSİNİ Docker image'a embed et; runtime'da hiçbir model/torch.hub indirme olmamalı. **8GB image limitine dikkat** — PyTorch + CUDA + birden fazla model şişebilir; gereksizleri ONNX'e çevirip torch bağımlılığını azaltmayı düşün.
- **Anti-cheat:** Environment detection (internet/GPU/test ortamı tespiti) yasak; sadece saf inference kodu.
- **TEKNOFEST format belirsizliği:** Bu yarışma (5G & Yapay Zeka ile Akıllı Yol Güvenliği, Turkcell yürütücü, ilk kez 2026) yeni; kesin teslim formatı şartnamede netleşecek. Geçmiş "Ulaşımda Yapay Zeka" yarışmasında frame'ler 7.5 fps ile veriliyor ve her frame için sonuç JSON olarak sunucuya gönderiliyordu — benzer bir akış beklenebilir, ama resmi şartnameyi teyit et.