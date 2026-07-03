# TEKNOFEST 2026 "5G & Yapay Zeka ile Akıllı Yol Güvenliği" — Açık Kaynak Veri Seti Rehberi

## TL;DR
- **DMS (sürücü davranışı) modülü için en pratik birleşim, hazır bir Roboflow "Driver Monitoring/DMS" YOLO seti (doğrudan bbox fine-tune) + State Farm Distracted Driver'dır (Kaggle, tam 22,424 etiketli eğitim görseli + 79,726 test görseli, ~4 GB, 10 sınıf);** DMD (Vicomtech) en zengin akademik settir fakat 2026'da verisi CC BY-NC-ND 4.0 (yalnız akademik, ticari kullanım YASAK) olduğu için yarışma çıktınız ticari sayılıyorsa kullanmayın.
- **Plaka için YOLO ile lokalizasyon (Roboflow "License Plate Recognition", tam 10,125 görsel, CC BY 4.0, ön-eğitimli model mAP@50 %97.8) + EasyOCR (Türkçe `tr` + `allowlist` + TR regex) en yüksek değer/efor oranını verir;** karakter okuma için sıfırdan eğitim yerine hazır OCR + regex doğrulaması (`34ABC123`) önerilir.
- **2 gün + offline + tek T4 kısıtında sıfırdan multi-task eğitim gerçekçi değildir;** önceden eğitilmiş ağırlıkları indirip (YOLOv8m COCO + hazır DMS/plaka YOLO ağırlıkları) hafif fine-tune ve YOLO-World/CLIP ile few-shot prompt yaklaşımı en düşük riskli yoldur.

## Key Findings

Modüler mimariniz (YOLOv8m COCO backbone + hafif sınıflandırma başlıkları + ayrı OCR + MediaPipe Face Mesh + YOLO-World) doğru ve açık-kaynak SOTA ile uyumludur. Her görev için ayrı, indirilebilir ve lisansı uygun veri setleri mevcuttur. En büyük lisans riski **DMD (CC BY-NC-ND 4.0, akademik) ve AUC (özel lisans, onay gerekir)**; en güvenli olanlar **Roboflow CC BY 4.0** setleri ve **State Farm yarışma verisidir**. COCO-pretrained YOLOv8m zaten `cell phone`, `bottle`, `cup`, `laptop` sınıflarını içerdiği için in-cabin nesnelerin büyük kısmı sıfır eğitimle gelir.

## Details

### A) SÜRÜCÜ DAVRANIŞLARI (DMS) — EN KRİTİK

| Dataset | Erişim linki | Boyut | Etiket tipi/format | Lisans | Kamera açısı/senaryo | Benchmark (kaynaklı) | T4/2-gün uygunluk |
|---|---|---|---|---|---|---|---|
| **State Farm Distracted Driver** | kaggle.com/c/state-farm-distracted-driver-detection | Tam **22,424 etiketli eğitim + 79,726 etiketsiz test görseli (toplam 102,150), ~4 GB, 480×640 RGB**; driver_imgs_list.csv 22,424 satır; sınıflar dengeli (1,911 "hair and makeup" – 2,489 "safe driving") | Sınıflandırma, 10 sınıf (c0 güvenli, c1/c3 texting, c2/c4 telefon, c5 radyo, c6 içme, c7 arkaya uzanma, c8 saç/makyaj, c9 yolcu) | Kaggle competition rules (yarışma içi kullanım; ticari kullanım kısıtlı) | In-cabin, yan/dashboard açısı, gündüz, park halinde çekim | MobileNetV3/Inception ile %98+ test acc bildirilmiş (DebuggerCafe; GW blog) | Çok uygun — sınıflandırma başlığı fine-tune'u T4'te 1-2 saat |
| **AUC Distracted Driver (V1/V2)** | heshameraqi.github.io/data/auc.distracted.driver.dataset (Kaggle mirror: tejakalepalle/auc-distracted-driver-dataset-v1) | 14,478 frame, ~2.3 GB | Sınıflandırma, 10 sınıf (State Farm benzeri) | Özel lisans — MI-AUC grubundan onay gerekir; "freely for this purpose" ama ticari için izin şart | In-cabin, yan açı | TML %96.3 acc (AUC); cross-dataset Drive&Act %66.9 | Uygun ama lisans onayı gerektiği için yarışmada riskli |
| **DMD (Driver Monitoring Dataset, Vicomtech)** | dmd.vicomtech.org ; github.com/Vicomtech/DMD-Driver-Monitoring-Dataset | **37 sürücüden 41 saat video, 3 kamera (yüz/gövde/eller), ~42 TB ham veri (Ortega et al., arXiv:2008.12085, 2020)**; 2026 revizyonunda simülatör + IR + Depth KALDIRILDI: "Simulator recordings, IR, and Depth streams have been removed. Only RGB material from the real-car scenario is currently available" | Video + OpenLABEL temporal anotasyon; dBehaviourMD alt-seti 13 distraction sınıfı; DEx aracı ile YOLO/sınıflandırma export | **Veri: CC BY-NC-ND 4.0 (yalnız akademik, ticari YASAK, türetme dağıtımı yasak); Araçlar (TaTo/DEx): MIT** | In-cabin, 3 kamera, gündüz | dBehaviourMD ile CPU-only gerçek-zaman sistem (CVIU 2025, doi:10.1016/j.cviu.2025.104593) | Orta — video → frame export emek ister; lisans non-commercial |
| **SynDD1 / SynDD2** | arxiv.org/abs/2204.08096 ; SynDD1 doi:10.1016/j.dib.2022.108793 | Çok video, 3 kamera (dashboard, dikiz aynası, sağ köşe), InfraRed videolar | Video + CSV temporal anotasyon (distracted activities + gaze zones; şapka/gözlük blokları) | Açık (NHTSA/Iowa State, AI City Challenge) — açık erişim | In-cabin, 3 açı, **IR dahil** | DeepLocalization SynDD2: %57.5 event classification, %51 event detection | Orta — temporal video, frame çıkarımı gerekir |
| **NTHU-DDD** | cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD | 36 sürücü, ~9.5 saat, 640×480 AVI | Frame-level drowsiness + göz/ağız/baş etiketi (drowsy/non-drowsy) | Lisans anlaşması imzası + e-posta ile erişim; akademik | In-cabin dashboard, **aktif IR (gece) + gündüz**, gözlük/güneş gözlüğü | LMDF %90.05 eval acc; iki-akışlı ağlar | Orta — IR/gece kapsamı değerli; erişim formu yavaş |
| **YawDD** | ieee-dataport.org/open-access/yawdd-yawning-detection-dataset | 322 + 29 video, ~4.9 GB | Video (normal/konuşma/esneme); türev YOLO seti (Roboflow utarlddv1, 2,160 görsel); sınıflandırma türevi (HippoYD) | Akademik (atıf şartı, ACM MMSys 2014); IEEE DataPort open-access | In-cabin: (1) dikiz aynası altı, (2) dashboard; gündüz, değişken ışık | Türev YOLO setlerinde yerel testlerde mAP ~0.99 | Esneme/ağız modülü için çok uygun; küçük |
| **Driver Drowsiness "yawn-eye" (Kaggle, serenaraju)** | kaggle.com/datasets/serenaraju/yawn-eye-dataset-new | ~2,900 görsel, 4 sınıf | Sınıflandırma (Closed, Open, yawn, no_yawn) | Kaggle public (lisans belirtilmemiş — dikkat) | In-cabin yüz/göz kırpımı | InceptionV3 ile %93+ (MDPI Appl. Sci. 2025) | Çok uygun — hafif, hızlı |
| **MRL Eye Dataset** | kaggle.com/datasets/prasadvpatil/mrl-dataset | Büyük göz açık/kapalı seti | Sınıflandırma (open/closed eyes) | Akademik (MRL) | In-cabin göz kırpımı, IR dahil | PERCLOS/göz-kapama için kullanılır | Çok uygun |
| **FL3D (Frame Level Driver Drowsiness)** | kaggle.com/datasets/matjazmuc/frame-level-driver-drowsiness-detection-fl3d | Frame-level | Sınıflandırma (alert, microsleep, yawning) | Kaggle public | In-cabin | — | Uygun |
| **Roboflow "Driver Monitoring/DMS" YOLO setleri** | universe.roboflow.com (ör. "DRIVER MENTORING" 8,047 görsel: drinking/drowsy/seatbelt/smoking/Distracted Phone; "Driver Monitoring System" 2,099 görsel) | 2k–9.5k görsel | **YOLO (bbox) — telefon, sigara, içme, emniyet kemeri, esneme, göz açık/kapalı** | Çoğunlukla CC BY 4.0 (her proje ayrı kontrol) | In-cabin, çeşitli açılar | mAP proje bazlı değişken | **En uygun** — doğrudan YOLOv8 fine-tune, hazır ağırlık var |

**Hangisi yarışma senaryosu için en uygun?** In-cabin + gündüz/gece + değişen açı kriterinde tek bir set yeterli değildir. **Önerilen birleşim:** (1) Bir Roboflow DMS YOLO seti (telefon/sigara/içme/kemer bbox için, hazır YOLOv8 ağırlığı) + (2) State Farm (geniş RGB sınıflandırma omurgası) + (3) gece/IR varyasyonu için NTHU-DDD veya SynDD (IR videolar) + (4) esneme/göz için YawDD türevi veya yawn-eye. DMD'yi yalnızca yarışma ticari değilse ekleyin. Drowsiness/PERCLOS için MediaPipe Face Mesh EAR (Eye Aspect Ratio) + MAR (Mouth Aspect Ratio) heuristic'i, etiketli veri ihtiyacını büyük ölçüde azaltır.

### B) IN-CABIN NESNELER (sigara, telefon, içecek, laptop)

| Dataset | Erişim linki | Boyut | Format | Lisans | Senaryo | Benchmark | T4 uygunluk |
|---|---|---|---|---|---|---|---|
| **COCO (YOLOv8m hazır)** | docs.ultralytics.com | 80 sınıf (`cell phone`, `bottle`, `cup`, `laptop` dahil) | YOLO/COCO, hazır ağırlık | Ultralytics AGPL-3.0 (ağırlıklar) | Genel | COCO mAP | **Sıfır eğitim** — telefon/şişe/bardak/laptop zaten var |
| **Smoker YOLO (Roboflow cigaretteple)** | universe.roboflow.com/cigaretteple-7m0hn/smoker-yolo | 4,100 görsel | YOLO (1 sınıf: sigara) | Roboflow (proje bazlı) | Genel sahne | YOLOv8 mAP ~0.93 (UNNES çalışması, sigara+paket) | Çok uygun |
| **Smoking & Drinking Detection (Roboflow)** | universe.roboflow.com/yolo-dataset-rtznj/smoking-and-drinking-detection | 1,030 görsel | YOLO (smoking, drinking) | Roboflow | Genel | — | Uygun |
| **Mobile Phone Dataset (Kaggle DataClusterLabs)** | kaggle.com/datasets/dataclusterlabs/mobile-phone-image-dataset | 3,000+ HD görsel | COCO/YOLO/VOC | DataClusterLabs (geniş set ticari lisanslı; örnek açık) | Genel, çeşitli ışık | — | Uygun |
| **Markalı kutu/teneke (ör. "teknocan") — YOLO-World / few-shot** | github.com/AILab-CVC/YOLO-World ; docs.ultralytics.com/models/yolo-world | — | Open-vocabulary, metin veya görsel-örnek prompt | YOLO-World açık (GPLv3); Ultralytics implementasyonu | Genel | LVIS zero-shot | **En uygun** — referans görseli + metin prompt ile sıfır eğitim |

**Markalı kutu için yaklaşım:** Birkaç referans görsel toplayıp (a) YOLO-World metin prompt'u ("a branded soda can", "teknocan can"), (b) YOLOE/SAVPE görsel-prompt few-shot, veya (c) CLIP embedding + nearest-neighbor ile few-shot sınıflandırma kullanın. Sıfırdan eğitime gerek yok; offline için prompt embedding'lerini önceden hesaplayıp gömün ("prompt-then-detect" — YOLO-World offline vocabulary).

### C) TÜRK PLAKASI (ALPR) — Tespit + OCR

| Dataset/Model | Erişim linki | Boyut | Format | Lisans | Senaryo | Benchmark (kaynaklı) | T4 uygunluk |
|---|---|---|---|---|---|---|---|
| **License Plate Recognition (Roboflow Universe Projects)** | universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e | **Tam 10,125 görsel** (en güncel sürüm v11, 2025-04-02) | YOLO (plaka bbox) | **CC BY 4.0** | Roadside/araç, çeşitli açı | Ön-eğitimli model **mAP@50 %97.8, Precision %98.6, Recall %94.7** | Çok uygun — tespit için ideal |
| **Turkish License Plate Dataset (Kaggle, smaildurcan)** | kaggle.com/datasets/smaildurcan/turkish-license-plate-dataset | Orta | YOLO (YOLOv5 formatı) | Kaggle public | Türkiye, roadside | — | Çok uygun — TR'ye özel |
| **Turkish Number Plates (Roboflow plakatanima)** | universe.roboflow.com/plakatanima-vnt3k/turkish-number-plates | 2,246 görsel + hazır model | YOLO (license_plate) | CC BY 4.0 | Türkiye | Hazır model var | Çok uygun |
| **License Plates of Vehicles in Turkey (Roboflow kemalkilicaslan)** | universe.roboflow.com/kemalkilicaslan-gzpvq/license-plates-of-vehicles-in-turkey-s3tbj | 3,501 görsel | YOLO (YOLOv11n) | CC BY 4.0 | Türkiye | — | Çok uygun |
| **Sentetik TR plaka üreteci** | github.com/adiladiloglu/ML.Synthetic.LicensePlate.Generator (TR config dizini) | Sınırsız üretim; ayrıca literatürde Üstünkök/Atılım Üniv. 100,000 sentetik TR plaka (1025×218 px, 33 karakter: 10 rakam + 23 harf, Q/W/X yok) | Görsel + etiket (bbox/köşe) | Repo lisansını kontrol edin | Sentetik | — | OCR karakter çeşitliliği için faydalı |
| **EasyOCR (recognition)** | github.com/JaidedAI/EasyOCR | Hazır model (Türkçe `tr` desteği) | CTC tabanlı (ResNet+LSTM+CTC) | Apache 2.0 | Plaka kırpımı | utkuatasoy hibrit (52,201 görsel; 6,784'ü TR): YOLOv11x mAP@50 0.98466, YOLOv11m mAP@50-95 0.71743 — bunlar tespit metriği, OCR için ayrı CER yayınlanmamış | **En uygun** — `allowlist` ile sadece TR plaka karakterleri + regex |
| **PaddleOCR (recognition)** | github.com/PaddlePaddle/PaddleOCR | Hazır + fine-tune | CRNN | Apache 2.0 | Plaka | ADA447 projesi TR karakter listesi (rakam + büyük TR harf) ile fine-tune | Uygun |

**Pipeline:** YOLOv8 ile plaka bbox → kırp → perspektif düzelt → `easyocr.Reader(['tr'])` (`allowlist` = `0123456789` + TR plaka harfleri, Q/W/X yok) → çıktıyı `^\d{2}[A-Z]{1,4}\d{1,4}$` regex ile doğrula → birleşik metin "34ABC123". Çıktıyı ASCII-güvenli büyük harfe normalize edin.

### D) ARAÇ TİPİ + RENK

| Dataset | Erişim linki | Boyut | Format | Lisans | Senaryo | Benchmark | T4 uygunluk |
|---|---|---|---|---|---|---|---|
| **VCoR (Vehicle Color Recognition)** | kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset | **≈10,500 görsel, 15 renk sınıfı (train 7.5k / val 1.5k / test 1.5k)** (Kezebou et al., MDPI AI 2021, doi:10.3390/ai2040041); sınıflar: white, black, grey, silver, red, blue, brown, green, beige, orange, gold, yellow, purple, pink, tan | Sınıflandırma (klasör bazlı) | Kaynak makale CC BY 4.0; **Kaggle data card lisans dizesi indirmeden önce manuel doğrulanmalı** | Frontal/roadside | ViT-B/16 ile yüksek acc (Veri-Car, arXiv:2411.06864) | Çok uygun — renk başlığı fine-tune'u hızlı |
| **Vehicle Dataset for YOLO** | datasetninja.com/vehicle-dataset-for-yolo | 3,000 görsel (2,100 train / 900 val) | YOLO (6 sınıf: car, bus, truck, motorbike, van, threewheel) | Açık (Kaggle/Stanford derlemesi) | Roadside | YOLOv5 | Uygun — tip tespiti |
| **Vehicle Classification V2 (Roboflow)** | universe.roboflow.com/vehicle-classfication/vehicle-classification-v2 | Orta | Instance seg/YOLO (21 sınıf: SUV, Pickup, Sedan, Hatchback, Van, Minibus: Small, vb.) | Roboflow | Roadside/overhead | — | Çok uygun — ince taneli TR tiplerine en yakın |
| **Stanford Cars** | morrisfl/stanford_cars_refined (renk türevi) | 16,185 görsel, 196 sınıf | Sınıflandırma | Akademik (erişim son yıllarda değişti — kontrol) | Çeşitli | Fine-grained SOTA | Marka/model türetme için |
| **BIT-Vehicle** | Akademik | 6 tip (Bus, Microbus, Minivan, Sedan, SUV, Truck) | Tespit/sınıflandırma | Akademik | Overhead traffic | — | Uygun |

**TR araç tipleri** (sedan/suv/hatchback/pickup/minibus/panelvan/truck): Roboflow Vehicle Classification V2 en yakın sınıf setini sunar; panelvan/minibüs için ek Roboflow araması veya YOLO-World prompt önerilir. Renk etiketlerini ASCII-küçük harf yapın (ör. `beyaz`, `gri`, `siyah`, `gumus`).

### E) ŞERİT / SLALOM (lane departure, weaving)

| Dataset | Erişim linki | Boyut | Format | Lisans | Senaryo | Benchmark | T4 uygunluk |
|---|---|---|---|---|---|---|---|
| **TuSimple** | github.com/TuSimple/tusimple-benchmark | **6,408 görsel toplam: 3,626 train + 358 val + 2,782 test, 1280×720; 1 sn'lik 20-kare klipler** (resmi readme); yalnız her klibin son karesi şerit polyline ile etiketli | Şerit nokta anotasyonu (JSON polyline) | Araştırma (bazı mirror'larda "No License" — atıf verin) | Roadside/highway, gündüz | ERFNet, LaneNet, UFLD | Uygun — şerit segmentasyonu |
| **CULane** | Akademik | 100k+ frame | Şerit (polyline) | Akademik | Şehir/highway, çeşitli koşul | SCNN, UFLD | Ağır — T4'te tam eğitim zor |
| **BDD100K** | Akademik | 100k video | Şerit + sürülebilir alan | BDD lisansı (akademik) | Çeşitli, gündüz/gece/yağmur | Çeşitli | Ağır |

**Slalom/düzensiz sürüş:** Doğrudan açık "weaving" seti nadirdir; TuSimple ile şerit tespit edip aracın şerit merkezine göre yanal sapma/salınım (oscillation) heuristic'i ile slalom çıkarımı yapın. Hazır UFLD (Ultra-Fast-Lane-Detection) ağırlıkları offline gömme için uygundur.

## SOTA Yaklaşımları (kaynaklı)

- **DMS:** Açık-kaynak SOTA, görsel ipuçlarını (yüz/el/gövde) birleştiren CNN/ViT tabanlı yaklaşımlardır; çok-görevli ve temporal modeller (Drive&Act; TML %96.3 AUC). Pratik gerçek-zaman için YOLOv8 + hafif sınıflandırma başlığı veya ME-YOLOv8 (MHSA+ECA dikkat modülleri, IET ITS 2024). Sizin modüler yaklaşımınız (COCO YOLOv8m + başlıklar + MediaPipe Face Mesh ile PERCLOS/esneme) SOTA-uyumludur ve T4 kısıtına en uygun olandır.
- **Plaka:** YOLO (v8/v11) tespit + CTC tabanlı OCR (EasyOCR/PaddleOCR) iki-aşamalı pipeline; utkuatasoy hibrit sistemi YOLOv11x mAP@50 0.98466 bildirir (tespit).
- **Araç tip/renk:** YOLO tespit + ResNet/ViT transfer-öğrenme sınıflandırma başlığı (MDPI Appl. Sci. 2024; ViT-B/16 ile VCoR, Veri-Car).
- **In-cabin nesne / açık-kelime:** YOLO-World (CVPR 2024) "prompt-then-detect" offline vocabulary; YOLOE görsel-prompt few-shot (Ultralytics).

## Recommendations

**En yüksek değer/efor — yalnızca 2-3 set indirilecekse:**

| Öncelik | Set | Neden | Efor |
|---|---|---|---|
| **1** | **Bir Roboflow DMS YOLO seti** (ör. "DRIVER MENTORING" 8,047 görsel: drinking/drowsy/seatbelt/smoking/phone) | Tek sette telefon+sigara+içme+kemer+uyku bbox; YOLOv8m'e doğrudan fine-tune; CC BY 4.0 | Düşük — birkaç saat |
| **2** | **License Plate Recognition (Roboflow, 10,125 görsel, CC BY 4.0)** + **EasyOCR `tr`** | Plaka tespit (mAP@50 %97.8 hazır model) + hazır OCR; sıfır OCR eğitimi; regex doğrulama | Düşük |
| **3** | **State Farm Distracted Driver** + **VCoR** | DMS sınıflandırma omurgası (22,424 görsel RGB) ve araç rengi (15 sınıf); ikisi de hızlı sınıflandırma başlığı | Düşük-orta |
| Opsiyonel | **YawDD/yawn-eye + NTHU-DDD** | Esneme/PERCLOS ve gece/IR varyasyonu | Orta |

**Aşamalı plan:**
1. **Gün 1 sabah:** YOLOv8m COCO ağırlığını gömün (telefon/şişe/bardak/laptop hazır). Roboflow DMS setini indirip YOLOv8m'i in-cabin sınıflar için fine-tune edin.
2. **Gün 1 öğleden sonra:** Plaka YOLO setini (10,125 görsel) indirip plaka-tespit başlığını eğitin; EasyOCR `tr` + `allowlist` + regex pipeline'ını kurun.
3. **Gün 2 sabah:** VCoR ile renk, Vehicle Classification V2 ile tip başlığı; YOLO-World prompt embedding'lerini markalı kutu için önceden hesaplayıp gömün. MediaPipe Face Mesh ile EAR/MAR tabanlı PERCLOS/esneme heuristic'i ekleyin.
4. **Gün 2 öğleden sonra:** TuSimple/UFLD ile şerit; tüm ağırlıkları <8 GB Docker imajına bakın, offline test, 10-dk limiti için ONNX/TensorRT optimizasyonu.

**Eşik/karar kriterleri:**
- Bir Roboflow DMS setinin mAP'i komite örnek videosunda **<0.5** ise → State Farm + ek Roboflow seti ile sınıf dengesini artırın.
- OCR karakter doğruluğu **<%90** ise → PaddleOCR'a geçin veya sentetik TR plaka üreteci ile EasyOCR recognition başlığını fine-tune edin.
- T4'te **10-dk limiti aşılırsa** → YOLOv8m yerine YOLOv8s + INT8 quantization; frame atlama (her N karede bir DMS).
- Docker imajı **>8 GB** ise → YOLO-World yerine yalnız gerekli prompt embedding'lerini gömün, gereksiz model varyantlarını çıkarın.

## Caveats
- **DMD ve AUC lisansları ticari değil / onay gerektirir.** Yarışma çıktısı ticari sayılırsa bunları kullanmayın. DMD verisi CC BY-NC-ND 4.0 (yalnız akademik), 2026'da IR/Depth/simülatör akışları kaldırılmış, sadece RGB gerçek-araç kalmıştır. State Farm yarışma verisidir; **Roboflow CC BY 4.0 setleri en güvenli seçenektir.**
- **VCoR Kaggle lisans dizesi ve tam dosya boyutu (GB) doğrulanamadı** (Kaggle JS render engeli); indirmeden önce data card'daki "License" alanını manuel kontrol edin. Görsel sayısı (≈10,500) ve sınıf sayısı (15) çok sayıda akademik kaynakla (MDPI AI 2021, Veri-Car) teyitlidir.
- **TuSimple ve bazı drowsiness Kaggle setlerinde açık lisans belirtilmemiştir;** atıf verin ve mümkünse yedek (CC BY) set bulundurun.
- **Benchmark sayıları (mAP/accuracy) farklı split ve koşullarda elde edilmiştir;** komite örnek videosunda doğrulamadan SOTA performans varsaymayın. utkuatasoy'un yüksek skorları plaka **tespiti** (mAP) metriğidir, OCR/karakter doğruluğu değildir.
- **Roboflow proje lisansları proje bazlıdır:** Her seti indirmeden önce o projenin "License" alanını (çoğu CC BY 4.0) tek tek doğrulayın; bazıları "snap" (Roboflow eğitimli) olup yeniden dağıtım kısıtı olabilir.
- Türkçe etiketleri ASCII-güvenli küçük harfe normalize edin (ör. `ı`→`i`, `ş`→`s`, `ö`→`o`, `ç`→`c`, `ü`→`u`, `ğ`→`g`); UTF-8/encoding sorunlarından kaçının.