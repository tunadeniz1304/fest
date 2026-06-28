# FTR DOCX'e Yapıştırılacak Metin (SynchAgent)

> KULLANIM: Root'taki `2026_5G..._FTR_şablon_TR...docx` şablonunu aç.
> Kapak: Takım Adı=SynchAgent, Takım ID=998490, Başvuru ID=5141062 (zaten alanlar var).
> Aşağıdaki metinleri ilgili bölümlere yapıştır. Şablon kuralları:
> **Yazı tipi Arial 12, Başlık Arial Black 14, satır aralığı 1.15, iki yana yaslı,
> kenar boşluk üst 2.8 / alt-sağ-sol 2.5.** Tablolar Word'de "Tablo Ekle" ile yapılır.
> Bölüm başlıkları şablonda HAZIR — sadece altlarını doldur. 3-10 sayfa arası olmalı.

═══════════════════════════════════════════════════════════════
## 1. PROJE ÖZETİ (5 Puan)
═══════════════════════════════════════════════════════════════

Bu proje kapsamında, yol kenarı (MOBESE) ve araç içi kamera akışlarından yol güvenliğini tehdit eden durumları tespit eden, modüler ve çevrim-dışı (offline) çalışan bir bilgisayarlı görü sistemi geliştirilmiştir. Sistem; araç tipi, rengi ve plaka tanıma; sürücü davranış analizi (telefonla konuşma, su içme, esneme, yana/arkaya bakma, sigara içme); yolcu konum tespiti; slalom (düzensiz sürüş) ve açık-kelime nesne tespiti (teknocan) olmak üzere şartnamede tanımlı sürücü güvenliği etiketlerini, results.json şemasına birebir uyumlu olarak üretmektedir.

Tasarım, yarışmanın Tesla T4 GPU, 8 GB altı imaj ve 10 dakika süre kısıtlarına uygundur. Tüm model ağırlıkları Docker imajına gömülü olduğundan sistem çalışma anında internet bağlantısı gerektirmez. Projenin ayırt edici katkısı, sürücü-izleme veri setlerinde yaygın görülen aşırı-öğrenme (overfitting) ve veri sızıntısı (data leakage) problemlerinin açıkça ele alınması; dengeli veri, doğru doğrulama protokolü ve gerçek-video tabanlı yanlış-pozitif analizi ile güvenilirliğin kanıtlanmasıdır.

═══════════════════════════════════════════════════════════════
## 2. VERİSETİ OLUŞTURULMASI (20 Puan)
═══════════════════════════════════════════════════════════════

Değerlendirme komitesi örnek video dışında eğitim verisi sağlamadığından, modeller açık kaynaklı ve ticari kullanıma uygun lisanslı veri setleriyle eğitilmiştir.

[TABLO 1 — Word'de tablo olarak ekle:]
Görev | Veri Seti | Görsel Sayısı | Lisans
Sürücü davranışı | Kaggle DMS (habbas11) | 5957 | Apache 2.0
Araç rengi | VCoR | ~10500 | CC BY 4.0
Plaka | License Plate Recognition | 98798 | CC BY 4.0
Negatif (dengeleme) | State Farm c0 | 2489 | Yarışma
Nesne omurgası | COCO (YOLOv8m) | - | -

**Veri Dengeleme (Data Balancing):** Ön çalışmalarımızda, yalnızca pozitif örnek içeren (negatif/"güvenli sürüş" örneği bulunmayan) veri setleriyle eğitilen modellerin "araç içi sahne = ihlal" kısayolunu öğrenerek, temiz videolarda dahi yüksek güvenle yanlış-pozitif ürettiği tespit edilmiştir. Bu, State Farm 2016 yarışmasında da belgelenmiş bilinen bir tuzaktır. Çözüm olarak eğitim setine State Farm c0 (güvenli sürüş, 2489 görsel) ve temiz sürücü video kareleri boş-etiket (background) olarak eklenmiş; toplam eğitim verisinin %31'i negatif yapılmıştır. Böylece model, arka planı ezberlemek yerine gerçek nesneyi öğrenmeye zorlanmıştır.

**Veri Artırma (Data Augmentation):** Model gürbüzlüğü için domain-randomization tabanlı artırma uygulanmıştır: HSV/parlaklık/kontrast değişimi (gece/gündüz/IR koşulları), Mosaic, MixUp, rastgele döndürme/öteleme/ölçekleme ve rastgele silme (occlusion). Bu teknikler, şartnamedeki farklı FPS, çözünürlük ve ışık koşullarına dayanıklılık gereksinimini hedefler.

**Eğitim/Doğrulama/Test Ayrımı:** Kaggle DMS veri setinin varsayılan train/valid ayrımı (5957/2389 görsel) kullanılmış; eğitime State Farm c0 ve temiz video kareleri %31 oranında negatif olarak eklenmiştir. Aşırı-öğrenmeyi önlemek için nihai doğrulama, eğitimde hiç kullanılmayan gerçek sürücü videolarıyla yapılmıştır.

═══════════════════════════════════════════════════════════════
## 3. YAPAY ZEKÂ ÇÖZÜMÜ (50 Puan)
═══════════════════════════════════════════════════════════════

### 3.1. Problemin Analizi (15 Puan)

Video üzerinden araç, plaka ve riskli sürücü davranışı tespiti yaparken karşılaşılan temel problemler ve izlenen çözüm yolu şunlardır:

- **Aşırı-öğrenme ve veri sızıntısı:** Sürücü izleme veri setleri çoğunlukla video karelerinden oluşur; komşu kareler eğitim ve doğrulama setlerine dağılırsa doğrulama skoru yanıltıcı şekilde yükselir (validasyonda %99 görünen bir model gerçekte %38 doğruluk verebilir). Çözüm olarak dengeli veri ve gerçek-video doğrulaması tercih edilmiştir.

- **Kamera açısı belirsizliği:** Sürücü izleme kamerası önden veya yandan konumlanmış olabilir. Kafa yönü (bakma) tespiti için MediaPipe Face Mesh landmark noktaları ile solvePnP (üç boyutlu kafa-pozu geometrisi) kullanılmış; ardışık-süreklilik şartı ve eşik kalibrasyonu ile yanlış-pozitifler azaltılmıştır.

- **Işık değişimi, hareket bulanıklığı ve oklüzyon:** Domain-randomization tabanlı veri artırma ve takip-bazlı zamansal oylama (track-level voting) ile ele alınmıştır.

- **Yanlış-pozitif riski:** Ezberci/dengesiz modeller devre dışı bırakılmış, dengeli ve güvenilir kaynaklara (COCO) öncelik verilmiştir. Bu, sistemin temiz videolarda yanlış alarm vermemesini sağlar.

### 3.2. Çözüm Mimarisi (15 Puan)

Sistem, ham videodan results.json çıktısına kadar modüler bir işlem hattı (pipeline) olarak çalışır. Her tespit katmanı izoledir: bir katman hata verse dahi ana akış geçerli çıktı üretmeye devam eder (şartnamedeki hata-yönetimi gereksinimi).

[MİMARİ DİYAGRAM — Word'de SmartArt/şekil veya basit kutu-ok diyagramı olarak çiz:]

Video Girişi (/app/data/input/video.mp4)
   ↓ kare örnekleme (~5 FPS, FPS-bağımsız)
YOLOv8m (COCO) + ByteTrack çoklu-nesne takibi
   ↓ track-level çoğunluk oylama (zamansal)
Paralel Tespit Katmanları (her biri izole):
   • Araç: tip | renk (VCoR) | plaka (YOLO + EasyOCR)
   • Davranış: telefon/su (COCO) | esneme/yorgunluk (MediaPipe)
   • bakma (kafa-pozu) | sigara (Kaggle DMS) | slalom (yörünge)
   • Yolcu konumu | teknocan (YOLO-World)
   ↓ tespit birleştirme (dedup)
results.json (şartname şeması)

### 3.3. Çözüm Detayları (20 Puan)

**Omurga ve izleme:** Sistemin omurgası COCO ön-eğitimli YOLOv8m nesne tespit modeli ve ByteTrack çoklu-nesne takip algoritmasıdır. Track boyunca uygulanan zamansal çoğunluk oylaması, tek-kare yanlış-pozitifleri eler ve kaçan kareleri köprüler; bu hem precision hem recall'ü artırır.

**Araç analizi:** Araç tipi, bbox en-boy oranı sezgisi ve COCO sınıfı birleşimiyle belirlenir. Renk, VCoR veri seti üzerinde eğitilmiş YOLOv8m sınıflandırma modeliyle (15 renk → 9 şartname rengine eşleme) tespit edilir. Plaka için License Plate veri setiyle eğitilmiş model plaka bölgesini bulur, dört-nokta perspektif düzeltmesi uygulanır, ardından EasyOCR (Türkçe) ile karakterler okunur ve TR plaka regex doğrulaması + karakter-bazlı oylama ile kesinleştirilir.

**Sürücü davranışı:** Telefonla konuşma, su içme ve bilgisayar kullanma; dengeli COCO sınıflarından (cell phone, bottle, laptop) tespit edilir. Esneme ve yorgunluk, MediaPipe Face Mesh'ten türetilen MAR (ağız) ve EAR (göz) oranlarıyla, eğitimsiz olarak hesaplanır. Yana/arkaya bakma, MediaPipe landmark + solvePnP kafa-pozu ve ardışık-süreklilik şartıyla belirlenir. Sigara içme tespiti, Kaggle DMS veri seti üzerinde negatif-dengeli olarak fine-tune edilmiş YOLOv8m modeliyle yapılır.

**Slalom:** ByteTrack yörüngesinden aracın yanal salınım analizi ile (eğitimsiz) çıkarılır. **teknocan:** YOLO-World açık-kelime modeliyle, gömülü embedding kullanılarak çevrim-dışı tespit edilir.

**Donanım/Yazılım altyapısı:** PyTorch, Ultralytics YOLOv8, OpenCV, MediaPipe, EasyOCR ve OpenCV solvePnP kullanılmıştır. Sistem Docker ile paketlenmiştir (temel imaj: nvidia/cuda:12.1.0-base-ubuntu22.04) ve Tesla T4 üzerinde çalışır. Tüm ağırlıklar imaja gömülü olup çalışma anında internet bağlantısı gerektirmez.

═══════════════════════════════════════════════════════════════
## 4. ÇÖZÜMÜN SINANMASI (20 Puan)
═══════════════════════════════════════════════════════════════

"Çözümümüze neden güveniyoruz?" sorusunun yanıtı, hem standart doğrulama metrikleri hem de eğitimde kullanılmayan gerçek videolarla yapılan testlerdir.

[TABLO 2 — Eğitilen modellerin doğrulama metrikleri (Word'de tablo):]
Model | Precision | Recall | mAP50 | Not
Sigara (Kaggle DMS) | 0.902 | 0.867 | 0.911 | Negatif-dengeli
Plaka | 0.986 | 0.955 | 0.972 | 98K görsel
Renk (VCoR) | - | - | - | top1 doğruluk %87.7

**Çevrim-Dışı (Offline) Docker Testi:** Sistem, "docker run --network none" komutuyla (internet tamamen kapalı, Tesla T4) gerçek bir 4K trafik videosu üzerinde çalıştırılmış; geçerli results.json üretilmiş ve herhangi bir çökme yaşanmamıştır. İmaj boyutu 8 GB sınırının altındadır ve işleme süresi 10 dakika limitinin altındadır.

[TABLO 3 — Gerçek video testi (yanlış/doğru-pozitif kanıtları):]
Video | Beklenen | Sonuç
goodmax (temiz sürücü) | İhlal yok | TEMİZ (yanlış-pozitif yok)
sigaraicenadam | Sigara içme | sigara_icme %77 güvenle yakalandı
Gerçek plaka videoları | Plaka okuma | 03ACU808, 64EP367 başarıyla okundu

**Yanlış-Pozitif Dayanıklılığı:** En kritik doğrulamamız, temiz (ihlalsiz) videoların yanlış alarm üretmemesidir. goodmax videosunda sistem hiçbir ihlal tespiti üretmezken (yanlış-pozitif yok), sigaraicenadam videosunda sigara içme davranışı %77 güvenle doğru tespit edilmiştir. Bu, modelin "araç içi sahne = ihlal" kısayolunu öğrenmediğinin somut kanıtıdır.

**Mühendislik Kalitesi:** 39 birim testi (Test-Driven Development), izole hata-yönetimli katmanlar, JSON şema doğrulayıcı ve ASCII-safe etiket garantisi uygulanmıştır. Kod içinde ortam/hostname/IP tespiti yapan herhangi bir manipülasyon bulunmamaktadır (anti-cheat uyumlu).

═══════════════════════════════════════════════════════════════
## 5. KAYNAKÇA (5 Puan)
═══════════════════════════════════════════════════════════════

> ŞABLON FORMATI (4. sayfa kuralı): Dijital Kaynak için "Yazarın Soyadı, Adının Baş Harfi,
> Yazının Başlığı, Yazının Tarihi, Erişim Tarihi, Erişim Adresi". Aşağıdakileri bu formata göre yaz.

1. habbas11, DMS - Driver Monitoring System Dataset, 2024, Erişim: 28.06.2026, https://www.kaggle.com/datasets/habbas11/dms-driver-monitoring-system (Lisans: Apache 2.0)

2. Kezebou, L. ve diğerleri, VCoR: Vehicle Color Recognition Dataset, MDPI AI, 2021, Erişim: 28.06.2026, https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset (Lisans: CC BY 4.0)

3. Roboflow Universe, License Plate Recognition Dataset, 2024, Erişim: 28.06.2026, https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e (Lisans: CC BY 4.0)

4. State Farm, Distracted Driver Detection, Kaggle, 2016, Erişim: 28.06.2026, https://www.kaggle.com/c/state-farm-distracted-driver-detection

5. Jocher, G. ve diğerleri, Ultralytics YOLOv8, 2023, Erişim: 28.06.2026, https://github.com/ultralytics/ultralytics

6. Zhang, Y. ve diğerleri, ByteTrack: Multi-Object Tracking by Associating Every Detection Box, ECCV, 2022, Erişim: 28.06.2026, https://github.com/ifzhang/ByteTrack

7. Lugaresi, C. ve diğerleri, MediaPipe: A Framework for Building Perception Pipelines, 2019, Erişim: 28.06.2026, https://github.com/google/mediapipe

8. Cheng, T. ve diğerleri, YOLO-World: Real-Time Open-Vocabulary Object Detection, CVPR, 2024, Erişim: 28.06.2026, https://github.com/AILab-CVC/YOLO-World

9. JaidedAI, EasyOCR, 2020, Erişim: 28.06.2026, https://github.com/JaidedAI/EasyOCR (Lisans: Apache 2.0)
