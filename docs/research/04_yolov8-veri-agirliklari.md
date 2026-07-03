# TEKNOFEST 2026 — Sürücü Davranış Tespiti için Açık Kaynak YOLOv8 Veri Setleri ve Hazır Ağırlıklar

## TL;DR
- For **smoking (sigara_icme)**, do NOT reuse a positives-only set; build a balanced dataset around the Mendeley/Kaggle "Smoker Detection" set (1,120 images, exactly **560 Smoking / 560 NotSmoking**, CC BY 4.0 — Group A) and add explicit negative (non-smoking driver) frames from your own clean videos.
- For **seatbelt (emniyet_kemeri)**, the cleanest fully-permissive two-class build is dms "seatbelt" (1,510, CC BY 4.0) **+** ineuron "no seatbelt" (1,180, MIT); the safest single unified option is Roboflow "Abnormal Driver Behaviour" (CC BY 4.0, with Cigarette + Seatbelt classes).
- For **ready weights**, use HuggingFace `Enos-123/smoking-detection` (YOLOv11m, MIT — Group A) as a baseline; **avoid DMD and AUC Distracted Driver** (Group C: CC BY-NC-ND / sign-and-request EULA — competition risk, and neither even contains both smoking+seatbelt labels).

## Key Findings
- The class-imbalance failure you hit is a textbook "background collapse": with 312 all-positive images and zero negatives, the model learns P(smoking | driver-in-car) ≈ 1. The fix is **data-level**: roughly 30–50% of training frames must be confirmed non-smoking drivers (and ideally hard negatives — hand near mouth, drinking, phone). YOLOv8 natively supports background/negative images (images with no corresponding label file).
- High-quality **balanced** smoking data exists and is openly licensed. The Mendeley "Smoker Detection Dataset" is exactly balanced and, importantly, its NotSmoking class was deliberately built with confusable gestures: per the Mendeley page, "the NotSmoking class consists of images of non-smokers with slightly similar gestures as that of smoking images such as people drinking water, using inhaler, holding the mobile phone, coughing etc." (all images resized to 250×250). These are precisely the hard negatives you need.
- Most balanced smoking sets are **classification** sets, so to use them in YOLOv8 detection you either (a) draw cigarette/driver-region boxes, or (b) use them in a two-stage classifier, or (c) use them as a negative/positive balance supplement to a detection set.
- Seatbelt data is abundant but mostly **single-class** ("seatbelt" worn only). True two-class (worn vs not-worn) detection sets exist (ROTRATECTION, KARAN PANJA, dms 3-class) but published balance figures are rare — you must verify per project.
- **Unified DMS datasets** covering smoking AND seatbelt in one image exist on Roboflow (University "Abnormal Driver Behaviour", Jui "Driver behaviors"), both CC BY 4.0.
- Several **restricted academic datasets (DMD, AUC Distracted Driver)** are a legal hazard for a competition because fine-tuning creates a derivative; DMD is CC BY-NC-ND (NoDerivatives), AUC requires a signed non-commercial EULA. Neither even contains both smoking+seatbelt labels.

## Details

### License grouping (as requested)
- **GROUP A (safe, usable):** CC BY 4.0 / MIT / Apache 2.0 — "safe, usable."
- **GROUP B (usable, with warning):** "Roboflow Public" / CC BY-SA / unclear-but-open — "license must be manually verified." Most Roboflow Public projects are actually CC BY 4.0; checking the License field on the project page before download is sufficient.
- **GROUP C (last resort, red flag):** CC-NC, CC BY-ND, or approval-required academic datasets (DMD, AUC). "COMMERCIAL/DERIVATIVE RESTRICTION — risk of using in competition." Fine-tuning = derivative; ND prohibits derivatives.

---

### 1) SMOKING DETECTION (sigara_icme) — MUST include NEGATIVES

**GROUP A — safe**

| Dataset | Link | Images | Pos/Neg balance | Classes | License | Format |
|---|---|---|---|---|---|---|
| Mendeley "Smoker Detection Dataset" (Khan, 2020) | https://data.mendeley.com/datasets/j45dj8bgfc/1 | 1,120 | **560 Smoking / 560 NotSmoking** (perfect) | 2 (classification) | CC BY 4.0 | raw images (250×250) → convert to YOLO |
| Mendeley "smoking vs non-smoking" (larger) | https://data.mendeley.com/datasets/7b52hhzs3r/1 | 2,400 | **1,200 smoking / 1,200 not-smoking** | 2 (classification) | CC BY 4.0 | raw images → convert to YOLO |
| Kaggle mirror (sujaykapadnis) | https://www.kaggle.com/datasets/sujaykapadnis/smoking | 1,120 | 560/560 (mirror of Mendeley) | 2 | CC BY 4.0 (verify on page) | one-click Kaggle API in Colab |
| HF `keremberke/smoke-object-detection` | https://huggingface.co/datasets/keremberke/smoke-object-detection | **21,578** | n/a (single class) | 1 ("smoke") | CC BY 4.0 | COCO (640×640) → convert to YOLO |

Notes:
- The **Mendeley 1,120-image set is your single best antidote** to the false-positive problem because the negatives are *confusable* (drinking/inhaler/phone/coughing), not just empty backgrounds.
- `keremberke/smoke-object-detection` is the Roboflow "Smoke100" set — per the HF card, "It includes 21578 images. Smoke are annotated in COCO format." **Caution:** this targets visual/ambient smoke, not a cigarette in a driver's mouth — use only as a supplement and validate semantics.

**GROUP B — usable, verify license field**

- **Kaggle "Smoking and Drinking Dataset for YOLO" (prajjwalkumarpanzade)** — https://www.kaggle.com/datasets/prajjwalkumarpanzade/smoking-and-drinking-dataset-for-yolo — already YOLO-format detection (smoking + drinking); balance not published — verify.
- **Roboflow "Smoking Person Detection" (SPD)** — https://universe.roboflow.com/spd/smoking-person-detection-h0a2x — 2,789 images. Verify license + whether negatives included.
- **Roboflow "Smoking detection" (detection-ys9qa)** — https://universe.roboflow.com/detection-ys9qa/smoking-detection-ersye — 5,727 images, pretrained model available. Verify license.

**GROUP C / cautionary**
- Avoid positives-only Roboflow smoking sets (e.g., `cigarette-detection/smoking-detection-ab2uk`, 291 images, single "Merokok" class) — these are exactly the type that caused your collapse. Use only if you add your own negatives.

---

### 2) SEATBELT DETECTION (emniyet_kemeri) — prefer TWO-CLASS

**GROUP A — safe (license confirmed on page)**

| Dataset | Link | Images | Classes | License | Notes |
|---|---|---|---|---|---|
| dms "seatbelt" | https://universe.roboflow.com/dms-vewel/seatbelt-smjqq | 1,510 (1,149 train / 361 val) | "seatbelt" (worn) | **CC BY 4.0** | pair with no-seatbelt set below |
| ineuron "no seatbelt" | https://universe.roboflow.com/ineuron-8bdse/no-seatbelt | 1,180 (943 train / 237 val) | "no seatbelt" | **MIT** | pairs perfectly with dms set → balanced 2-class |
| 2tech "Seat-Belt Detection" | https://universe.roboflow.com/2tech/seat-belt-detection-udcfg | 870 (v5 ≈ 2,087) | "seat_belt" | **CC BY 4.0** | has pretrained yolov8s (mAP@50 78.0%, P 82.3%, R 68.7%) |

The recommended Group-A build: **combine dms "seatbelt" (worn) + ineuron "no seatbelt" → a clean, fully-permissive two-class chest-strap detector (~1.5k worn + 1.2k not-worn).** Both export to YOLOv8.

**GROUP B — usable, verify license field**

- **ROTRATECTION "seatbelt / no-seatbelt"** — https://universe.roboflow.com/rotratection/rotratection-dataset — ~9.94k images (largest two-class option found), classes include seatbelt + no-seatbelt. Balance not published — verify.
- **KARAN PANJA "seat belt detection"** — https://universe.roboflow.com/karan-panja/seat-belt-detection-uhqwa — 660 images; classes person-seatbelt / person-noseatbelt / "Not Clear" (worn vs not-worn vs ambiguous). Verify license.
- **dms 3-class (person-seatbelt / person-noseatbelt / seatbelt)** — https://universe.roboflow.com/dms-vewel — ~1.51k images, two-class chest-strap. Verify license.
- **Ujjawal "Seat Belt 2"** — https://universe.roboflow.com/ujjawal/seat-belt-2-77oye — classes seatbelt / no-seatbelt. Verify license.
- **Computer Vision "Seatbelt detection"** — https://universe.roboflow.com/computer-vision-uiivc/seatbelt-detection-jhdcy — 215 images, classes "no seat-belt" + "windshield", CC BY 4.0. Small.

---

### 3) UNIFIED DMS DATASETS (smoking + seatbelt + phone in one set)

**GROUP A — safe (CC BY 4.0 confirmed)**

- **University "Abnormal Driver Behaviour"** — https://universe.roboflow.com/university-exrks/abnormal-driver-behaviour
  - Images: 2,110 (versions up to ~2.1k). Classes (5): **Phone, Cigarette, Drinking, Seatbelt, Eating**. License: **CC BY 4.0** (confirmed). Pretrained model available. **Best single unified source** — gives both sigara_icme (Cigarette) and emniyet_kemeri (Seatbelt) in real driver-cabin imagery. YOLOv8 export.
- **Jui "Driver behaviors"** — https://universe.roboflow.com/jui/driver-behaviors
  - Images: 9,900. Classes (11): smoke, phone, cell phone, cigarette, smoking, Mobile phone, seatbelt, etc. License: **CC BY 4.0** (confirmed). Pretrained model (mAP 73.8%, P 82.2%, R 69.8%). Large and multi-label; **caveat:** noisy/overlapping taxonomy — remap classes down to your two TEKNOFEST labels before training.

**GROUP B — verify**
- **1-o2o2n "Driver behaviors"** — https://universe.roboflow.com/1-o2o2n/driver-behaviors-fwlbl — 20k images, same 11-class taxonomy, CC BY 4.0 listed. Large; verify.

**GROUP C — AVOID for competition (restricted licenses; also not fit for purpose)**

- **DMD — Driver Monitoring Dataset (Vicomtech)** — https://dmd.vicomtech.org/
  - License: **CC BY-NC-ND 4.0** (NonCommercial + NoDerivatives), academic-use-only, 18+. Verbatim from the site: "You must be 18 or older to download. This dataset can only be used for academic purposes... published under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 License... if you remix, transform, or build upon the material, you may not distribute the modified material." Since fine-tuning produces a derivative, the **ND clause makes this unusable** for a competition deliverable. Also, per the site, a 2026 license revision **reduced** content ("material from specific participants has been removed... we decided to reduce the content"; only RGB remains; Copyright © 2026 Vicomtech). Built from 37 volunteers / ~42 TB raw data (VI-DAS, EC Horizon 2020 Grant 690772). Critically, **DMD's activity labels do NOT include smoking or seatbelt** (they cover phone/texting/drinking/hair-makeup/yawning/hands-on-wheel/gaze). → **COMMERCIAL/DERIVATIVE RESTRICTION — competition risk + not fit for purpose.**
- **AUC Distracted Driver Dataset (American University in Cairo)** — https://abouelnaga.io/projects/auc-distracted-driver-dataset/ (license PDF: https://heshameraqi.github.io/data/auc.distracted.driver.dataset/Distracted_Driver_Dataset_V1_License_Agreement.pdf)
  - Access: custom MI-AUC EULA — **available only after signing the license agreement and emailing the authors.** Verbatim: "Any commercial use of the dataset is strictly prohibited"; redistribution prohibited without authorization; "The dataset shall remain the exclusive property of the MI-AUC." 10 distracted-driving posture classes — **no smoking or seatbelt class.** → **COMMERCIAL/DERIVATIVE RESTRICTION — competition risk + not fit for purpose.**

---

### 4) READY PRETRAINED WEIGHTS (.pt / .onnx) — download in advance (internet OFF at runtime)

**GROUP A — safe**

- **HuggingFace `Enos-123/smoking-detection`** — https://huggingface.co/Enos-123/smoking-detection
  - YOLOv11-Medium, single class "cigarette". License: **MIT**. Metrics (verbatim from the model card): **Precision 85.62% | Recall 76.92% | mAP@0.5 82.90% | mAP@0.5:0.95 44.69%.** Base model Ultralytics/YOLO11; trained on Roboflow "Cigarette Smoke Detection" (`universe.roboflow.com/yolo-pdvpx/cigarette-h2p1m`). `best.pt` downloadable; supports `model.export(format="onnx")`. YOLOv11 (not v8) but runs in the same Ultralytics API and is T4-compatible. **Best ready cigarette detector with a clean license.**

**GROUP B / cautionary (AGPL-3.0 = copyleft)**

- **HuggingFace `kittendev/YOLOv8m-smoke-detection`** — https://huggingface.co/kittendev/YOLOv8m-smoke-detection — YOLOv8m, label "smoke", mAP@0.5 self-reported 0.995. License: **AGPL-3.0**. Trained on `keremberke/smoke-object-detection` (ambient smoke). Validate that it detects a cigarette vs ambient smoke before relying on it.
- **Roboflow "deploy" weights** — the pretrained models attached to "Abnormal Driver Behaviour", "Seat-Belt Detection (2tech)", and "Driver behaviors (Jui)" can be exported via the Roboflow SDK; license follows the dataset (CC BY 4.0). Download in advance and run offline.
- **Base Ultralytics weights** — https://huggingface.co/Ultralytics/YOLOv8 — yolov8n/s/m.pt (COCO, AGPL-3.0) as your fine-tuning starting point; the COCO "cell phone" class is already present for the phone sub-task.

**Seatbelt ready weights:** No clean, single-purpose seatbelt `.pt` with a permissive license was found on HuggingFace. Practical path: fine-tune YOLOv8 on the Group-A seatbelt sets above, or deploy the 2tech yolov8s seat-belt model via Roboflow.

---

## Recommendations

**Stage 1 — Fix the smoking false-positive problem first (highest priority).**
1. Build a balanced detection set: take a cigarette-detection positive source (e.g., "Abnormal Driver Behaviour" → Cigarette, CC BY 4.0) for positives, and inject **negatives** — non-smoking driver frames from your own clean videos plus the NotSmoking class from the Mendeley set (CC BY 4.0). Target **≥30–40% background/negative frames**.
2. Add hard negatives: drivers drinking, hand-to-face, phone-to-mouth — these directly break the "driver = smoking" shortcut.
3. Benchmark: after retraining, measure false-positive rate on a held-out set of clean (non-smoking) driver videos. **Target FPR < 5% at conf 0.5.** If still high, raise the negative ratio toward 50%.

**Stage 2 — Seatbelt two-class detector.**
1. Combine dms "seatbelt" (CC BY 4.0, 1,510) + ineuron "no seatbelt" (MIT, 1,180) → balanced two-class set. Both Group A, both YOLOv8 export.
2. For more volume, add ROTRATECTION (~9.94k, Group B — verify license field first).
3. Benchmark on cabin-angle frames matching your camera; strap detection is viewpoint-sensitive.

**Stage 3 — Unified model option.**
- For a single model covering both labels, start from "Abnormal Driver Behaviour" (CC BY 4.0) and remap its 5 classes to your two (Cigarette → sigara_icme; Seatbelt/no-seatbelt → emniyet_kemeri). Fine-tune on Colab A100, export `.pt` **and** `.onnx`, and pre-download for offline T4 inference.

**Stage 4 — Weights & deployment.**
- Download `Enos-123/smoking-detection` best.pt (MIT) as a baseline/sanity check before your own training finishes. Export to ONNX/TensorRT for the T4. **Pre-cache all weights locally** since inference runs with internet OFF.

**Threshold that changes the plan:** If balanced-data retraining still yields high-confidence false positives, switch from a "cigarette object" detector to a **two-stage** approach (driver crop → smoking/not-smoking classifier using the Mendeley balanced set), which is structurally more robust to background collapse.

## Caveats
- **Balance figures for most Roboflow sets are not published.** Always open the project's "Health Check"/class-balance view and the License field before downloading. Treat any positives-only set as dangerous for your use case.
- **Verify the License field per project at download time** for all Group B items — Roboflow "Public" usually maps to CC BY 4.0 but is not guaranteed.
- **Classification vs detection:** the strongest balanced smoking sets (Mendeley/Kaggle) are classification sets; converting to YOLOv8 detection requires drawing boxes. Budget annotation time, or use them in a two-stage classifier.
- **"Smoke" ≠ "cigarette":** many HF/Roboflow "smoke" models/datasets target ambient/visual smoke or wildfire, not a cigarette in a driver's mouth. Validate class semantics before training.
- **AGPL-3.0 weights** (kittendev, base Ultralytics) carry copyleft obligations; MIT/CC BY options are cleaner for a competition deliverable.
- **DMD and AUC are both unfit and restricted** — they lack smoking+seatbelt labels AND carry NC/ND/EULA restrictions; do not use.
- Roboflow image counts shift between dataset versions; the figures above reflect the versions seen on the project pages and may differ slightly from the latest version. All URLs in this report were retrieved/verified during research; the only links not directly opened (but returned by search) were a few Group-B Roboflow projects — confirm those pages load before relying on them.