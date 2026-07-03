# Leakage-Free Driver Distraction Detection: Datasets, ROI Training, Augmentation, and Ready Weights for YOLOv8

## TL;DR
- Your 99.5% mAP was an artifact of **frame-level leakage** (adjacent video frames split across train/val) plus **background/appearance memorization** — both documented in the State Farm competition. The fix is a **clip-disjoint, subject-independent split** plus **ROI cropping** (head + right-hand region), exactly the recipe the 1st-place solution (Kaggle user jacobkie) used. Re-evaluate honestly first; published evidence shows honest subject-disjoint accuracy on this kind of data can fall to ~38% with the same model that scores 99% under a leaky split.
- For real-world generalization, **100-Driver** (470,208 RGB+NIR images, 100 drivers, 5 vehicles, 4 cameras, day+night, cross-vehicle/cross-view/cross-modality protocols) and the **NHTSA AI City Challenge / SynDD** naturalistic data are the strongest; **DMD** is the best *real-driving-format* academic set but is **non-commercial (CC BY-NC-ND 4.0)** and has **no smoking class**. AUC and Drive&Act are also non-commercial and lab/side-view biased. **No major public dataset includes all four of your target behaviors (phone, drinking, reaching back, smoking) — smoking is the gap.**
- For a deployable offline-T4 pipeline: build a **two-stage ROI pipeline** (MediaPipe/YOLO person+hand+face detector → crop → YOLOv8 classifier), train on **A100 with subject-disjoint splits + aggressive domain-randomization augmentation**, and seed from existing Roboflow YOLOv8 driver weights. Note that **Ultralytics YOLOv8 itself is AGPL-3.0**, a real commercial constraint independent of the dataset licenses.

## Key Findings

### Dataset comparison (real-world generalization)
| Dataset | Subjects | Modalities | Viewpoint | Smoking? | Subject/clip-disjoint split? | License | Commercial use? |
|---|---|---|---|---|---|---|---|
| **DMD (Vicomtech)** | 37 | RGB, IR, Depth (being reduced to RGB-only in 2025/26) | Face, body (left side), hands — 3 cams | **No** | **Yes** — 24:7:6 driver split (dBehaviourMD) | CC BY-NC-ND 4.0 (tools MIT) | **No** |
| **AUC Distracted Driver** | 31 (V2) / 44 (V1) | RGB | Side view | **No** | Not official; community subject splits exist | MI-AUC custom | **No (strictly prohibited)** |
| **SynDD1 / SynDD2 (NHTSA / AI City)** | 15–99 | IR video, 3 synced cams | Dashboard, rearview, right window | **No** | **Yes** — A1/A2/B driver-disjoint | US Government work / public | **Likely yes (gov work)** |
| **Drive&Act** | 15 | RGB, NIR, Depth, 3D skeleton | 6 views | **No** | **Yes** — 3 splits, no driver overlap | "Research only" (Fraunhofer IOSB) | **No** |
| **100-Driver** | 100 | RGB + NIR | 4 cams (3 front, 1 side), 5 vehicles, day+night | (not confirmed for smoking) | **Yes** — cross-vehicle/view/modality | License form (academic) | **No** |

**Verdict on generalization.** 100-Driver is purpose-built to *measure and improve* cross-domain robustness — per Wang et al., IEEE TITS 2023 (100-driver.github.io), it has "more than 470K images taken by 4 cameras observing 100 drivers over 79 hours from 5 vehicles," with explicit cross-vehicle, cross-view, and cross-modality settings. The danger of leaky training is quantified in the same data family: on 100-Driver cross-view, ResNet50 accuracy collapses from 50.1% (camera D1→D2) to **4.1%** (D4→D2), per Duan et al., "Score Softmax Classifier" (arXiv:2310.05202, Table IV). DMD is the most "real driving" of the academic sets (recorded in a real moving car, not a simulator), which is why it transfers better than Drive&Act (stationary lab simulator with documented "relaxed" behavior and no real illumination/weather). For *legally clean commercial* training data, the **NHTSA AI City / SynDD** material is the safest because it is a US-government-produced work.

### The State Farm 1st-place ("done right") recipe
The winner (Kaggle user **jacobkie**; writeup mirrored at github.com/bdutta19/kaggle_statefarm and competition discussion 22906) used three anti-overfitting techniques:
1. **ROI cropping** — a modified VGG16 ("VGG16_3") was trained on two regions of interest (the head area and the radio / lower-right-hand area) *together with* the original image, forcing the network to look at behavior-relevant regions rather than memorize the cabin background.
2. **KNN averaging** — using the VGG16 `pool5` feature space, each image's 10 nearest neighbors (which reconstruct the underlying video sequence) were weight-averaged. jacobkie's 10-NN weighted average "improves single model score by 0.10~0.12"; the related CS229 Stanford report (cs229.stanford.edu) independently notes KNN "can improve the LB score (log loss) by 0.03 ~ 0.1."
3. **Segment averaging** — test frames are grouped by `pool5` features and consistent, confident groups are renormalized to share predictions.

The root-cause lesson (Felix Yu, flyyufelix.github.io): the images are video frames from **fewer than 100 drivers**, so adjacent frames are near-duplicates and a random split leaks. The corrective is a **person-ID / clip-based split**.

### ROI / two-stage pipelines (open source)
- **MediaPipe (Apache-2.0)** — Holistic/Pose/Hands use a detector→tracker that produces ROI crops for the face and each hand (BlazePose derives "three regions of interest (ROI) crops for each hand (2×) and the face"). Ideal, offline-capable first stage to localize head + hands.
- **shivsondhi/distracted-driver-detection** — explicitly "generate new data... by cropping the hands and faces of the drivers" and trains a three-arm ensemble (full image, hands, faces).
- **saicharan/Distracted-Driver-MultiAction-Classification** + the Medium/TDS writeup (Pachigolla et al.) — explicitly splits by person ID instead of random 80/20; the honest result is stark: "We achieved a loss of 1.76 and an accuracy of 38.5%" — i.e., the leaky split was hiding a model that is barely better than chance on unseen drivers.
- **e-candeloro/Driver-State-Detection (MIT)** — real-time MediaPipe-based driver attention monitor (head pose, gaze, EAR/PERCLOS).

### Augmentation for domain generalization
- **Domain randomization** — texture, exposure, brightness/contrast, Hide&Seek occlusion, and Gaussian noise, combined via **RandAugment**, produce the largest cross-domain gains in the DG literature (the in-orbit pose study, arXiv:2406.11743, reports RandAugment combining augmentations cut error by ~89–93%, illustrating the principle).
- **Lighting/color (HSV) jitter** simulates day/night/IR shifts; color-space adjustment is the augmentation most directly tied to illumination generalization (arXiv:2404.07514).
- **Mosaic + MixUp** (YOLOv8 defaults) plus rotation/flip/brightness were used in published YOLOv8 driver-distraction work to "enhance model generalization" (IEEE 11005355).
- **PQ-DAF** (Liu et al., arXiv:2508.10397v1, 14 Aug 2025) — pose-driven conditional-diffusion augmentation with vision-language quality filtering; in 10-shot training it "attains a Top-1 accuracy of 54.00%, representing an improvement of 17.33 percentage points over the best baseline model, ResNet50 (36.67%)" on State Farm, rising to 88.00% at 30-shot.
- **Score-Softmax** (github.com/congduan-HNU/SSoftmax) reduces background overfitting and raised cross-dataset accuracy by 21.34% / 11.89% / 18.77% on SFDDD / AUCDD / 100-Driver respectively.

### Ready pretrained weights (Roboflow Universe / HuggingFace)
- **"DRIVER MENTORING" Roboflow project (by gonzalo)** — 8,047 images; classes include **drinking, drowsy, seatbelt, smoking, Distracted Behind / Left / Phone / Right** — the only public model found that covers smoking + phone + drinking + reaching back simultaneously.
- **driver-monitoring/dmd-tfiw0** (universe.roboflow.com/driver-monitoring/dmd-tfiw0) — YOLOv8n, **CC BY 4.0**; classes DangerousDriving / Distracted / Drinking / SafeDriving / SleepyDriving / Yawn.
- **yolov8-ei4l6/distracted-driving-yolov8** — **CC BY 4.0**; classes Hand-on-Wheel / Texting / Calling / Drinking / Reach-Behind (directly matches three of your four behaviors).
- **arnabdhar/YOLOv8-Face-Detection (HuggingFace)** — YOLOv8 face detector (10k+ faces, V100-trained) for the ROI stage; loads via `hf_hub_download` + Ultralytics `YOLO()`.
All Roboflow YOLOv8 models deploy on NVIDIA T4 GPUs (Roboflow explicitly lists T4 as a target) and export as `.pt` for fully offline Ultralytics inference.

## Details

### 1. Datasets — full detail
**DMD (Driver Monitoring Dataset).** Download via dmd.vicomtech.org (Box account + form); tooling at github.com/Vicomtech/DMD-Driver-Monitoring-Dataset. Per Ortega et al., ECCV 2020 Workshops (arXiv:2008.12085): "distraction... in 41 hours of RGB, depth and IR videos from 3 cameras capturing face, body and hands of 37 drivers." Three RealSense cameras (face frontal, hands from back, body from left side). The behaviour subset **dBehaviourMD has 13 distraction activities** — safe driving, texting L/R, phone call L/R, radio, drinking, reach side, hair & makeup, talking to passenger, reach backseat, change gear, stand-still — and **smoking is absent across all seven annotation levels and all object labels (only Cellphone, Hair Comb, Bottle)**. The official split is **driver-disjoint: 24 train / 7 validation / 6 test drivers** ("the training, validation, and testing clip sets are split... which corresponds to 24:7:6 drivers"). Within-DMD top-1 accuracy is ~88–95% with real-time CPU (OpenVINO) deployment. **License: data is CC BY-NC-ND 4.0 (non-commercial, no-derivatives); the TaTo/DEx Python tools are MIT — do not conflate the two.** Per the official site's 2025/2026 notice, simulator, IR and depth material are being removed, leaving RGB-only and a reduced participant set. Best "real driving" academic option, but non-commercial and smoking-free. (DMD's own paper does **not** report a formal train-on-DMD/test-elsewhere cross-dataset number.)

**AUC Distracted Driver (V1/V2).** heshameraqi.github.io/distraction_detection. V1: 14,478 frames, 44 participants, 10 classes (incl. Drinking 1,076 and Reaching Behind 1,034). V2: 17,308 images (12,977 train / 4,331 test) from 31 participants of 7 countries (22 male, 9 female). Side-view RGB. **License (V1/V2 agreement): "Any commercial use of the dataset is strictly prohibited. Commercial use includes, but is not limited to: Testing commercial systems... Selling data... Broadcasting data from the dataset."** No official subject split; some studies report label/test-set quality issues. No smoking.

**SynDD1 / SynDD2 (NHTSA).** SynDD1 in *Data in Brief* (PMC9730022); SynDD2 = 7th AI City Challenge Track 3. Three synchronized in-vehicle IR cameras (dashboard, rearview, right-window). SynDD2 has 16 activities including Drinking, Phone Call L/R, Text L/R, Reaching behind, Yawning — **no smoking**. Driver-disjoint **A1/A2/B** split (e.g., the 2024 set: 99 drivers split 69/15/15). As US-government-produced naturalistic data it is the most commercially defensible source. Access via aicitychallenge.org data pages (registration/agreement). Note AI City rules require public-only data for leaderboard eligibility.

**Drive&Act.** driveandact.com. 15 drivers; 5 NIR cameras + depth + RGB + 3D skeleton; 83 hierarchical classes (12/34/six-triplet levels); three predefined splits with **no driver overlap**, plus an explicit cross-view benchmark. **"Copyright Fraunhofer IOSB. Usage for research only."** Recorded in a stationary lab simulator, so transfer to real driving is weaker (relaxed behavior, no real illumination/weather, per arXiv:2408.09833). No smoking.

**100-Driver.** 100-driver.github.io (paper PDF + license form; code at github.com/Shenqishaonv/100-Driver-Source). 470,208 images, 100 drivers, 4 cameras, RGB+NIR, 5 vehicles (2 sedans, 2 SUVs, 1 van), day+night, 22 classes (21 distracted + normal). Four evaluation protocols including cross-vehicle/cross-view/cross-modality — the strongest public generalization testbed. Download requires a signed academic license form emailed to the authors.

### 2. ROI approach — how to build it
**Stage 1 (localize):** run MediaPipe Pose/Holistic (or a YOLOv8 person/hand/face detector) to obtain head and right-lower-hand ROIs. MediaPipe's detector→tracker is efficient, runs the detector only when tracking is lost, and is fully offline. **Stage 2 (classify):** crop those ROIs and feed a YOLOv8 classifier/detector. This reproduces jacobkie's "head + radio-area" crops and forces behavior features over cabin background. For phone/drinking/smoking specifically, the hand+face crop is where the discriminative object (phone, bottle, cigarette) appears; for reaching back, the full-body/pose channel matters, so keep a whole-frame arm in the ensemble as shivsondhi does.

### 3. Augmentation — concrete recipe
Apply: HSV/brightness/contrast jitter (day/night/IR), random rotation/translation/scale, Mosaic + MixUp, CutMix / Hide&Seek random occlusion of the cabin, Gaussian noise, and ideally background replacement / domain randomization so the model cannot memorize specific cabins. Wrap the variation under a RandAugment policy. **Crucially, validate gains on a held-out driver + vehicle, never on random frames** — otherwise augmentation "improvements" are measured against a leaky baseline.

### 4. Pretrained weights — deployment
Download the Roboflow `.pt` weights and run with Ultralytics offline on the T4 (`model = YOLO('weights.pt')`). CC BY 4.0 models permit commercial use with attribution; the "DRIVER MENTORING" project is the closest class match because it includes smoking. Treat all of these as **seeds, not finished models** — none publishes a documented subject-disjoint evaluation, so they may carry the same leakage you are trying to escape. Re-validate every candidate on your own held-out drivers and vehicles before trusting it.

### 5. Reference "done-right" repos
- **bdutta19/kaggle_statefarm** — documents jacobkie's ROI + KNN + segment-average pipeline.
- **saicharan/Distracted-Driver-MultiAction-Classification** (+ Pachigolla TDS article) — the person-ID split fix and the honest 38.5% reality check.
- **shivsondhi/distracted-driver-detection** — hand/face ROI three-arm ensemble.
- **congduan-HNU/SSoftmax** — cross-dataset generalization classifier (Score-Softmax).
- **e-candeloro/Driver-State-Detection (MIT)** — clean MediaPipe real-time DMS baseline.

## Recommendations
1. **Re-measure honestly first (this week).** Re-split your existing State Farm + Roboflow data by driver/clip ID and re-evaluate. Expect a large drop; this is your true baseline. **Decision threshold: if subject-disjoint mAP is below ~70%, the model is not deployable** regardless of its old 99% number.
2. **Adopt the two-stage ROI pipeline.** Stage-1 MediaPipe/YOLO head+hand+face detector → Stage-2 YOLOv8 classifier on crops, with one whole-frame arm retained for "reaching back." Keep ROI only if it beats whole-frame on the *cross-driver* metric.
3. **Train on diverse, properly-split data.** For research/prototyping combine 100-Driver (generalization) + DMD/SynDD (real driving). For a *shippable commercial* model rely on **SynDD (US-gov)** + **CC BY 4.0 Roboflow data** + **your own collected & labeled footage** — mandatory for **smoking, which no major public set annotates.**
4. **Augment for domains, not just images:** domain randomization + RandAugment + Mosaic/MixUp + cabin occlusion; consider PQ-DAF-style pose-conditioned synthesis if smoking samples are scarce.
5. **Resolve licensing before shipping.** **Ultralytics YOLOv8 is AGPL-3.0** — a closed-source commercial product needs a commercial Ultralytics license. DMD, AUC, Drive&Act, and 100-Driver are all non-commercial; use them only to prototype and train shippable models on SynDD + CC BY 4.0 + owned data.
6. **Promotion gate:** only ship a model that holds up on **unseen drivers AND an unseen vehicle/camera angle** simultaneously — the two failure axes (driver appearance, cabin/viewpoint) that broke your first model.

## Caveats
- **No public dataset cleanly covers all four target behaviors; smoking specifically must be self-collected/labeled or sourced from the Roboflow "DRIVER MENTORING" project** (whose split quality is undocumented).
- Roboflow community weights rarely disclose split methodology — assume their headline metrics are optimistic until you re-test on held-out drivers.
- DMD's ongoing data reduction (moving to RGB-only, removing simulator/IR/depth and some participants) means the currently downloadable dataset differs from the 2020 paper's description — verify before relying on IR/depth.
- Cross-dataset numbers vary widely by method and protocol; treat any single reported figure (including the ones above) as indicative, not guaranteed, and reproduce on your own data.
- DMD's paper does not provide a formal cross-dataset transfer score, so its real-world generalization advantage over AUC/Drive&Act is inferred from its real-driving recording condition, not from a published head-to-head transfer benchmark.