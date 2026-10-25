# OilTankSight-AI

**Automated Oil Storage Tank Detection from Satellite Imagery using YOLOv8m**

> *Fine-tuned YOLOv8m detector on Airbus SPOT satellite imagery (1.2 m/pixel) to enable scalable, repeatable global oil inventory monitoring.*

## Performance at a Glance

Oil storage tanks range from \~10m to \~120+m in diameter. Upgrading from the baseline YOLOv5 to an **anchor-free YOLOv8m** naturally handles this 12× size variation, pushing our mAP@0.5 to **92.18%** (+3.8% over baseline).

| Metric | Value | Metric | Value | 
| ----- | ----- | ----- | ----- | 
| **mAP@0.50** | `92.18%` | **Precision** | `77.84%` | 
| **mAP@0.50:0.95** | `63.72%` | **Recall** | `84.58%` | 
| **F1-Score** | `81.07%` | **AUC-PR** | `79.38%` | 

*⚠️ All metrics evaluated strictly on a held-out test set of 275 tiles, never seen during training.*

## Visual Showcase

### Model Predictions vs. Ground Truth

*(Green = Ground Truth Labels | Cyan = YOLOv8m Predictions)*

### Evaluation Analytics

## Architecture & Pipeline

```mermaid
flowchart TD
    A[Raw SPOT Scenes<br>98 JPEGs, 2560x2560 px] --> B{CLAHE Enhancement<br>Optional / LAB Color}
    B --> C[Sliding-Window Tiling<br>640px, overlap 64px]
    C --> D[Annotation Clipping<br>Discard <30% visible]
    D --> E[Stratified Split<br>70% Train | 20% Val | 10% Test]
    E --> F[(YOLOv8m Training<br>100 Epochs, AMP, AdamW)]
    F --> G([Test-Set Evaluation<br>275 held-out tiles])
    G --> H[/Hough Circle Fitting<br>Post-Processing for Radius/]
    
    style F fill:#00d4ff,stroke:#000,stroke-width:2px,color:#000
    style H fill:#ff9900,stroke:#000,stroke-width:2px,color:#000
```

### BONUS: Circle Fitting Post-Processing

Oil tanks appear as near-perfect circles from nadir satellite view. After bounding-box detection, a two-stage circle fit is applied:

1. **Hough Circle Transform:** On the CLAHE-enhanced crop bounded by the YOLO prediction.

2. **Contour Circularity Fallback:** If Hough fails, accepts the largest contour if `circularity > 0.45`.

Combined with solar azimuth from EXIF metadata, this physical radius estimation (`radius_px × 1.2 m/px`) enables **automated fill-level estimation**.

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/architjha15/OilTankSight-AI.git
cd OilTankSight-AI

# Recommended: Use uv for exact environment matching
uv venv oilDec_env --python 3.11
source oilDec_env/bin/activate
uv pip install -r requirements.txt
```

### 2. Data Preparation

*Requires Kaggle API credentials (`kaggle.json`) configured.*

```bash
# Download + tile (640px) + stratified split + CLAHE enhancement
python scripts/prepare_data.py --download --clahe
```

### 3. Training & Evaluation

```bash
# Train on GPU (100 epochs, batch 16)
python scripts/train.py

# Evaluate strictly on held-out test set
python scripts/evaluate.py --weights best.pt
```

### 4. Inference (Predict on new imagery)

```bash
# Detect + apply Hough circle fitting for physical radius
python scripts/infer.py --weights best.pt --source image.jpg --circle-fit
```

## Roadmap & Future Work

* [ ] **1024×1024 Tiling:** Double the pixel representation of XS (<24m) tanks to combat JPEG compression artifacts.
* [ ] **Test-Time Augmentation (TTA):** Boost mAP by ~2-3% at zero training cost.
* [ ] **Volume Estimation:** Combine detected radius + shadow angle + solar elevation to calculate cylindrical volume.
* [ ] **SAR Fusion:** Utilize Sentinel-1 to confirm detections through heavy cloud cover.