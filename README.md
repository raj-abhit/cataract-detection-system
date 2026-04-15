# Cataract Detection (YOLO + Streamlit)

This project trains a YOLO model to detect two classes in eye images:
- `Cataract`
- `Normal`

It also includes a Streamlit UI for testing with:
- image upload
- 10 built-in sample images

## Project Files

- Dataset (prepared split): `cataract.yolov12_ready/`
- Best trained model: `runs/cataract_yolo_s50/weights/best.pt`
- Streamlit app: `app.py`
- Split script: `prepare_yolo_split.py`
- Quick train script (5 epochs): `train_yolo_quick.py`
- Main train script (50 epochs): `train_yolo_s50.py`

## Environment

```bash
python --version
```

```bash
pip install ultralytics streamlit pillow
```

## Dataset Preparation

Creates leakage-safe grouped split (`train/valid/test`) from original export:

```bash
python prepare_yolo_split.py
```

Prepared dataset config:
- `cataract.yolov12_ready/data.yaml`

## Training Commands

### 1) Quick GPU baseline (5 epochs)

```bash
python train_yolo_quick.py
```

Output:
- `runs/cataract_yolo_fast_gpu/`

### 2) Main GPU training (YOLOv8s, 50 epochs)

```bash
python train_yolo_s50.py
```

Output:
- `runs/cataract_yolo_s50/`

## Evaluation Commands

### Validate/Test best model (PowerShell)

```powershell
@'
from ultralytics import YOLO
model = YOLO(r"runs/cataract_yolo_s50/weights/best.pt")
model.val(data=r"cataract.yolov12_ready/data.yaml", split="test", device=0, imgsz=640, batch=16, workers=0)
'@ | python -
```

### Validate/Test best model (cross-platform)

```bash
python -c "from ultralytics import YOLO; YOLO('runs/cataract_yolo_s50/weights/best.pt').val(data='cataract.yolov12_ready/data.yaml', split='test', device=0, imgsz=640, batch=16, workers=0)"
```

## Evaluation Metrics

## Quick Baseline (`cataract_yolo_fast_gpu`, 5 epochs)

- Validation:
  - Precision: `0.799`
  - Recall: `0.806`
  - mAP50: `0.859`
  - mAP50-95: `0.468`
- Test:
  - Precision: `0.817`
  - Recall: `0.773`
  - mAP50: `0.842`
  - mAP50-95: `0.493`

## Main Model (`cataract_yolo_s50`, 50 epochs)

- Validation:
  - Precision: `0.820`
  - Recall: `0.826`
  - mAP50: `0.842`
  - mAP50-95: `0.534`
- Test:
  - Precision: `0.823`
  - Recall: `0.834`
  - mAP50: `0.863`
  - mAP50-95: `0.543`
- Test class-wise:
  - Cataract: `P=0.777`, `R=0.772`, `mAP50=0.852`, `mAP50-95=0.500`
  - Normal: `P=0.868`, `R=0.896`, `mAP50=0.873`, `mAP50-95=0.586`

## Run Streamlit UI

```bash
streamlit run app.py
```

UI features:
- Upload your own image
- Select one of 10 sample images from `examples/`
- View predicted boxes and confidence

## Notes

- The model is a detection model, not a clinical diagnosis system.
- For screening-focused use, tune confidence threshold to favor higher cataract recall.
