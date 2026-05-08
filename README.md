# AI-Assisted Breed Recognition for Indian Cattle and Buffaloes (Colab Training-Only)

This repo is now simplified to focus on **training in Google Colab** and **exporting low-hardware model artifacts**.
Production deployment files were removed for now.

## Kept files (only required for training/export)
- `colab_breed_recognition.py` (single-file Colab training + prediction flow)
- `src/config.py`
- `src/preprocess.py`
- `src/train.py`
- `src/infer.py`
- `scripts/export_low_hardware.py`
- `requirements.txt`

---

## Colab training steps

### 1) Clone repo
```bash
!git clone <YOUR_REPO_URL>
%cd AI-ASSISTED-BREED-RECOGNITION-FOR-INDIAN-CATTLE-AND-BUFFALOES
```

### 2) Install dependencies
```bash
!pip install -r requirements.txt
```

### 3) Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4) Run full flow
```bash
!python colab_breed_recognition.py --mode all --work_dir /content
```
It will ask:
- dataset directory path
- image path for a post-training prediction

(Or pass dataset path directly)
```bash
!python colab_breed_recognition.py --mode all --dataset_dir "/content/drive/MyDrive/datasets/breeds" --work_dir /content
```

---

## Extract model for low hardware

After training, run:
```bash
python scripts/export_low_hardware.py --model_path models/breed_classifier.pt --classes_path models/class_names.json --out_dir models
```

Generated files:
- `models/breed_classifier_int8.pt`
- `models/breed_classifier_ts.pt`
- `models/breed_classifier.onnx`
- `models/class_names.json`

Minimal deploy bundle:
```bash
tar -czf cattle_model_low_hw.tar.gz models/breed_classifier_int8.pt models/class_names.json
```

---

## Next phase
After you finalize the trained/extracted model, we can rebuild a clean production deployment project for Railway.
