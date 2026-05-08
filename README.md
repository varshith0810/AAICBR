# AI-Assisted Breed Recognition for Indian Cattle and Buffaloes


## Google Drive unzipped dataset only (zip code removed)

The pipeline now **does not unzip data**. It expects dataset already extracted in Google Drive.

## Required dataset folder formats
Either of these:
1. `/content/drive/MyDrive/.../breeds/train` and `/content/drive/MyDrive/.../breeds/test`
2. `/content/drive/MyDrive/.../train` and `/content/drive/MyDrive/.../test`

## Colab run order

### 1) Install dependencies
```bash
!pip install -r requirements.txt
```

### 2) Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3) Validate + Train
```bash
!python colab_breed_recognition.py --mode all --dataset_dir "/content/drive/MyDrive/datasets/breeds" --work_dir /content
```

### 4) Launch App
```bash
!python colab_breed_recognition.py --mode app --dataset_dir "/content/drive/MyDrive/datasets/breeds" --work_dir /content
```

## Modular order (optional)
```bash
python -m src.preprocess /content/drive/MyDrive/datasets/breeds
=======
This project implements an **end-to-end software-only breed recognition pipeline** for the following 41 breeds:

`vechur, umblachery, toda, tharparkar, surti, sahiwal, redsindhi, reddane, rathi, pulikulam, ongole, nimari, niliravi, nagpuri, nagori, murrah, mehsana, malnadgidda, krishnavalley, khillari, kherigarh, kenkatha, kasargod, kankrej, kangayam, jersey, jaffrabadi, holsteinfriesian, hariana, hallikar, guernsey, gir, deoni, dangi, brownswiss, bhadawari, bargur, banni, ayrshire, amritmahal, alambadi`.

## What this builds

1. **Data preprocessing**
   - Unzips `dataset.zip`.
   - Expects structure: `breeds/train/<breed_name>/*.jpg` and `breeds/test/<breed_name>/*.jpg`.
   - Validates split and breed coverage.

2. **Model training**
   - Uses transfer learning with `EfficientNet-B0`.
   - Trains multiclass classifier for breed recognition.
   - Saves best weights as `models/breed_classifier.pt` and class map as `models/class_names.json`.

3. **User-facing recognition app**
   - Gradio app asks user to upload an animal image.
   - Returns predicted breed + top-5 confidence scores.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run full pipeline

```bash
./scripts/run_pipeline.sh
```

Or step-by-step:

```bash
python -m src.preprocess

python -m src.train
python app.py
```

## Notes

- Software-only model.
- App asks user to upload image.


- This implementation is strictly **software model only** (no hardware component).
- Place the provided dataset zip at repo root as `dataset.zip` before preprocessing.

