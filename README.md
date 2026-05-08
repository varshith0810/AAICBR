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
python -m src.train
python app.py
```

## Notes
- Software-only model.
- App asks user to upload image.
