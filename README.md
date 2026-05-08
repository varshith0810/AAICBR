# AI-Assisted Breed Recognition for Indian Cattle and Buffaloes

## Updated behavior (as requested)

Now the script does the following in sequence:
1. **Asks dataset path** before training (if `--dataset_dir` is not provided).
2. Trains model.
3. **Asks image path** after training and predicts breed immediately.

## Run in Colab

### 1) Install dependencies
```bash
!pip install -r requirements.txt
```

### 2) Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3) Run full flow (interactive prompts enabled)
```bash
!python colab_breed_recognition.py --mode all --work_dir /content
```

It will ask:
- Dataset directory path (`train/test` or `breeds/train/test`)
- Image path for single prediction after training

### Optional (skip prompt by passing dataset path)
```bash
!python colab_breed_recognition.py --mode all --dataset_dir "/content/drive/MyDrive/datasets/breeds" --work_dir /content
```

## Launch upload-based app anytime
```bash
!python colab_breed_recognition.py --mode app --dataset_dir "/content/drive/MyDrive/datasets/breeds" --work_dir /content
```

## Dataset folder format expected
Either:
- `/.../breeds/train` and `/.../breeds/test`
- `/.../train` and `/.../test`

## Notes
- Software-only model.
- Upload-based UI remains available through Gradio app mode.
