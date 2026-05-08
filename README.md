# AI-Assisted Breed Recognition for Indian Cattle and Buffaloes

## Dataset path prompt behavior (updated)

The code now asks for dataset path if not provided.

- `colab_breed_recognition.py` asks dataset path before training when `--dataset_dir` is missing.
- `src/preprocess.py` asks dataset path interactively if you run without argument.
- `src/train.py` asks dataset path interactively if you run without argument.

## Colab run

```bash
!pip install -r requirements.txt
```

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Full flow with prompt
```bash
!python colab_breed_recognition.py --mode all --work_dir /content
```

### Full flow with explicit path
```bash
!python colab_breed_recognition.py --mode all --dataset_dir "/content/drive/MyDrive/datasets/breeds" --work_dir /content
```

## Modular run

### Option 1 (interactive prompt)
```bash
python -m src.preprocess
python -m src.train
python app.py
```

### Option 2 (pass path explicitly)
```bash
python -m src.preprocess /content/drive/MyDrive/datasets/breeds
python -m src.train /content/drive/MyDrive/datasets/breeds
python app.py
```
