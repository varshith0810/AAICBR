from pathlib import Path

import zipfile
from collections import Counter

from src.config import Paths, BREEDS



def resolve_breeds_root(dataset_dir: Path) -> Path:
    if (dataset_dir / "train").exists() and (dataset_dir / "test").exists():
        return dataset_dir
    if (dataset_dir / "breeds").exists() and (dataset_dir / "breeds" / "train").exists() and (dataset_dir / "breeds" / "test").exists():
        return dataset_dir / "breeds"
    raise FileNotFoundError(
        f"Invalid dataset_dir: {dataset_dir}. Expected train/test or breeds/train + breeds/test."
    )

def unzip_dataset(zip_path: Path, target_dir: Path) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir.parent)



def validate_structure(root: Path) -> dict:
    expected_splits = ["train", "test"]
    report = {"missing_splits": [], "missing_breeds": {}, "counts": {}}

    for split in expected_splits:
        split_dir = root / split
        if not split_dir.exists():
            report["missing_splits"].append(split)
            continue

        existing = {p.name.lower() for p in split_dir.iterdir() if p.is_dir()}
        missing = sorted(set(BREEDS) - existing)
        report["missing_breeds"][split] = missing

        counter = Counter()
        for breed_dir in split_dir.iterdir():
            if breed_dir.is_dir():
                counter[breed_dir.name.lower()] = len([x for x in breed_dir.iterdir() if x.is_file()])
        report["counts"][split] = dict(counter)

    return report



def main(dataset_dir: str = ""):
    if not dataset_dir:
        raise ValueError("Pass dataset_dir path that contains train/test folders.")
    root = resolve_breeds_root(Path(dataset_dir))
    report = validate_structure(root)

def main():
    paths = Paths()
    if not paths.data_zip.exists():
        raise FileNotFoundError(f"Dataset zip not found: {paths.data_zip}")

    unzip_dataset(paths.data_zip, paths.extracted_data)
    report = validate_structure(paths.extracted_data)

    print("Preprocessing completed.")
    print(report)


if __name__ == "__main__":

    # Example: python -m src.preprocess /content/drive/MyDrive/datasets/breeds
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    main(arg)

    main()

