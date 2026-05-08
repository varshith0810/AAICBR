from pathlib import Path
import zipfile
from collections import Counter

from src.config import Paths, BREEDS


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


def main():
    paths = Paths()
    if not paths.data_zip.exists():
        raise FileNotFoundError(f"Dataset zip not found: {paths.data_zip}")

    unzip_dataset(paths.data_zip, paths.extracted_data)
    report = validate_structure(paths.extracted_data)
    print("Preprocessing completed.")
    print(report)


if __name__ == "__main__":
    main()
