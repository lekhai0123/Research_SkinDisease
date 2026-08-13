import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skin_disease import config

DATASET_SLUG = "kelixo25/31-classes-of-skin-disease"
DATASET_SUBFOLDER = "Atlas dan ISIC2019 (31 classes)"


def download_dataset():
    config.DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KAGGLEHUB_CACHE", str(config.DEFAULT_DATA_DIR))

    import kagglehub

    path = Path(kagglehub.dataset_download(DATASET_SLUG))
    dataset_root = path / DATASET_SUBFOLDER
    return dataset_root if dataset_root.exists() else path


if __name__ == "__main__":
    dataset_root = download_dataset()
    print("Path to dataset files:", dataset_root)
