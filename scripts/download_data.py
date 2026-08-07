"""Download the card-transaction dataset from Kaggle into data/raw/.

Credentials are read from the Kaggle access token at ~/.kaggle/access_token and
are never read from, or written to, this repository.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

DATASET = "ealtman2019/credit-card-transactions"

# The archive is ~2.35 GB and is unzipped in place, so require headroom for both.
REQUIRED_BYTES = 6 * 1024**3


def main() -> int:
    dest = Path(__file__).resolve().parents[1] / "data" / "raw"
    dest.mkdir(parents=True, exist_ok=True)

    free = shutil.disk_usage(dest).free
    if free < REQUIRED_BYTES:
        print(
            f"Insufficient disk: {free / 1024**3:.1f} GiB free, "
            f"need {REQUIRED_BYTES / 1024**3:.1f} GiB",
            file=sys.stderr,
        )
        return 1
    print(f"Disk check passed: {free / 1024**3:.1f} GiB free at {dest}")

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {DATASET} into {dest}")
    api.dataset_download_files(DATASET, path=str(dest), unzip=True, quiet=False)

    print("\nFiles on disk:")
    for path in sorted(dest.iterdir()):
        print(f"  {path.name}  {path.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
