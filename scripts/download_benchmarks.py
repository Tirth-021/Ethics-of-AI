from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path


DATASET_SOURCES = {
    "etth1": {
        "url": "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/ETTh1.csv",
        "filename": "ETTh1.csv",
        "description": "Electricity Transformer Temperature hourly benchmark.",
    },
    "weather": {
        "url": "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/weather.csv",
        "filename": "weather.csv",
        "description": "Weather benchmark from the Autoformer dataset collection.",
    },
    "exchange": {
        "url": "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/exchange_rate.csv",
        "filename": "exchange_rate.csv",
        "description": "Exchange rate benchmark from the Autoformer dataset collection.",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark datasets into ./data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", *DATASET_SOURCES.keys()],
        help="Datasets to download. Defaults to all proposal datasets.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Target directory for downloaded CSV files. Defaults to ./data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they already exist.",
    )
    args = parser.parse_args()

    selected = list(DATASET_SOURCES.keys()) if "all" in args.datasets else args.datasets
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for name in selected:
        spec = DATASET_SOURCES[name]
        target = data_dir / spec["filename"]
        if target.exists() and not args.force:
            print(f"[skip] {target} already exists")
            continue
        _download_file(spec["url"], target)
        print(f"[done] {name}: {target}")


def _download_file(url: str, target: Path) -> None:
    print(f"[download] {url}")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "llm-ts-audit-dataset-downloader/0.1",
        },
    )
    with urllib.request.urlopen(request) as response:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            shutil.copyfileobj(response, handle)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)

