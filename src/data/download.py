"""Fetch UCI's official archive; record provenance and validate ZIP contents."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
from zipfile import ZipFile

DATASET_URL = "https://archive.ics.uci.edu/static/public/217/reuter+50+50.zip"


def download(root: str | Path = "data/raw") -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "C50.zip"
    if not path.exists():
        request = Request(DATASET_URL, headers={"User-Agent": "AP-LDA-research/0.1"})
        with urlopen(request, timeout=90) as response:
            content = response.read()
        temporary = path.with_suffix(".download")
        temporary.write_bytes(content)
        with ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("Archive CRC check failed")
        temporary.replace(path)
    with ZipFile(path) as archive:
        n_documents = sum(name.endswith(".txt") for name in archive.namelist())
        if n_documents != 5000:
            raise ValueError(f"Unexpected archive: {n_documents} documents")
    metadata = {"url": DATASET_URL,
        "landing_page": "https://archive.ics.uci.edu/dataset/217/reuter+50+50",
        "citation": "Liu, Z. (2006). Reuter_50_50. DOI: 10.24432/C5DS42",
        "license_on_uci": "CC BY 4.0", "documents": n_documents,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "verified_at_utc": datetime.now(timezone.utc).isoformat()}
    (root / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(download())
