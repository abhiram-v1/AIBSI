"""Split the full moderate-cleaning log-mel dataset into Drive-sized chunks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "neural_comparison/advanced_clean_spectral/outputs/moderate_log_mel/prepared"
CHUNK_ROWS = 700


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    log_mel = np.load(SOURCE / "log_mel.npy", mmap_mode="r")
    labels = np.load(SOURCE / "labels.npy", mmap_mode="r")
    subject_index = np.load(SOURCE / "subject_index.npy", mmap_mode="r")
    assert log_mel.shape == (6721, 19, 24, 13)
    assert labels.shape == subject_index.shape == (6721,)
    original = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    manifest = {k: v for k, v in original.items() if k != "participants"}
    manifest["participants"] = [{k: v for k, v in p.items() if k != "path"} for p in original["participants"]]
    manifest["source_array"] = "moderate_log_mel/prepared/log_mel.npy"
    manifest["data_kind"] = "full prepared derived log-mel dataset; not raw EEG"
    manifest["label_encoding"] = {"0": "AD", "1": "FTD", "2": "HC"}
    manifest["chunks"] = []
    for part, start in enumerate(range(0, len(labels), CHUNK_ROWS), 1):
        stop = min(start + CHUNK_ROWS, len(labels))
        name = f"06_logmel_full_part_{part:02d}.npz"
        path = OUT / name
        np.savez_compressed(path, log_mel=np.asarray(log_mel[start:stop]),
                            labels=np.asarray(labels[start:stop]),
                            subject_index=np.asarray(subject_index[start:stop]))
        manifest["chunks"].append({"file": name, "start": start, "stop": stop,
                                   "rows": stop - start, "bytes": path.stat().st_size,
                                   "sha256": sha256(path)})
    (OUT / "06_complete_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"chunks": len(manifest["chunks"]), "rows": sum(x["rows"] for x in manifest["chunks"]),
                      "total_mb": round(sum(x["bytes"] for x in manifest["chunks"]) / 1048576, 1)}, indent=2))


if __name__ == "__main__":
    main()
