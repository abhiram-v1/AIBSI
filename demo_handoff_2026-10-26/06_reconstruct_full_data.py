"""Reconstruct complete prepared arrays after downloading all 06_* files together."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / "06_complete_data_manifest.json").read_text(encoding="utf-8"))
parts = {k: [] for k in ("log_mel", "labels", "subject_index")}
for chunk in manifest["chunks"]:
    path = HERE / chunk["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != chunk["sha256"]:
        raise ValueError(f"Missing or corrupted chunk: {path.name}")
    with np.load(path, allow_pickle=False) as source:
        for key in parts:
            assert len(source[key]) == chunk["rows"]
            parts[key].append(source[key])
for key, arrays in parts.items():
    result = np.concatenate(arrays)
    assert len(result) == 6721
    np.save(HERE / f"{key}.npy", result)
    print(key, result.shape, result.dtype)
