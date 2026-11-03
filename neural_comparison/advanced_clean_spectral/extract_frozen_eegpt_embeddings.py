"""Extract frozen (never fine-tuned) EEGPT embeddings for every accepted window.

The pretrained encoder never sees project labels here, so the embeddings can be
computed once for all 174 participants and reused inside any cross-validation
scheme without label leakage. Output: one 2048-d vector per window
(16 temporal patches averaged, 4 summary tokens x 512 kept).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import train_eegpt_attention as tea


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "eegpt" / "frozen_embeddings"


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen EEGPT window embeddings")
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    checkpoint_hash = tea.sha256(tea.CHECKPOINT)
    if checkpoint_hash != tea.EXPECTED_SHA256:
        raise RuntimeError(f"Checkpoint SHA256 mismatch: {checkpoint_hash}")
    OUT.mkdir(parents=True, exist_ok=True)

    x = np.load(tea.PREPARED / "raw_windows.npy", mmap_mode="r")
    manifest = json.loads((tea.PREPARED / "manifest.json").read_text(encoding="utf-8"))
    device = torch.device("cuda")
    encoder = tea.build_encoder().to(device).eval()
    channel_ids = encoder.prepare_chan_ids(manifest["channels"]).to(device)

    embeddings = np.empty((len(x), 4 * 512), dtype=np.float32)
    started = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for start in range(0, len(x), args.chunk_size):
            batch = torch.from_numpy(np.asarray(x[start : start + args.chunk_size], dtype=np.float32)).to(device)
            encoded = encoder(batch, channel_ids)
            embeddings[start : start + len(batch)] = encoded.float().mean(dim=1).flatten(1).cpu().numpy()
            if (start // args.chunk_size) % 20 == 0:
                print(f"  {start + len(batch)}/{len(x)} windows, {time.time() - started:.0f}s", flush=True)

    if not np.isfinite(embeddings).all():
        raise RuntimeError("Non-finite embeddings")
    np.save(OUT / "window_embeddings.npy", embeddings)
    meta = {
        "checkpoint_sha256": checkpoint_hash,
        "windows": int(len(x)),
        "dimension": int(embeddings.shape[1]),
        "layout": "4 summary tokens x 512, mean over 16 temporal patches, frozen encoder, eval mode",
        "across_window_std_mean": float(embeddings.std(axis=0).mean()),
        "seconds": time.time() - started,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
