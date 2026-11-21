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


def amplitude_normalizer(x: np.ndarray, subject_index: np.ndarray, mode: str):
    """Return a function mapping a raw window batch to amplitude-normalized windows.

    window_channel: z-score every channel of every window (removes all amplitude cues).
    recording_gain: divide each recording by its own median window RMS (removes only the
    recording-level gain, keeps relative channel and temporal structure).
    Both rescale to the global median RMS so inputs stay in the range EEGPT expects.
    """
    window_rms = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), 500):
        chunk = np.asarray(x[start : start + 500], dtype=np.float32)
        window_rms[start : start + 500] = np.sqrt((chunk ** 2).mean(axis=(1, 2)))
    target = float(np.median(window_rms))
    gain = np.ones(len(x), dtype=np.float32)
    for subject in np.unique(subject_index):
        mask = subject_index == subject
        gain[mask] = target / max(float(np.median(window_rms[mask])), 1e-6)

    def normalize(batch: np.ndarray, start: int) -> np.ndarray:
        if mode == "window_channel":
            batch = batch - batch.mean(axis=2, keepdims=True)
            return batch / np.maximum(batch.std(axis=2, keepdims=True), 1e-6) * target
        return batch * gain[start : start + len(batch), None, None]

    return normalize


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen EEGPT window embeddings")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--normalize", choices=["none", "window_channel", "recording_gain"], default="none")
    args = parser.parse_args()
    out_dir = OUT if args.normalize == "none" else OUT.with_name(f"frozen_embeddings_{args.normalize}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    checkpoint_hash = tea.sha256(tea.CHECKPOINT)
    if checkpoint_hash != tea.EXPECTED_SHA256:
        raise RuntimeError(f"Checkpoint SHA256 mismatch: {checkpoint_hash}")
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.load(tea.PREPARED / "raw_windows.npy", mmap_mode="r")
    normalize = None
    if args.normalize != "none":
        normalize = amplitude_normalizer(x, np.load(tea.PREPARED / "subject_index.npy"), args.normalize)
    manifest = json.loads((tea.PREPARED / "manifest.json").read_text(encoding="utf-8"))
    device = torch.device("cuda")
    encoder = tea.build_encoder().to(device).eval()
    channel_ids = encoder.prepare_chan_ids(manifest["channels"]).to(device)

    embeddings = np.empty((len(x), 4 * 512), dtype=np.float32)
    started = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for start in range(0, len(x), args.chunk_size):
            raw = np.asarray(x[start : start + args.chunk_size], dtype=np.float32)
            if normalize is not None:
                raw = normalize(raw, start)
            batch = torch.from_numpy(raw).to(device)
            encoded = encoder(batch, channel_ids)
            embeddings[start : start + len(batch)] = encoded.float().mean(dim=1).flatten(1).cpu().numpy()
            if (start // args.chunk_size) % 20 == 0:
                print(f"  {start + len(batch)}/{len(x)} windows, {time.time() - started:.0f}s", flush=True)

    if not np.isfinite(embeddings).all():
        raise RuntimeError("Non-finite embeddings")
    np.save(out_dir / "window_embeddings.npy", embeddings)
    meta = {
        "checkpoint_sha256": checkpoint_hash,
        "windows": int(len(x)),
        "dimension": int(embeddings.shape[1]),
        "layout": "4 summary tokens x 512, mean over 16 temporal patches, frozen encoder, eval mode",
        "across_window_std_mean": float(embeddings.std(axis=0).mean()),
        "seconds": time.time() - started,
    }
    meta["amplitude_normalization"] = args.normalize
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
