"""Frozen, participant-local inputs for the four literature-method adaptations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, stft
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs/prepared"
REFERENCE = ROOT / "neural_comparison/graph_ntf_svm/outputs/prepared/manifest.json"
sys.path.insert(0, str(ROOT / "neural_comparison/stft_gpu_3class"))
import run_pipeline as legacy

BANDS = [(1., 4.), (4., 8.), (8., 13.), (13., 25.), (25., 40.)]
PAIRS = np.triu_indices(19, 1)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def complexity(v):
    """Katz FD, approximate entropy(m=2,r=.2 SD), bounded Rosenstein LE."""
    x = np.asarray(v[::2], float)  # 64 Hz, 512 samples, no discontinuity joins
    sd = x.std()
    if sd < 1e-15:
        return [1., 0., 0.]
    x = (x - x.mean()) / sd
    length = np.abs(np.diff(x)).sum()
    distance = np.max(np.abs(x - x[0]))
    n = len(x) - 1
    denom = np.log10(n) + np.log10(max(distance / max(length, 1e-15), 1e-15))
    kfd = np.log10(n) / max(denom, 1e-8)
    phi = []
    for dim in [2, 3]:
        emb = np.lib.stride_tricks.sliding_window_view(x, dim).copy()
        counts = cKDTree(emb).query_ball_point(emb, .2, p=np.inf, return_length=True)
        phi.append(float(np.log(np.maximum(counts, 1) / len(emb)).mean()))
    apen = phi[0] - phi[1]
    # Explicit fixed reconstruction and divergence horizon; do not tune on labels.
    dimension, delay, horizon, theiler = 6, 4, 12, 32
    count = len(x) - (dimension - 1) * delay - horizon
    emb = np.stack([x[k * delay:k * delay + count] for k in range(dimension)], axis=1)
    _, idx = cKDTree(emb).query(emb, k=min(80, count))
    valid = np.abs(idx - np.arange(count)[:, None]) > theiler
    eligible = valid.any(axis=1)
    base = np.flatnonzero(eligible)
    nearest = idx[base, valid[eligible].argmax(axis=1)]
    divergence = []
    for step in range(horizon):
        a = np.stack([x[base + step + k * delay] for k in range(dimension)], axis=1)
        b = np.stack([x[nearest + step + k * delay] for k in range(dimension)], axis=1)
        divergence.append(np.log(np.maximum(np.linalg.norm(a - b, axis=1), 1e-8)).mean())
    le = np.polyfit(np.arange(horizon) / 64., divergence, 1)[0] if len(base) else 0.
    return [float(kfd), float(apen), float(le)]


def local_features(windows):
    """No labels, other participants, or learned cohort statistics are inputs."""
    w = windows.astype(np.float64)
    freq, psd = welch(w, fs=128, nperseg=128, noverlap=64, axis=-1)
    keep = (freq >= 1) & (freq <= 40)
    psd_db = 10 * np.log10(np.maximum(psd[..., keep], 1e-30)).mean(axis=0)
    # Both DICE inputs have shape window x electrode x band x 7 frames.
    _, _, spectrum = stft(w, fs=128, nperseg=256, noverlap=128,
                          boundary=None, padded=False, axis=-1)
    ff = np.fft.rfftfreq(256, 1 / 128.)
    power = np.abs(spectrum) ** 2
    bp = np.stack([power[:, :, (ff >= lo) & (ff < hi)].sum(2) for lo, hi in BANDS], axis=2)
    bp /= np.maximum(bp.sum(2, keepdims=True), 1e-30)
    z = mne.time_frequency.tfr_array_morlet(
        windows.astype(float), 128, freqs=[2, 6, 10, 18, 35],
        n_cycles=[2, 4, 7, 7, 7], output='complex', n_jobs=1, verbose='ERROR')
    coherence = []
    for start in range(0, 1024 - 256 + 1, 128):
        zz = z[..., start:start + 256]
        cross = np.einsum('ncft,ndft->nfcd', zz, zz.conj(), optimize=True) / 256
        p = np.mean(np.abs(zz) ** 2, axis=-1).transpose(0, 2, 1)
        co = np.abs(cross) / np.sqrt(np.maximum(p[:, :, :, None] * p[:, :, None, :], 1e-30))
        coherence.append(((co.sum(-1) - np.diagonal(co, axis1=-2, axis2=-1)) / 18).transpose(0, 2, 1))
    dice = np.stack([bp, np.stack(coherence, axis=-1)], axis=1).astype(np.float32)
    del z
    # Four published feature families; summaries remain within windows first.
    features = [w.mean((0, 2)), w.var(2).mean(0)]
    complexity_windows = np.linspace(0, len(w) - 1, min(4, len(w))).round().astype(int)
    broad_and_bands = [w] + [sosfiltfilt(butter(4, [lo, hi], fs=128, btype='bandpass', output='sos'), w, axis=-1) for lo, hi in BANDS]
    for band in broad_and_bands:
        features.append(np.mean(band ** 2, axis=(0, 2)))
        # Maximum normalized cross-correlation over a fixed +/- 0.25s lag.
        vals = []
        for i, j in zip(*PAIRS):
            a, b = band[:, i], band[:, j]
            # FFT vectorized over independent windows (no cross-window correlation).
            cross = np.fft.irfft(np.fft.rfft(a, 2048) * np.fft.rfft(b, 2048).conj(), 2048)
            selected = np.concatenate([cross[:, :33], cross[:, -32:]], axis=1)
            scale = np.sqrt(np.sum(a*a, 1) * np.sum(b*b, 1))
            vals.append((np.max(np.abs(selected), axis=1) / np.maximum(scale, 1e-30)).mean())
        features.append(np.asarray(vals))
        _, _, zf = stft(band, fs=128, nperseg=128, noverlap=64, boundary=None, padded=False, axis=-1)
        cross = np.einsum('ncft,ndft->nfcd', zf, zf.conj(), optimize=True) / zf.shape[-1]
        pp = np.mean(np.abs(zf)**2, axis=-1).transpose(0, 2, 1)
        coh = np.abs(cross)**2 / np.maximum(pp[:, :, :, None]*pp[:, :, None, :], 1e-30)
        features.append(coh.mean((0, 1))[PAIRS])
        comp = np.array([[complexity(band[k, c]) for c in range(19)] for k in complexity_windows])
        features.append(comp.mean(0).ravel())
    handcrafted = np.concatenate(features).astype(np.float32)
    if not all(np.isfinite(v).all() for v in [psd_db, dice, handcrafted]):
        raise ValueError('Nonfinite extracted features')
    return psd_db.astype(np.float32), dice, handcrafted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    parts = args.out / 'participants'; parts.mkdir(exist_ok=True)
    manifest = json.loads(REFERENCE.read_text())
    participants = manifest['participants']
    contract = {'reference_manifest_sha256': sha(REFERENCE), 'preparation_code_sha256': sha(__file__),
                'participants': participants, 'classes': ['AD', 'FTD', 'HC'], 'channels': manifest['channels'],
                'input': 'Existing moderate 8s/128Hz/1-40Hz windows, max40/person; recreated from source EEG',
                'inherited_limit': 'Legacy window selection does not explicitly reject EEGLAB boundary annotations.',
                'adaptations': 'Common 8s representation; DICE uses seven 2s/1s-hop frames instead of30; tensor1-40Hz instead1-45; SVM Butterworth subbands instead of DWT, four evenly spaced windows for nonlinear measures; microstate128Hz/8s instead256Hz/10s.'}
    protocol_file = args.out / 'preparation_protocol.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != contract:
        raise RuntimeError('Preparation contract changed; use another output directory')
    write_json(protocol_file, contract)
    with threadpool_limits(limits=args.threads):
        mne.set_log_level('ERROR')
        for i, person in enumerate(participants):
            if i % args.shards != args.shard:
                continue
            dest = parts / f'{i:03d}.npz'
            if dest.exists():
                continue
            record = legacy.Record(person['subject_id'], person['source'], person['site'], person['label'], person['path'])
            data, header = legacy.load_harmonized(record)
            candidates = [data[:, start:start+1024] for start in range(1280, data.shape[1]-1280-1024+1, 1024)
                          if legacy.valid_window(data[:, start:start+1024])[0]]
            windows = np.stack(legacy.evenly_limit(candidates, 40)).astype(np.float32)
            if len(windows) != person['windows']:
                raise ValueError(f'Frozen window membership changed: {person["subject_id"]}')
            psd, dice, handcrafted = local_features(windows)
            np.savez_compressed(dest, windows=windows, psd_db=psd, dice=dice, handcrafted=handcrafted)
            hashes = {str(Path(person['path'])): sha(person['path'])}
            fdt = Path(person['path']).with_suffix('.fdt')
            if fdt.exists(): hashes[str(fdt)] = sha(fdt)
            write_json(dest.with_suffix('.json'), {'subject_id': person['subject_id'], 'source_hashes': hashes, 'header': header,
                                                  'windows': len(windows), 'array_sha256': hashlib.sha256(windows.tobytes()).hexdigest()})
            print(f'Prepared {i+1}/{len(participants)}: {person["subject_id"]}, {len(windows)} windows', flush=True)
        if args.shards > 1:
            return
        records = [np.load(parts / f'{i:03d}.npz') for i in range(len(participants))]
        total = sum(len(r['windows']) for r in records)
        raw = np.lib.format.open_memmap(args.out/'windows.npy', mode='w+', dtype=np.float32, shape=(total,19,1024))
        dice = np.lib.format.open_memmap(args.out/'dice.npy', mode='w+', dtype=np.float32, shape=(total,2,19,5,7))
        owners, offset, seen = [], 0, {}
        for i, r in enumerate(records):
            w = r['windows']; n = len(w)
            raw[offset:offset+n] = w; dice[offset:offset+n] = r['dice']; owners.extend([i]*n); offset += n
            for window in w:
                key = hashlib.sha256(window.tobytes()).hexdigest()
                if key in seen and seen[key] != i:
                    raise RuntimeError(f'Identical EEG window in different participants: {seen[key]}, {i}')
                seen[key] = i
        raw.flush(); dice.flush()
        np.save(args.out/'owners.npy', np.asarray(owners, np.int32))
        np.savez_compressed(args.out/'subjects.npz', psd_db=np.stack([r['psd_db'] for r in records]),
                            handcrafted=np.stack([r['handcrafted'] for r in records]),
                            y=np.array([['AD','FTD','HC'].index(p['label']) for p in participants]),
                            sites=np.array([p['site'] for p in participants]))
        write_json(args.out/'manifest.json', {**contract, 'window_count': total, 'duplicate_cross_participant_windows': 0,
                    'array_hashes': {name: sha(args.out/name) for name in ['windows.npy','dice.npy','owners.npy','subjects.npz']}})
        print(f'Complete: {len(participants)} participants, {total} windows', flush=True)


if __name__ == '__main__':
    main()
