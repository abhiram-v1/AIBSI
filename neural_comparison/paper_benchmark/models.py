"""Documented multiclass adaptations of the four cited literature methods."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from scipy.special import ndtr
from scipy.stats import kstwo, mannwhitneyu, ttest_ind
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'neural_comparison/graph_ntf_svm'))
from core import GraphNTF


class CP6:
    """Six nonnegative CP components, fitted without any held-out participants."""
    def __init__(self, seed=42, maxiter=500, starts=3):
        self.seed, self.maxiter, self.starts = seed, maxiter, starts

    def normalized(self, x):
        return np.clip((x - self.minimum_) / self.range_, 0, 1)[..., None]

    def fit(self, x):
        self.minimum_ = float(np.min(x))
        self.range_ = max(float(np.max(x) - self.minimum_), 1e-8)
        self.factorizer_ = GraphNTF(rank=6, graph_lambda=0, maxiter=self.maxiter,
                                   starts=self.starts, seed=self.seed)
        self.factorizer_.fit(self.normalized(x), np.zeros((19, 19)))
        return self

    def transform(self, x):
        return self.factorizer_.transform(self.normalized(x))


class StatisticalSelector(BaseEstimator, TransformerMixin):
    """Train-only KS normality, then Welch t or Mann-Whitney ranking."""
    def __init__(self, k=100):
        self.k = k

    @staticmethod
    def normality(values):
        z = np.sort((values-values.mean(0))/np.maximum(values.std(0), 1e-12), axis=0)
        n = len(values)
        cdf = ndtr(z)
        d = np.maximum((np.arange(1, n+1)[:, None]/n-cdf).max(0),
                       (cdf-np.arange(n)[:, None]/n).max(0))
        return kstwo.sf(d, n) > .05

    def fit(self, x, y):
        a, b = x[y == 0], x[y == 1]
        normal = self.normality(a) & self.normality(b)
        with np.errstate(all='ignore'):
            t = ttest_ind(a, b, axis=0, equal_var=False).pvalue
            u = mannwhitneyu(a, b, axis=0, method='asymptotic').pvalue
        self.pvalues_ = np.nan_to_num(np.where(normal, t, u), nan=1.)
        k = x.shape[1] if self.k == 'all' else min(int(self.k), x.shape[1])
        self.indices_ = np.argsort(self.pvalues_, kind='stable')[:k]
        return self

    def transform(self, x):
        return x[:, self.indices_]


class HierarchicalSVM:
    """Dementia vs HC gate, then AD vs FTD; gives a final three-class label."""
    def __init__(self, k=100, C=1., kernel='rbf'):
        self.k, self.C, self.kernel = k, C, kernel

    def fit(self, x, y):
        from sklearn.pipeline import Pipeline
        self.models_ = []
        for stage in range(2):
            use = np.arange(len(y)) if stage == 0 else np.flatnonzero(y != 2)
            target = (y[use] != 2).astype(int) if stage == 0 else y[use]
            pipe = Pipeline([('selection', StatisticalSelector(self.k)), ('scale', StandardScaler()),
                             ('svm', SVC(C=self.C, kernel=self.kernel, class_weight='balanced', gamma='scale'))])
            pipe.fit(x[use], target)
            self.models_.append(pipe)
        return self

    def predict(self, x):
        dementia = self.models_[0].predict(x)
        subtype = self.models_[1].predict(x)
        return np.where(dementia == 0, 2, subtype)


def sinusoidal(length, channels):
    size = int(np.ceil(channels / 2) * 2)
    angle = torch.arange(length)[:, None] / (10000 ** (torch.arange(0, size, 2)/size))[None, :]
    encoded = torch.stack([angle.sin(), angle.cos()], dim=-1).flatten(1)
    return encoded[:, :channels]


class DICENet(nn.Module):
    """Author's late-concatenation architecture, corrected batch axis, 3 logits."""
    def __init__(self, frames=30):
        super().__init__()
        length = frames - 4
        self.conv = nn.ModuleList([nn.Conv2d(19, 19, (5, 5), groups=19) for _ in range(2)])
        self.tokens = nn.ParameterList([nn.Parameter(torch.randn(1, length, 1)) for _ in range(2)])
        self.register_buffer('position', sinusoidal(length, 19)[None])
        self.encoders = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=20, nhead=2, batch_first=True),
                                  num_layers=1, enable_nested_tensor=False) for _ in range(2)])
        self.head = nn.Sequential(nn.LayerNorm(length*2), nn.Dropout(.2), nn.Linear(length*2, 24),
                                  nn.BatchNorm1d(24), nn.ReLU(), nn.Dropout(.2), nn.Linear(24, 3))

    def forward(self, inputs):
        branches = []
        for j in range(2):
            value = torch.nn.functional.gelu(self.conv[j](inputs[:, j])).squeeze(2).transpose(1, 2)
            value = value + self.position
            value = torch.cat([self.tokens[j].expand(len(value), -1, -1), value], dim=-1)
            branches.append(self.encoders[j](value)[:, :, 0])
        return self.head(torch.cat(branches, dim=1))


class NFResidual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(nn.Conv1d(in_channels, out_channels, 7, padding=3), nn.ReLU(),
                                  nn.Conv1d(out_channels, out_channels, 7, padding=3))
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        return torch.relu((self.main(x)+self.shortcut(x)) / np.sqrt(2))


class MicrostateNFNet(nn.Module):
    """Supplement S2 widths, main-text optimizer/AGC; two binary stages."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(8, 64, 15, stride=2, padding=7), nn.ReLU(),
                                  NFResidual(64, 128), NFResidual(128, 256), nn.MaxPool1d(4),
                                  nn.Conv1d(256, 64, 1))
        self.attention = nn.MultiheadAttention(64, 4, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(.5), nn.Linear(64, 2))

    def forward(self, x):
        z = self.stem(x).transpose(1, 2)
        attended, _ = self.attention(z, z, z, need_weights=False)
        return self.head((z+attended).mean(1))


def adaptive_gradient_clip(model, threshold=.04, eps=1e-3):
    for p in model.parameters():
        if p.grad is None:
            continue
        dims = tuple(range(1, p.ndim)) if p.ndim > 1 else None
        pn = torch.linalg.vector_norm(p.detach(), dim=dims, keepdim=p.ndim > 1).clamp(min=eps)
        gn = torch.linalg.vector_norm(p.grad, dim=dims, keepdim=p.ndim > 1).clamp(min=1e-8)
        p.grad.mul_(torch.minimum(torch.ones_like(gn), threshold*pn/gn))


def fit_templates(peak_maps, owners, train_subjects, seed=42, starts=5, maxiter=40):
    """Peak maps = [window, band, peak, electrode]. Only training owners enter."""
    rng = np.random.default_rng(seed)
    indices = []
    for subject in train_subjects:
        available = np.flatnonzero(owners == subject)
        indices.extend(available[np.linspace(0, len(available)-1, min(8, len(available))).round().astype(int)])
    templates = []
    diagnostics = []
    for band in range(4):
        maps = peak_maps[indices, band].reshape(-1, 19).astype(np.float32)
        maps = maps[np.linalg.norm(maps, axis=1) > .1]
        if len(maps) > 24000:
            maps = maps[rng.choice(len(maps), 24000, replace=False)]
        best, best_score, history = None, -np.inf, []
        for start in range(starts):
            current = maps[rng.choice(len(maps), 4, replace=False)].copy()
            old = None
            for iteration in range(maxiter):
                assignment = np.abs(maps @ current.T).argmax(1)
                if old is not None and np.array_equal(assignment, old):
                    break
                old = assignment
                for k in range(4):
                    members = maps[assignment == k]
                    if len(members):
                        _, vectors = np.linalg.eigh(members.T @ members)
                        current[k] = vectors[:, -1]
            score = float((np.max(np.abs(maps @ current.T), axis=1)**2).mean())
            history.append({'start': start, 'iterations': iteration+1, 'mean_squared_correlation': score})
            if score > best_score:
                best, best_score = current.copy(), score
        if band:
            left, right = linear_sum_assignment(-np.abs(templates[0] @ best.T))
            aligned = np.empty_like(best); aligned[left] = best[right]; best = aligned
        templates.append(best)
        diagnostics.append(history)
    return np.asarray(templates), diagnostics


def microstate_sequences(filtered_maps, templates, batch=256):
    """Absolute spatial correlations. No labels or population estimates."""
    result = np.empty((len(filtered_maps), 12, filtered_maps.shape[2]), np.float32)
    for start in range(0, len(result), batch):
        # filtered_maps: window x 4bands x time x 19, normalized per time sample
        maps = np.asarray(filtered_maps[start:start+batch], np.float32)
        for j, band in enumerate([1, 2, 3]):  # delta/theta/alpha, broadband is alignment anchor
            result[start:start+len(maps), j*4:(j+1)*4] = np.abs(maps[:, band] @ templates[band].T).transpose(0, 2, 1)
    return result
