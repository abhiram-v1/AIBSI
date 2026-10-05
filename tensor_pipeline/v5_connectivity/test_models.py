from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import LogEuclideanTangent, candidate_configs, sample_weights


def main():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(12, 4, 19, 25))
    covariance = np.einsum("nbct,nbdt->nbcd", a, a) / 25 + 0.1 * np.eye(19)[None, None]
    mapper = LogEuclideanTangent().fit(covariance[:8])
    z = mapper.transform(covariance)
    assert z.shape == (12, 4 * 190) and np.isfinite(z).all()
    y = np.repeat([0, 1, 2], 4)
    subjects = np.array([f"s{i // 2}" for i in range(12)])
    w = sample_weights(y, subjects)
    assert np.isfinite(w).all() and np.all(w > 0)
    configs = candidate_configs()
    assert len(configs) == 150 and len({str(sorted(c.items())) for c in configs}) == len(configs)
    print("V5 model tests passed")


if __name__ == "__main__":
    main()
