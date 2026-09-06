"""Numerical/inductive tests for novel decomposition code."""
import numpy as np
from tensorly.decomposition import tensor_train
from tensorly.tt_tensor import tt_to_tensor
from threadpoolctl import threadpool_limits
from models import TTProjection,TuckerProjection,anatomical_smoothing

def main():
    rng=np.random.default_rng(504)
    x=rng.normal(size=(15,4,5,3))
    test=rng.normal(size=(3,4,5,3))
    m=TTProjection(rank=7,channel_rank=3,spectral_rank=5).fit(x)
    assert m.orthogonality_error_<1e-10
    z=np.moveaxis(x-m.mean_,0,-1)
    reference=tensor_train(z,rank=[1,3,5,7,1])
    expected=tt_to_tensor(reference)
    actual=np.moveaxis((m.transform(x)@m.basis_.T).reshape(x.shape),0,-1)
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<1e-9
    before=m.basis_.copy()
    assert np.allclose(m.transform(test),np.vstack([m.transform(t[None]) for t in test]))
    assert np.array_equal(before,m.basis_)
    full=TTProjection(rank=15,channel_rank=4,spectral_rank=20).fit(x)
    assert np.allclose(full.transform(x)@full.basis_.T,(x-full.mean_).reshape(15,-1))
    tuk=TuckerProjection((4,5,3)).fit(x)
    assert tuk.orthogonality_error_<1e-10
    assert np.allclose(tuk.transform(x)@tuk.basis_.T,(x-tuk.mean_).reshape(15,-1))
    coords=rng.normal(size=(19,3))
    smoother,a=anatomical_smoothing(coords)
    assert np.allclose(a,a.T) and np.linalg.eigvalsh(smoother).min()>0
    print('PASS: TT equals TensorLy TT-SVD; orthogonality; full-rank reconstruction; inductive batch invariance; Tucker; graph symmetry.')

if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
