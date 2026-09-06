"""Independent numerical checks before examining experimental outcomes."""
import numpy as np
from scipy.optimize import check_grad
from wave_models import C,L,weights,DelayProjection,SupervisedTT,energies
from threadpoolctl import threadpool_limits
from tensorly.decomposition import tensor_train
from tensorly.tt_tensor import tt_to_tensor


def main():
    rng = np.random.default_rng(906)
    raw = rng.normal(size=(6,C*L,25))
    cov = raw@raw.transpose(0,2,1)/25
    y = np.array([0,1,2,0,1,2])
    b = rng.normal(size=(C*L,4))
    direct = np.mean(np.einsum('ndt,dk->nkt',raw,b)**2,axis=-1)/np.sum(b*b,axis=0)
    np.testing.assert_allclose(energies(cov,b),direct,rtol=1e-12,atol=1e-12)
    for kind in ['tt','tucker']:
        m = DelayProjection(kind).fit(cov,y)
        np.testing.assert_allclose(m.basis_.T@m.basis_,np.eye(m.basis_.shape[1]),atol=1e-10)
        before = m.basis_.copy()
        np.testing.assert_allclose(m.transform(cov),np.concatenate([m.transform(s[None]) for s in cov]),atol=1e-12)
        assert np.array_equal(before,m.basis_)
    tt = DelayProjection('tt',rank=16,channel_rank=6).fit(cov,y)
    observed = raw.transpose(1,0,2).reshape(C,L,-1)
    reference = tt_to_tensor(tensor_train(observed,rank=[1,6,16,1]))
    reconstructed = (tt.basis_@(tt.basis_.T@observed.reshape(C*L,-1))).reshape(observed.shape)
    np.testing.assert_allclose(reconstructed,reference,atol=1e-8,rtol=1e-8)
    model = SupervisedTT(channel_rank=2,filters=3,maxiter=3).fit(cov,y)
    theta = model.theta_.copy()+rng.normal(0,.01,size=len(model.theta_))
    w = weights(y); w /= w.sum()
    fun = lambda t:model.objective(t,cov,y,w)[0]
    grad = lambda t:model.objective(t,cov,y,w)[1]
    # Directional checks exercise all U,V,head,bias parameters at once.
    g = grad(theta)
    errors = []
    for _ in range(8):
        direction = rng.normal(size=len(theta)); direction /= np.linalg.norm(direction)
        numeric = (fun(theta+1e-5*direction)-fun(theta-1e-5*direction))/2e-5
        errors.append(abs(numeric-g@direction))
    assert max(errors)<1e-6, errors
    z = np.array([0]*10+[1]*4+[2]*7)
    w = weights(z)
    np.testing.assert_allclose([w[z==c].sum() for c in range(3)],len(z)/3)
    print(f'PASS: raw response energy == moment shortcut; orthonormal bases; batch-invariant projection; supervised gradient max error {max(errors):.2e}; equal class loss mass.')


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
