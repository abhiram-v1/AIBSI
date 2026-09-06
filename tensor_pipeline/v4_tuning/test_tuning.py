"""Smoke tests for installed XGBoost and modified tensor regularization."""
import numpy as np
from threadpoolctl import threadpool_limits
from tune_models import RegularizedTT,weights,fit_classifier,configs
import xgboost


def main():
    rng=np.random.default_rng(9064)
    raw=rng.normal(size=(9,304,20)); x=raw@raw.transpose(0,2,1)/20
    y=np.array([0,1,2]*3)
    model=RegularizedTT(maxiter=2).fit(x,y)
    theta=model.theta_+rng.normal(0,.01,size=len(model.theta_))
    w=weights(y); w/=w.sum()
    loss,grad=model.objective(theta,x,y,w)
    errors=[]
    for _ in range(6):
        d=rng.normal(size=len(theta)); d/=np.linalg.norm(d)
        numeric=(model.objective(theta+1e-5*d,x,y,w)[0]-model.objective(theta-1e-5*d,x,y,w)[0])/2e-5
        errors.append(abs(numeric-grad@d))
    assert max(errors)<1e-6
    cfg=next(c for c in configs() if c['kind']=='xgboost')
    m=fit_classifier(cfg,rng.normal(size=(9,8)),y)
    assert m.predict(rng.normal(size=(3,8))).shape==(3,)
    print(f'PASS: XGBoost {xgboost.__version__} trains on CPU; regularized tensor gradient max error {max(errors):.2e}; {len(configs())} frozen candidates.')


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
