"""Meaningful leakage and inference checks before the bounded benchmark."""
import copy
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from models import CP6, DICENet, MicrostateNFNet, StatisticalSelector, fit_templates
from run import folds, heldout_selection_split, fit_neural
from prepare import OUT


def main():
    rng=np.random.default_rng(17)
    torch.set_num_threads(2)
    torch.manual_seed(17)
    for model,x in [(DICENet(),torch.randn(3,2,19,5,30)),
                    (MicrostateNFNet(),torch.randn(3,8,1024))]:
        model.eval()
        with torch.no_grad():
            batch=model(x);single=torch.cat([model(x[i:i+1]) for i in range(3)])
        assert torch.allclose(batch,single,atol=1e-5,rtol=1e-5)
    print('PASS: both neural models give batch-independent evaluation predictions')
    model=CP6(seed=17,maxiter=12,starts=1).fit(rng.normal(size=(9,19,40)))
    before=copy.deepcopy(model.factorizer_.factors_)
    minimum,scale=model.minimum_,model.range_
    z=model.transform(np.full((2,19,40),10000.))
    assert z.shape==(2,6) and np.isfinite(z).all() and (z>=0).all()
    assert model.minimum_==minimum and model.range_==scale
    assert all(np.array_equal(a,b) for a,b in zip(before,model.factorizer_.factors_))
    print('PASS: held-out tensor transformation does not refit scaling or factors')
    x=rng.normal(size=(20,30));y=np.tile([0,1],10)
    selector=StatisticalSelector(5).fit(x,y);indices=selector.indices_.copy()
    selector.transform(np.full((2,30),1e9))
    assert np.array_equal(selector.indices_,indices)
    print('PASS: statistical feature selection is frozen at held-out transformation')
    owners=np.repeat(np.arange(4),2)
    peaks=rng.normal(size=(8,4,8,19)).astype(np.float32)
    peaks-=peaks.mean(-1,keepdims=True);peaks/=np.linalg.norm(peaks,axis=-1,keepdims=True)
    a,_=fit_templates(peaks,owners,[0,1],seed=17,starts=1,maxiter=3)
    changed=peaks.copy();changed[owners>=2]*=1000
    b,_=fit_templates(changed,owners,[0,1],seed=17,starts=1,maxiter=3)
    assert np.array_equal(a,b)
    print('PASS: changing held-out EEG peak maps cannot change training microstate templates')
    if (OUT/'subjects.npz').exists():
        d=np.load(OUT/'subjects.npz');outer=folds(d['y'],d['sites'],5,20261102)
        covered=np.zeros(174,int)
        for train,test in outer:
            covered[test]+=1
            a,b=heldout_selection_split(train,d['y'],d['sites'],20261102)
            assert not set(a)&set(b) and not (set(a)|set(b))&set(test)
            assert set(a)|set(b)==set(train)
            for c,e in folds(d['y'][train],d['sites'][train],3,42):
                assert not set(train[c])&set(train[e])
                assert not (set(train[c])|set(train[e]))&set(test)
        assert np.all(covered==1)
        reference=Path('neural_comparison/graph_ntf_svm/outputs/full')
        for i,(train,test) in enumerate(outer):
            old=json.loads((reference/f'repeat_00_fold_{i:03d}/gntf/result.json').read_text())
            assert test.tolist()==old['test_indices']
        print('PASS: all174 people covered exactly once; nested memberships disjoint; outer splits match prior repeat0')
    if torch.cuda.is_available():
        # Real backward/optimizer/checkpoint path, with train/validation owners disjoint.
        owners=np.repeat(np.arange(9),8);y=np.tile([0,1,2],3)
        x=torch.randn(72,2,19,5,30,device='cuda')
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            model,epoch=fit_neural(DICENet,x,owners,y,np.arange(6),np.arange(6,9),17,1,1,'dice',0,Path(temp)/'test')
            assert epoch==1
        print('PASS: real CUDA optimization, inner validation, checkpoint save, and batch-invariance replay')
    print('ALL CHECKS PASSED')


if __name__=='__main__':main()
