import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from core import GraphNTF,FeatureModel,electrode_graph,unit_columns,ProbabilityModel,make_svm,calibrators_from_oof
from data import summarize
from run import participant_folds


class PipelineTests(unittest.TestCase):
    def test_gradients_spectral_and_coupled(self):
        rng=np.random.default_rng(7)
        for coupling in [0.,.3]:
            model=GraphNTF(rank=2,coupling=coupling,graph_lambda=.1)
            model.shapes_=[(4,2),(3,2),(4,2),(2,2)]+([(3,2)] if coupling else [])
            vector=np.concatenate([rng.uniform(.2,.9,size=s).ravel() for s in model.shapes_])
            x=rng.uniform(size=(4,3,4,2));y=rng.uniform(size=(4,3,3,3)) if coupling else None
            lap=np.array([[1,-1,0],[-1,2,-1],[0,-1,1.]])
            numerical=approx_derivative(lambda z:model._loss_grad(z,x,y,lap)[0],vector,method='3-point').ravel()
            exact=model._loss_grad(vector,x,y,lap)[1]
            np.testing.assert_allclose(exact,numerical,atol=1e-7,rtol=1e-5)

    def test_nonnegative_summary_and_gain_invariance(self):
        x=np.random.default_rng(8).normal(size=(6,19,24,13))
        a,d=summarize(x);b,_=summarize(x+3)
        self.assertEqual(a.shape,(19,24,4));self.assertEqual(d.shape,(912,))
        self.assertTrue((a>=0).all());np.testing.assert_allclose(a,b,atol=1e-7)

    def test_graph_and_synthetic_recovery(self):
        rng=np.random.default_rng(10)
        pos=rng.normal(size=(8,3));adj,lap=electrode_graph(pos)
        np.testing.assert_allclose(lap,lap.T);self.assertGreaterEqual(np.linalg.eigvalsh(lap).min(),-1e-12)
        self.assertEqual(np.diag(adj).sum(),0)
        factors=[rng.uniform(.2,1,(20,2)),unit_columns(rng.uniform(.1,1,(8,2))),
                 unit_columns(rng.uniform(.1,1,(6,2))),unit_columns(rng.uniform(.1,1,(4,2)))]
        x=np.einsum('nr,cr,fr,sr->ncfs',*factors)
        m=GraphNTF(2,0,maxiter=300,starts=2).fit(x[:15],lap)
        z=m.transform(x[15:]);self.assertTrue((z>=0).all())
        residual=np.linalg.norm((x[15:]/m.stat_scale_).reshape(5,-1)-z@m.basis_.T)
        self.assertLess(residual/np.linalg.norm(x[15:]/m.stat_scale_),.08)
        for factor in m.factors_:np.testing.assert_allclose(np.linalg.norm(factor,axis=0),1)
        for start in m.diagnostics_:self.assertLessEqual(start['objective'],start['history'][0])
        first=m.transform(x[15:16]);all_values=m.transform(x[15:])
        np.testing.assert_allclose(first,all_values[:1])

    def test_probability_serialization_and_unseen_site(self):
        rng=np.random.default_rng(20);n=54
        labels=np.tile(np.arange(3),18);sites=np.array(['A']*27+['B']*27)
        direct=rng.normal(size=(n,12))+labels[:,None]
        x=rng.uniform(size=(n,3,4,2))
        splits=participant_folds(labels,sites,3,11)
        scores=np.empty((n,3));cfg={'kernel':'linear','C':1.,'gamma':'scale'}
        for fit,valid in splits:
            self.assertFalse(set(fit)&set(valid))
            fm=FeatureModel({'method':'direct','pca':None},np.eye(3),20,1,1).fit(x[fit],sites[fit],direct[fit])
            clf=make_svm(cfg).fit(fm.transform(x[fit],sites[fit],direct[fit]),labels[fit])
            scores[valid]=clf.decision_function(fm.transform(x[valid],sites[valid],direct[valid]))
        fm=FeatureModel({'method':'direct','pca':None},np.eye(3),20,1,1).fit(x,sites,direct)
        clf=make_svm(cfg).fit(fm.transform(x,sites,direct),labels)
        model=ProbabilityModel(fm,clf,calibrators_from_oof(scores,labels))
        expected=model.predict_proba(x[:2],['new','new'],direct[:2])
        np.testing.assert_allclose(expected.sum(1),1);self.assertTrue((expected>0).all())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.joblib';joblib.dump(model,path)
            np.testing.assert_allclose(joblib.load(path).predict_proba(x[:2],['new','new'],direct[:2]),expected)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):unittest.main()
