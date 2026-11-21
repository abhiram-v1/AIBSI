import unittest
import tempfile
from pathlib import Path
import joblib
import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from brainlat_data import geometry,logmel,pool,accepted_windows
from brainlat_model import Transform,ProbabilityModel,classifier,MonotoneSigmoid
from sklearn.linear_model import LogisticRegression
from run import folds,grid


class Tests(unittest.TestCase):
    def test_weak_calibration_falls_back_to_prevalence(self):
        m=MonotoneSigmoid().fit([-2,-1,1,2],[1,1,0,0])
        self.assertTrue(m.uninformative_)
        np.testing.assert_allclose(m.predict_proba([[-3],[3]])[:,1],.5,atol=1e-7)

    def test_geometry_and_native_selection(self):
        maps,meta=geometry()
        self.assertEqual(maps['native128'].shape,(128,128));self.assertEqual(maps['aligned19'].shape,(19,128))
        self.assertEqual(np.sum(maps['native19']),19)
        self.assertEqual(np.count_nonzero(maps['native19'].sum(0)),19)
        for m in maps.values():np.testing.assert_allclose(m.sum(1),1,atol=1e-6)
        self.assertFalse(np.allclose(maps['aligned19'],maps['legacy19']))

    def test_spectrum_native_shapes_and_frequency(self):
        t=np.arange(8*512)/512
        for channels in [19,128]:
            wave=np.ones((3,channels,1))*20e-6*np.sin(2*np.pi*10*t)
            m=logmel(wave);self.assertEqual(m.shape,(3,channels,32,8))
            f=pool(m-np.median(m));self.assertEqual(f.shape,(channels,32,4));self.assertTrue(np.isfinite(f).all())

    def test_dual_pca_and_no_test_statistics(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(50,600));sites=np.array(['AR']*25+['CL']*25)
        c={'site_adjustment':'none','scaler':'standard','pca':20}
        transform=Transform(c).fit(x[:40],sites[:40]);z=transform.scaler_.transform(x[:40])
        ref=PCA(20,svd_solver='full').fit(z)
        # Projection subspaces match despite arbitrary eigenvector signs.
        np.testing.assert_allclose(transform.components_.T@transform.components_,ref.components_.T@ref.components_,atol=1e-9)
        first=transform.transform(x[40:41],['unseen'])
        np.testing.assert_allclose(first,transform.transform(np.vstack([x[40],x[41]*10000]),['unseen','unseen'])[:1])

    def test_folds_probability_and_serialization(self):
        people=[{'site':s,'label':label} for s in ['AR','CL'] for label in ['AD','FTD','HC'] for _ in range(8)]
        coverage=np.zeros(len(people),int)
        for a,b in folds({'participants':people},np.arange(len(people)),4,1):
            self.assertFalse(set(a)&set(b));coverage[b]+=1
        np.testing.assert_array_equal(coverage,1)
        rng=np.random.default_rng(3);x=rng.normal(size=(48,30));y=np.tile([0,1],24);sites=np.array(['AR']*48)
        c={**grid()[0],'C':1.,'pca':10,'threshold':0.}
        t=Transform(c).fit(x,sites);svm=classifier(c).fit(t.transform(x,sites),y)
        cal=MonotoneSigmoid().fit([-2,-1,1,2],[0,0,1,1])
        m=ProbabilityModel(c,t,svm,cal);p=m.predict_proba(x[:2],sites[:2]);np.testing.assert_allclose(p.sum(1),1)
        np.testing.assert_array_equal(p[:,1]>=m.probability_threshold,m.predict(x[:2],sites[:2]))
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'model.joblib';joblib.dump(m,file);np.testing.assert_allclose(joblib.load(file).predict_proba(x[:2],sites[:2]),p)


if __name__=='__main__':
    with threadpool_limits(limits=1):unittest.main()
