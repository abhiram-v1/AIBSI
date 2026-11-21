import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from features import segment_windows,spectral_power,pool_power
from model import BinaryModel,fit_pca
from run import candidate_plan,folds,selection_score


class Tests(unittest.TestCase):
    def test_pca_fallback(self):
        from sklearn.decomposition import PCA
        values=np.random.default_rng(4).normal(size=(30,45))
        expected=PCA(10,svd_solver='full').fit(values)
        with patch.object(PCA,'fit',side_effect=np.linalg.LinAlgError('forced gesdd failure')):
            result=fit_pca(values,10)
        np.testing.assert_allclose(result.explained_variance_,expected.explained_variance_,atol=1e-10)
        np.testing.assert_allclose(result.transform(values),expected.transform(values),atol=1e-10)

    def test_time_frequency_and_rate_consistency(self):
        outputs=[]
        for fs in [128,256]:
            time=np.arange(8*fs)/fs
            wave=(20e-6*np.sin(2*np.pi*10*time))[None,None,:]*np.ones((2,19,1))
            cfg={'sampling_hz':fs,'stft_seconds':2,'hop_seconds':.5}
            freq,power=spectral_power(wave,cfg)
            self.assertEqual(power.shape,(2,19,79,13))
            self.assertEqual(freq[power[0,0].mean(1).argmax()],10)
            outputs.append(pool_power(freq,power,24))
        np.testing.assert_allclose(outputs[0][...,:3],outputs[1][...,:3],atol=.3)

    def test_boundary_rejection_and_duration_cap(self):
        rng=np.random.default_rng(9);data=rng.normal(0,20e-6,(19,400*256)).astype(np.float32)
        windows,audit=segment_windows(data,8,[(20,30)],256)
        self.assertLessEqual(audit['seconds'],320);self.assertGreater(audit['boundary'],0)
        self.assertEqual(windows.shape[1:],(19,2048))

    def test_candidate_coverage_and_group_folds(self):
        candidates=candidate_plan(256)
        self.assertEqual({c['sampling_hz'] for c in candidates},{128,256})
        self.assertEqual({c['segment_seconds'] for c in candidates},{2,4,6,8})
        for c in candidates:self.assertGreaterEqual(c['segment_seconds'],c['stft_seconds']+2*c['hop_seconds'])
        plan={'participants':[{'site':s,'label':v} for s in ['A','B'] for v in ['AD','FTD','HC'] for _ in range(8)]}
        coverage=np.zeros(len(plan['participants']),int)
        for a,b in folds(plan,np.arange(len(coverage)),4,1):
            self.assertFalse(set(a)&set(b));coverage[b]+=1
        np.testing.assert_array_equal(coverage,1)

    def test_train_only_transform_and_serialization(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(60,30));y=np.tile([0,1],30);sites=np.array(['A']*30+['B']*30)
        cfg={**candidate_plan(256)[0],'pca':None,'threshold':.5}
        model=BinaryModel(cfg).fit(x[:50],y[:50],sites[:50])
        model.calibrator_=LogisticRegression().fit(np.array([-2,-1,1,2])[:,None],[0,0,1,1])
        expected=model.transform(x[50:51],['unseen'])
        np.testing.assert_allclose(expected,model.transform(np.vstack([x[50],x[51]*10000]),['unseen','unseen'])[:1])
        p=model.predict_proba(x[50:],[*sites[50:]])
        np.testing.assert_allclose(p.sum(1),1)
        np.testing.assert_array_equal(p[:,1]>=model.probability_threshold,model.predict(x[50:],sites[50:]))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.joblib';joblib.dump(model,path)
            np.testing.assert_allclose(joblib.load(path).predict_proba(x[50:],sites[50:]),p)


if __name__=='__main__':
    with threadpool_limits(limits=1):unittest.main()
