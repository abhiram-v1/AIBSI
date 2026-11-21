import unittest
import numpy as np
from scipy.signal import welch
from spectral_data import descriptors,DESCRIPTORS,RELATIVE_INDICES,feature_names
from fourier_model import FeatureTransform,metrics,select
from analyze import fdr,correlate


class FourierTests(unittest.TestCase):
    def test_fourier_recovers_known_frequency_and_power(self):
        t=np.arange(2048)/256;x=10*np.sin(2*np.pi*10*t)
        f,p=welch(x,fs=256,window='hann',nperseg=1024,noverlap=512,nfft=1024)
        ix=(f>=1)&(f<=40);v=descriptors(p[ix],f[ix])
        self.assertAlmostEqual(10**v[DESCRIPTORS.index('log_total_power')],50,places=8)
        self.assertEqual(v[DESCRIPTORS.index('alpha_peak_7_13_hz')],10)
        self.assertAlmostEqual(v[DESCRIPTORS.index('relative_alpha_8_13')],1,places=10)

    def test_relative_features_ignore_pure_gain(self):
        f=np.arange(1,40.01,.25);p=2/f**1.5+4*np.exp(-((f-9.5)/1.5)**2)
        a=descriptors(p,f);b=descriptors(100*p,f)
        np.testing.assert_allclose(a[RELATIVE_INDICES],b[RELATIVE_INDICES],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(b[:5]-a[:5],2,atol=1e-12)
        nn=feature_names(f);self.assertEqual(len(nn['absolute_spectrum']),19*157)

    def test_feature_ranking_is_training_only(self):
        rng=np.random.default_rng(11);x=rng.normal(size=(40,30));y=np.tile([0,1],20);x[:,3]+=10*y
        transform=FeatureTransform().fit(x,y,5);self.assertIn(3,transform.selected_)
        before=transform.transform(x);transform.transform(np.ones((2,30))*1e8)
        np.testing.assert_array_equal(before,transform.transform(x))

    def test_metrics_and_multiple_testing(self):
        m=metrics(np.array([0,0,1,1]),np.array([0,1,1,1]),2)
        self.assertAlmostEqual(m['macro_f1'],(2/3+4/5)/2)
        np.testing.assert_allclose(fdr([.01,.04,.03,1]),[.04,.053333333333,.053333333333,1])
        np.testing.assert_allclose(correlate(np.array([[0,1],[1,1],[2,1]],float),np.array([0,1,2])),[1,0])

    def test_multiclass_inner_search(self):
        rng=np.random.default_rng(7);y=np.tile(np.arange(3),12);x=rng.normal(size=(36,10));x[:,0]+=3*y
        best,trials=select(x,y,y,3,91)
        self.assertIsNone(best['config']['threshold']);self.assertTrue(0<=best['macro_f1']<=1)
        self.assertEqual(len(trials),16)


if __name__=='__main__':unittest.main()
