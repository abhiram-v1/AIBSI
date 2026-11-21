import unittest
import numpy as np
from scipy.special import logsumexp
from audit_data import accepted,bank,pool,spectrum
from compat_model import Transform,covariance,symmetric_power,select


class CompatibilityTests(unittest.TestCase):
    def test_physical_signal_units_and_relative_features(self):
        t=np.arange(2048)/256
        wave=10e-6*np.sin(2*np.pi*10*t)
        windows=np.tile(wave,(3,19,1))
        log,bp,profile,m=spectrum(windows)
        self.assertEqual(log.shape,(3,19,24,13))
        self.assertEqual(pool(log).shape,(912,))
        self.assertAlmostEqual(m['median_channel_rms_uv'],10/np.sqrt(2),places=5)
        self.assertTrue(np.all(bp[:,2]>49.9))
        self.assertTrue(np.all(bp[:,[0,1,3,4]]<1e-10))
        relative=log-logsumexp(log*np.log(10),axis=2,keepdims=True)/np.log(10)
        np.testing.assert_allclose((10**relative).sum(2),1,atol=1e-12)
        np.testing.assert_allclose(bank(np.arange(1,40.01,.5)).sum(1),1)

    def test_boundary_guard_and_window_selection(self):
        rng=np.random.default_rng(3);x=rng.normal(0,3e-6,(19,256*64))
        starts,meta=accepted(x,[(15,21)])
        self.assertTrue(all(not (s/256<21 and (s+2048)/256>15) for s in starts))
        self.assertEqual(meta['boundary'],2)

    def test_covariance_alignment_math(self):
        rng=np.random.default_rng(5);a=rng.normal(size=(40,8))*np.arange(1,9)
        b=rng.normal(size=(60,8))@rng.normal(size=(8,8))
        source=covariance(a);target=covariance(b)
        transform=symmetric_power(source,-.5)@symmetric_power(target,.5)
        np.testing.assert_allclose(transform.T@source@transform,target,rtol=1e-8,atol=1e-8)

    def test_transforms_use_training_only_and_accept_unseen_site(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(48,80));sites=np.repeat(['a','b','c'],16)
        for arm in ('raw','site_z','site_coral'):
            transform=Transform(arm).fit(x,sites);before=transform.transform(x,sites)
            # A held-out extreme value must not update learned centering/PCA/covariances.
            out=transform.transform(np.ones((2,80))*1e5,np.array(['a','unseen']))
            self.assertTrue(np.isfinite(out).all())
            np.testing.assert_array_equal(transform.transform(x,sites),before)
            if arm=='site_z':
                adjusted=transform.adjust(x,sites)
                for site in np.unique(sites):np.testing.assert_allclose(adjusted[sites==site].mean(0),0,atol=1e-12)

    def test_nested_selection_returns_finite_deterministic_scores(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(36,30));y=np.tile([0,1],18)
        site=np.repeat(['a','b','c'],12);strata=np.array([f'{s}{v}' for s,v in zip(site,y)])
        a,trials=select(x,y,site,strata,'site_coral',39)
        b,_=select(x,y,site,strata,'site_coral',39)
        self.assertEqual(a,b);self.assertEqual(len(trials),24)


if __name__=='__main__':unittest.main()
