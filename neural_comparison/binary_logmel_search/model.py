"""Fold-local site adjustment, scaling, reduction, SVM, and sigmoid calibration."""
from __future__ import annotations

import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.utils.extmath import svd_flip
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.svm import SVC


def fit_pca(values,components):
    """Same full PCA; use LAPACK gesvd if divide-and-conquer gesdd fails."""
    pca=PCA(n_components=components,svd_solver='full')
    try:
        return pca.fit(values)
    except np.linalg.LinAlgError:
        if not np.isfinite(values).all():raise
        mean=values.mean(0)
        u,s,v=svd(values-mean,full_matrices=False,lapack_driver='gesvd')
        u,v=svd_flip(u,v,u_based_decision=False)
        variance=s*s/(len(values)-1)
        pca.mean_=mean;pca.components_=v[:components].copy()
        pca.n_components_=components;pca.n_samples_=len(values);pca.n_features_in_=values.shape[1]
        pca.explained_variance_=variance[:components].copy()
        pca.explained_variance_ratio_=(variance/variance.sum())[:components].copy()
        pca.singular_values_=s[:components].copy()
        pca.noise_variance_=float(variance[components:].mean()) if components<len(variance) else 0.
        pca._fit_svd_solver='full';pca.fallback_lapack_driver_='gesvd'
        return pca


class BinaryModel:
    def __init__(self,config): self.config=config

    def fit(self,x,y,sites):
        c=self.config; x=np.asarray(x,float);sites=np.asarray(sites)
        self.global_mean_=x.mean(0);self.global_sd_=np.maximum(x.std(0),1e-3)
        self.site_stats_={s:(x[sites==s].mean(0),np.maximum(x[sites==s].std(0),1e-3)) for s in np.unique(sites)}
        self.scaler_=(StandardScaler() if c['scaler']=='standard' else RobustScaler()).fit(self.center(x,sites))
        z=self.scaler_.transform(self.center(x,sites))
        self.pca_=None
        if c['pca'] is not None:
            self.pca_=fit_pca(z,min(c['pca'],len(x)-1,z.shape[1]))
            z=self.pca_.transform(z)
        weights={0:c['healthy_weight'],1:1.} if isinstance(c['healthy_weight'],(int,float)) else c['healthy_weight']
        self.svm_=SVC(C=c['C'],kernel=c['kernel'],gamma=c['gamma'],class_weight=weights,probability=False,
                      cache_size=128,random_state=20261105).fit(z,y)
        return self

    def center(self,x,sites):
        if self.config['site_adjustment']=='none': return x.copy()
        stats=[self.site_stats_.get(s,(self.global_mean_,self.global_sd_)) for s in sites]
        z=x-np.stack([v[0] for v in stats])
        return z/np.stack([v[1] for v in stats]) if self.config['site_adjustment']=='zscore' else z

    def transform(self,x,sites):
        z=self.scaler_.transform(self.center(np.asarray(x,float),sites))
        return self.pca_.transform(z) if self.pca_ is not None else z

    def decision_function(self,x,sites): return self.svm_.decision_function(self.transform(x,sites))

    def predict(self,x,sites): return (self.decision_function(x,sites)>=self.config['threshold']).astype(int)

    def predict_proba(self,x,sites):
        scores=self.decision_function(x,sites)
        return self.calibrator_.predict_proba(scores[:,None])

    @property
    def probability_threshold(self):
        return float(self.calibrator_.predict_proba([[self.config['threshold']]])[0,1])
