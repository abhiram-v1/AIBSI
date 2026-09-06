"""Inductive Tensor Train and Tucker projections in a common learned basis.

TT-SVD uses feature modes first and observations last. Left-orthogonal cores
define a reusable orthonormal feature basis, avoiding unrelated per-subject TT
core gauges. A covariance eigensolve yields the same leading left singular
vectors as SVD (up to signs/degenerate rotations).
"""
import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier

def left_vectors(a, rank):
    rank=min(rank,*a.shape)
    cov=a@a.T
    values,vec=eigh(cov,subset_by_index=(len(cov)-rank,len(cov)-1),check_finite=False)
    vec=vec[:,::-1]
    # Fix arbitrary signs for reproducible exported cores.
    signs=np.sign(vec[np.abs(vec).argmax(axis=0),np.arange(vec.shape[1])])
    return vec*np.where(signs==0,1,signs)

class TTProjection:
    def __init__(self, rank=32, channel_rank=8, spectral_rank=24):
        self.rank=rank
        self.channel_rank=channel_rank
        self.spectral_rank=spectral_rank

    def fit(self, x):
        x=np.asarray(x,dtype=np.float64)
        self.mean_=x.mean(0)
        dims=x.shape[1:]
        z=np.moveaxis(x-self.mean_,0,-1)
        cores=[]
        last=1
        for d,r in zip(dims,[self.channel_rank,self.spectral_rank,self.rank]):
            mat=z.reshape(last*d,-1)
            u=left_vectors(mat,r)
            cores.append(u.reshape(last,d,u.shape[1]))
            z=u.T@mat
            last=u.shape[1]
        self.cores_=cores
        self.basis_=np.einsum('aib,bjc,ckd->ijkd',*cores,optimize=True).reshape(np.prod(dims),last)
        self.orthogonality_error_=float(np.max(np.abs(self.basis_.T@self.basis_-np.eye(last))))
        zflat=(x-self.mean_).reshape(len(x),-1)
        self.relative_centered_reconstruction_error_=float(1-np.sum((zflat@self.basis_)**2)/max(np.sum(zflat*zflat),1e-12))
        return self

    def transform(self,x):
        return (np.asarray(x)-self.mean_).reshape(len(x),-1)@self.basis_

class TuckerProjection:
    """Training-only multilinear PCA via HOSVD, signed log-power compatible."""
    def __init__(self,ranks=(6,8,3)):
        self.ranks=ranks

    def fit(self,x):
        x=np.asarray(x,dtype=float)
        self.mean_=x.mean(0)
        z=x-self.mean_
        self.factors_=[left_vectors(np.moveaxis(z,i+1,0).reshape(z.shape[i+1],-1),rank)
                       for i,rank in enumerate(self.ranks)]
        self.basis_=np.einsum('ia,jb,kc->ijkabc',*self.factors_,optimize=True).reshape(np.prod(x.shape[1:]),-1)
        self.orthogonality_error_=float(np.max(np.abs(self.basis_.T@self.basis_-np.eye(self.basis_.shape[1]))))
        return self

    def transform(self,x):
        return (np.asarray(x)-self.mean_).reshape(len(x),-1)@self.basis_

def anatomical_smoothing(coords, strength=.5):
    distance=np.linalg.norm(coords[:,None]-coords[None,:],axis=-1)
    np.fill_diagonal(distance,np.inf)
    neighbors=np.argsort(distance,axis=1)[:,:4]
    a=np.zeros_like(distance)
    scale=np.median(np.take_along_axis(distance,neighbors,axis=1))
    for i,ns in enumerate(neighbors):
        a[i,ns]=np.exp(-distance[i,ns]**2/(2*scale**2))
    a=np.maximum(a,a.T)
    di=np.diag(1/np.sqrt(a.sum(1)))
    laplacian=np.eye(len(a))-di@a@di
    vals,vec=eigh(laplacian)
    smoother=(vec*(1+strength*np.maximum(vals,0))**-.5)@vec.T
    return smoother, a

def apply_smoothing(x,s):
    return np.einsum('cd,ndft->ncft',s,x,optimize=True)

def classifier_grid():
    return ([dict(kind='logistic',C=c) for c in (.1,1.,10.)]
            +[dict(kind='svm_linear',C=c) for c in (.1,1.)]
            +[dict(kind='svm_rbf',C=c) for c in (1.,10.)]
            +[dict(kind='hist_boost',C=1.)])

def build_classifier(cfg, nfeatures, selection=None):
    if cfg['kind']=='logistic':
        model=LogisticRegression(C=cfg['C'],max_iter=4000,class_weight='balanced',tol=1e-5)
    elif cfg['kind'].startswith('svm_'):
        model=SVC(C=cfg['C'],kernel=cfg['kind'][4:],class_weight='balanced',probability=False,
                  gamma='scale',decision_function_shape='ovr')
    else:
        model=HistGradientBoostingClassifier(max_iter=100,max_depth=2,min_samples_leaf=5,
                    learning_rate=.05,l2_regularization=2.,class_weight='balanced',early_stopping=False,random_state=5041)
    steps=[]
    if selection is not None:
        steps.append(('selection',SelectKBest(f_classif,k=min(selection,nfeatures))))
    steps.extend([('scaler',StandardScaler()),('classifier',model)])
    return Pipeline(steps)

def predict_with_scores(model,x):
    pred=model.predict(x)
    if hasattr(model,'decision_function'):
        scores=model.decision_function(x)
        score_type='decision_function_not_probability'
    else:
        scores=model.predict_proba(x)
        score_type='predict_proba'
    return pred,scores,score_type
