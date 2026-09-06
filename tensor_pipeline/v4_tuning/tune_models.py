"""Bounded classifier tuning and strongly regularized supervised TT."""
from pathlib import Path
import sys
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

V3=Path(__file__).resolve().parents[1]/'v3_waveform'
sys.path.insert(0,str(V3))
from wave_models import SupervisedTT,DelayProjection,weights


class RegularizedTT(SupervisedTT):
    def __init__(self,penalty=1.,anchor=.1,channel_rank=2,filters=4,maxiter=500):
        super().__init__(penalty,maxiter,channel_rank,filters)
        self.anchor=anchor

    def objective(self,theta,x,y,w):
        loss,g=super().objective(theta,x,y,w)
        u,v,h,b=self.unpack(theta)
        gu,gv,gh,gb=self.unpack(g)
        extra=self.anchor-.001
        loss+=.5*extra*(np.sum((u-self.u0_)**2)+np.sum((v-self.v0_)**2))
        gu=gu+extra*(u-self.u0_); gv=gv+extra*(v-self.v0_)
        return float(loss),self.pack(gu,gv,gh,gb)


def feature_bank(x,controls,y,train):
    models={}; features={'spectral':controls}
    for name,p in [('tt8',DelayProjection('tt',rank=8)),('tt16',DelayProjection('tt',rank=16)),
                   ('tucker4',DelayProjection('tucker',channel_rank=4,lag_rank=4)),
                   ('tucker6',DelayProjection('tucker',channel_rank=6,lag_rank=6))]:
        models[name]=p.fit(x[train],y[train]); features[name]=p.transform(x)
    features['fusion']=np.concatenate([features['tt16'],controls],axis=1)
    return features,models


def classifier_specs():
    specs=[dict(kind='logistic',C=c) for c in [.01,.1,1.]]
    specs += [dict(kind='svm',C=c,gamma=g) for c in [.1,1.,10.] for g in ['scale',.01]]
    # Eight prespecified regularized tree settings; no search expansion after scores.
    tuples=[(1,.03,250,1,10,.1,1.),(1,.1,100,3,10,.1,.8),
            (2,.03,250,1,1,0.,1.),(2,.1,100,3,10,.1,.8),
            (2,.03,150,5,10,.5,1.),(3,.03,250,3,10,.5,.8),
            (2,.1,100,1,1,.1,.8),(3,.1,100,5,10,.5,1.)]
    for depth,rate,n,child,reg,alpha,sub in tuples:
        specs.append(dict(kind='xgboost',max_depth=depth,learning_rate=rate,n_estimators=n,
                          min_child_weight=child,reg_lambda=reg,reg_alpha=alpha,subsample=sub,colsample_bytree=.8))
    specs += [dict(kind='random_forest',max_depth=3,min_samples_leaf=3),
              dict(kind='random_forest',max_depth=None,min_samples_leaf=5)]
    return specs


def configs():
    reps=[('tt8',None),('tt16',None),('tucker4',None),('tucker6',None),
          ('spectral',16),('spectral',32),('fusion',32)]
    result=[dict(representation=rep,selection=k,**c) for rep,k in reps for c in classifier_specs()]
    for rank,filters,anchor,penalty in [(2,4,.1,.1),(2,4,.1,1.),(4,8,.1,.1),(4,8,.1,1.),(2,4,1.,1.),(4,8,1.,1.)]:
        result.append(dict(kind='supervised_tt',representation='supervised_tt',selection=None,
                           channel_rank=rank,filters=filters,anchor=anchor,penalty=penalty,maxiter=500))
    return result


def fit_classifier(cfg,z,y):
    kind=cfg['kind']
    if kind=='logistic': m=LogisticRegression(C=cfg['C'],max_iter=5000,tol=1e-5)
    elif kind=='svm': m=SVC(C=cfg['C'],kernel='rbf',gamma=cfg['gamma'],probability=False)
    elif kind=='random_forest':
        m=RandomForestClassifier(n_estimators=200,max_depth=cfg['max_depth'],min_samples_leaf=cfg['min_samples_leaf'],
                                  max_features='sqrt',random_state=9064,n_jobs=1)
    else:
        params={k:v for k,v in cfg.items() if k not in ['kind','representation','selection']}
        m=XGBClassifier(**params,tree_method='hist',device='cpu',objective='multi:softprob',num_class=3,
                        eval_metric='mlogloss',random_state=9064,n_jobs=1,verbosity=0)
    steps=[]
    if cfg['selection'] is not None: steps.append(('select',SelectKBest(f_classif,k=cfg['selection'])))
    steps += [('scale',StandardScaler()),('clf',m)]
    pipe=Pipeline(steps); w=weights(y)
    pipe.fit(z,y,scale__sample_weight=w,clf__sample_weight=w)
    return pipe


def fit_supervised(cfg,x,y):
    p={k:cfg[k] for k in ['penalty','anchor','channel_rank','filters','maxiter']}
    return RegularizedTT(**p).fit(x,y)


def scores(model,z):
    if hasattr(model,'decision_function'): return model.decision_function(z),'decision_function'
    return model.predict_proba(z),'probabilities'


def group_rep(cfg):
    r=cfg['representation']
    if r.startswith('tt'): return 'wave_tt'
    if r.startswith('tucker'): return 'wave_tucker'
    return r
