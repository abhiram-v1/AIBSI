"""Feature ranking, normalization and SVM selection fitted within each training fold."""
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
from spectral_data import AUDIT
from compat_model import fit_svm

ARMS=('amplitude','summary','relative_summary','absolute_spectrum','relative_spectrum')
PARAMS=[{'kernel':k,'C':c} for k in ('linear','rbf') for c in (.01,.1,1.,10.)]
COUNTS=(5,15,40,'all')
THRESHOLDS=(-.5,0.,.5)


def metrics(y,p,classes):
    cm=np.zeros((classes,classes),int);np.add.at(cm,(y,p),1)
    recall=np.divide(cm.diagonal(),cm.sum(1),out=np.zeros(classes),where=cm.sum(1)>0)
    f1=np.divide(2*cm.diagonal(),cm.sum(1)+cm.sum(0),out=np.zeros(classes),where=(cm.sum(1)+cm.sum(0))>0)
    return {'macro_f1':float(f1.mean()),'balanced_accuracy':float(recall.mean()),'accuracy':float(cm.trace()/cm.sum()),
            'per_class_recall':recall.tolist(),'confusion_matrix':cm.tolist()}


class FeatureTransform:
    def fit(self,x,y,k):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);warnings.simplefilter('ignore',UserWarning)
            scores,_=f_classif(x,y)
        scores=np.nan_to_num(scores,nan=-np.inf,posinf=np.finfo(float).max,neginf=-np.inf)
        count=x.shape[1] if k=='all' else min(k,x.shape[1])
        order=np.argsort(-scores,kind='stable');self.selected_=order[:count]
        self.scaler_=StandardScaler().fit(x[:,self.selected_]);return self

    def transform(self,x):return self.scaler_.transform(x[:,self.selected_])


def counts(x):return list(dict.fromkeys(x.shape[1] if k=='all' else min(k,x.shape[1]) for k in COUNTS))


def select(x,y,strata,classes,seed):
    inner=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(y,strata))
    choices={};runs=[]
    for k in counts(x):
        oof=np.empty((len(PARAMS),len(y)),float if classes==2 else int)
        for train,test in inner:
            transform=FeatureTransform().fit(x[train],y[train],k);a=transform.transform(x[train]);b=transform.transform(x[test])
            for i,cfg in enumerate(PARAMS):
                svm=fit_svm(a,y[train],cfg)
                oof[i,test]=svm.decision_function(b) if classes==2 else svm.predict(b)
        for i,cfg in enumerate(PARAMS):
            for threshold in THRESHOLDS if classes==2 else (None,):
                pred=(oof[i]>=threshold).astype(int) if classes==2 else oof[i].astype(int)
                m=metrics(y,pred,classes);runs.append({'config':{**cfg,'k':k,'threshold':threshold},
                    'macro_f1':m['macro_f1'],'balanced_accuracy':m['balanced_accuracy']})
    best=max(runs,key=lambda v:(v['macro_f1'],v['balanced_accuracy'],-v['config']['k'],-v['config']['C']))
    return best,runs


def train_predict(x,y,xt,config,classes):
    t=FeatureTransform().fit(x,y,config['k']);svm=fit_svm(t.transform(x),y,{k:config[k] for k in ('C','kernel')})
    if classes==2:
        score=svm.decision_function(t.transform(xt));pred=(score>=config['threshold']).astype(int)
    else:score=svm.decision_function(t.transform(xt));pred=svm.predict(t.transform(xt))
    return pred,score,t.selected_.tolist()
