"""Inductive, training-only site harmonization and a small nested SVM search."""
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score,confusion_matrix

ARMS=('raw','common','gain','relative','site_z','site_coral')
GRID=[{'kernel':k,'C':c} for k in ('linear','rbf') for c in (.01,.1,1.,10.)]
THRESHOLDS=(-.5,0.,.5)


def metrics(y,p):
    cm=confusion_matrix(y,p,labels=[0,1]);rec=cm.diagonal()/cm.sum(1)
    return {'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y,p)),
            'accuracy':float(accuracy_score(y,p)),'healthy_recall':float(rec[0]),
            'dementia_recall':float(rec[1]),'confusion_matrix':cm.tolist()}


def symmetric_power(c,p):
    eig,v=np.linalg.eigh(c);eig=np.maximum(eig,1e-8)
    return (v*eig**p)@v.T


def covariance(z):
    c=np.atleast_2d(np.cov(z,rowvar=False));scale=max(float(np.trace(c)/len(c)),1e-6)
    return .7*c+.3*scale*np.eye(len(c))


class Transform:
    def __init__(self,arm):self.arm=arm

    def fit(self,x,sites):
        x=np.asarray(x,float);sites=np.asarray(sites);self.site_={}
        if self.arm=='site_z':
            self.site_={s:(x[sites==s].mean(0),np.maximum(x[sites==s].std(0),1e-3)) for s in np.unique(sites)}
        centered=self.adjust(x,sites);self.scaler_=StandardScaler().fit(centered)
        z=self.scaler_.transform(centered);self.pca_mean_=z.mean(0);z-=self.pca_mean_
        eig,u=np.linalg.eigh(z@z.T);order=np.argsort(eig)[::-1];eig=eig[order];u=u[:,order]
        count=min(20,int((eig>max(eig[0]*1e-12,1e-12)).sum()),len(z)-1)
        if count<1:raise ValueError('No usable feature dimensions')
        self.components_=(u[:,:count].T@z)/np.sqrt(eig[:count,None]);self.coral_={}
        if self.arm=='site_coral':
            z=z@self.components_.T;target=symmetric_power(covariance(z),.5);mean=z.mean(0)
            for s in np.unique(sites):
                local=z[sites==s]
                if len(local)<3:raise ValueError('Too few training participants for covariance alignment')
                self.coral_[s]=(local.mean(0),mean,symmetric_power(covariance(local),-.5)@target)
        return self

    def adjust(self,x,sites):
        z=np.asarray(x,float).copy()
        for s,(mean,sd) in self.site_.items():z[np.asarray(sites)==s]=(z[np.asarray(sites)==s]-mean)/sd
        return z

    def transform(self,x,sites):
        z=(self.scaler_.transform(self.adjust(x,sites))-self.pca_mean_)@self.components_.T
        for s,(mean,target,a) in self.coral_.items():
            mask=np.asarray(sites)==s;z[mask]=(z[mask]-mean)@a+target
        return z


def fit_svm(x,y,config):
    for limit in (200000,2000000,10000000):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',ConvergenceWarning)
            model=SVC(**config,gamma='scale',class_weight='balanced',cache_size=64,max_iter=limit).fit(x,y)
        if model.fit_status_==0:return model
    raise RuntimeError('SVM solver failed to converge')


def feature_key(arm):return 'gain' if arm in ('site_z','site_coral') else arm


def select(x,y,sites,strata,arm,seed):
    # Each inner fold fits every transformation, including PCA and harmonization.
    splits=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(x,strata))
    scores=np.empty((len(GRID),len(y)))
    for train,test in splits:
        transform=Transform(arm).fit(x[train],sites[train]);a=transform.transform(x[train],sites[train]);b=transform.transform(x[test],sites[test])
        for i,cfg in enumerate(GRID):scores[i,test]=fit_svm(a,y[train],cfg).decision_function(b)
    trials=[]
    for i,cfg in enumerate(GRID):
        for threshold in THRESHOLDS:
            m=metrics(y,(scores[i]>=threshold).astype(int))
            trials.append({'config':{**cfg,'threshold':threshold},'macro_f1':m['macro_f1'],'balanced_accuracy':m['balanced_accuracy'],'grid_index':i})
    # Stable deterministic tie resolution favors the earlier/simpler grid entry.
    best=max(trials,key=lambda v:(v['macro_f1'],v['balanced_accuracy'],-GRID[v['grid_index']]['C']))
    return best,trials


def train_predict(x,y,sites,xt,st,arm,config):
    transform=Transform(arm).fit(x,sites)
    svm=fit_svm(transform.transform(x,sites),y,{k:config[k] for k in ('C','kernel')})
    score=svm.decision_function(transform.transform(xt,st))
    return (score>=config['threshold']).astype(int),score
