"""Train-only feature transforms; dual PCA avoids a wide-matrix SVD."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.svm import SVC


class MonotoneSigmoid:
    """Unweighted OOF sigmoid with non-negative slope.

    At the boundary, return the training prevalence rather than reversing an
    unreliable score direction. Main SVM operating-point predictions are separate.
    """
    def fit(self,scores,labels):
        x=np.asarray(scores,float).ravel();y=np.asarray(labels,float)
        def objective(v):
            a,b=v;z=a*x+b;residual=expit(z)-y
            return (float(np.mean(np.logaddexp(0,z)-y*z)+.5e-6*a*a),
                    np.array([np.mean(residual*x)+1e-6*a,np.mean(residual)]))
        prior=np.clip(y.mean(),1e-8,1-1e-8)
        result=minimize(objective,[0.,np.log(prior/(1-prior))],jac=True,method='L-BFGS-B',
                        bounds=[(0,None),(None,None)],options={'ftol':1e-12,'gtol':1e-8,'maxiter':1000})
        if not result.success:raise RuntimeError('Monotone calibration did not converge')
        self.slope_,self.intercept_=result.x;self.uninformative_=bool(self.slope_<1e-8)
        return self

    def predict_proba(self,scores):
        p=expit(self.slope_*np.asarray(scores,float).ravel()+self.intercept_)
        return np.column_stack([1-p,p])


class Transform:
    def __init__(self,config):self.config=config

    def fit(self,x,sites):
        x=np.asarray(x,float);sites=np.asarray(sites)
        self.mean_=x.mean(0);self.sd_=np.maximum(x.std(0),1e-3)
        self.site_={s:(x[sites==s].mean(0),np.maximum(x[sites==s].std(0),1e-3)) for s in np.unique(sites)}
        centered=self.center(x,sites)
        self.scaler_=(StandardScaler() if self.config['scaler']=='standard' else RobustScaler()).fit(centered)
        z=self.scaler_.transform(centered);self.components_=None
        if self.config['pca'] is not None:
            self.pca_mean_=z.mean(0);z=z-self.pca_mean_
            values,u=np.linalg.eigh(z@z.T)
            order=np.argsort(values)[::-1];values=values[order];u=u[:,order]
            valid=values>max(values[0]*1e-12,1e-12)
            count=min(self.config['pca'],int(valid.sum()),len(z)-1)
            if count<1:raise ValueError('No usable PCA dimensions')
            self.components_=(u[:,:count].T@z)/np.sqrt(values[:count,None])
        return self

    def center(self,x,sites):
        if self.config['site_adjustment']=='none':return x.copy()
        stats=[self.site_.get(s,(self.mean_,self.sd_)) for s in sites]
        return (x-np.stack([v[0] for v in stats]))/np.stack([v[1] for v in stats])

    def transform(self,x,sites):
        z=self.scaler_.transform(self.center(np.asarray(x,float),sites))
        return (z-self.pca_mean_)@self.components_.T if self.components_ is not None else z


def classifier(config,max_iter=200000):
    return SVC(C=config['C'],kernel=config['kernel'],gamma='scale',class_weight=config['weight'],
               probability=False,max_iter=max_iter,cache_size=64,random_state=20261106)


class ProbabilityModel:
    def __init__(self,config,transform,svm,calibrator):
        self.config=config;self.transform_=transform;self.svm_=svm;self.calibrator_=calibrator

    def decision_function(self,x,sites):return self.svm_.decision_function(self.transform_.transform(x,sites))

    def predict_proba(self,x,sites):return self.calibrator_.predict_proba(self.decision_function(x,sites)[:,None])

    def predict(self,x,sites):return (self.decision_function(x,sites)>=self.config['threshold']).astype(int)

    @property
    def probability_threshold(self):
        if self.calibrator_.uninformative_:return None
        return float(self.calibrator_.predict_proba([[self.config['threshold']]])[0,1])
