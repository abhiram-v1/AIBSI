"""Frozen single-amplitude SVM and source-only probability calibration."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from compat_model import Transform,fit_svm


class MonotoneSigmoid:
    # Same nonnegative-slope formulation used in the earlier native-montage audit.
    def fit(self,scores,y):
        x=np.asarray(scores,float).ravel();y=np.asarray(y,float)
        prior=np.clip(y.mean(),1e-8,1-1e-8)
        def objective(v):
            a,b=v;z=a*x+b;r=expit(z)-y
            return float(np.mean(np.logaddexp(0,z)-y*z)+.5e-6*a*a),np.array([np.mean(r*x)+1e-6*a,np.mean(r)])
        result=minimize(objective,[0,np.log(prior/(1-prior))],jac=True,method='L-BFGS-B',bounds=[(0,None),(None,None)],
                        options={'ftol':1e-12,'gtol':1e-8,'maxiter':1000})
        if not result.success:raise RuntimeError('Source probability calibration failed')
        self.slope_,self.intercept_=result.x;self.uninformative_=bool(self.slope_<1e-8)
        return self

    def predict_proba(self,scores):
        p=expit(self.slope_*np.asarray(scores,float).ravel()+self.intercept_)
        return np.column_stack([1-p,p])


class AmplitudeModel:
    def __init__(self,config):self.config=config

    def fit(self,x,y,sites,source_oof_scores):
        self.transform_=Transform('raw').fit(x,sites)
        self.svm_=fit_svm(self.transform_.transform(x,sites),y,{k:self.config[k] for k in ('C','kernel')})
        self.calibrator_=MonotoneSigmoid().fit(source_oof_scores,y)
        return self

    def decision_function(self,x):
        # No site correction: the source-trained scaler/PCA is fixed for all people.
        return self.svm_.decision_function(self.transform_.transform(np.asarray(x,float),np.repeat('unseen',len(x))))

    def predict(self,x):return (self.decision_function(x)>=self.config['threshold']).astype(int)

    def predict_proba(self,x):return self.calibrator_.predict_proba(self.decision_function(x))

    @property
    def probability_threshold(self):
        return None if self.calibrator_.uninformative_ else float(self.calibrator_.predict_proba([self.config['threshold']])[0,1])
