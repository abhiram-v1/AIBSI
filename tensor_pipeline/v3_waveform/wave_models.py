"""Raw delay-tensor energy filters, including supervised TT optimization.

An n x (channel*lag) x (channel*lag) moment is an exact computational shortcut
for pooling squared responses of linear filters over raw delay tensors.
It is not a PSD and not a freely learned whole-recording sequence model.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

C, L = 19, 16


def weights(y):
    count = np.bincount(y, minlength=3)
    assert np.all(count > 0)
    return len(y)/(3*count[y])


def top(a, rank):
    _, u = eigh(a, subset_by_index=(len(a)-rank, len(a)-1), check_finite=False)
    u = u[:, ::-1]
    s = np.sign(u[np.abs(u).argmax(0), np.arange(rank)])
    return u*np.where(s == 0, 1, s)


def balanced_moment(x, y):
    w = weights(y)
    return np.einsum('n,nij->ij', w/w.sum(), x, optimize=True)


def energies(x, b):
    sb = (x.reshape(-1, x.shape[-1])@b).reshape(len(x), x.shape[1], -1)
    return np.einsum('ik,nik->nk', b, sb, optimize=True)/np.sum(b*b, axis=0)


class DelayProjection:
    def __init__(self, kind='tt', rank=16, channel_rank=6, lag_rank=6):
        self.kind, self.rank, self.channel_rank, self.lag_rank = kind, rank, channel_rank, lag_rank

    def fit(self, x, y):
        a = balanced_moment(x, y)
        tensor = a.reshape(C,L,C,L)
        u = top(np.einsum('cldl->cd', tensor), self.channel_rank)
        if self.kind == 'tt':
            left = np.kron(u, np.eye(L))
            v = top(left.T@a@left, self.rank)
            self.basis_ = left@v
            self.cores_ = [u.reshape(1,C,self.channel_rank), v.reshape(self.channel_rank,L,self.rank)]
        else:
            v = top(np.einsum('clcm->lm', tensor), self.lag_rank)
            self.basis_ = np.kron(u,v)
            self.factors_ = [u,v]
        self.training_moment_ = a
        return self

    def transform(self, x):
        return np.log(np.maximum(energies(x,self.basis_),1e-8))


class SupervisedTT:
    """Two input-mode TT cores + log-energy pooling + softmax classification.

    U[channel,rank] and V[rank,lag,filter] are optimized jointly with the
    classification head using weighted diagnosis loss. Parameters are compact;
    moments let all approved waveform positions/windows enter the loss exactly.
    """
    def __init__(self, penalty=.1, maxiter=100, channel_rank=4, filters=8):
        self.penalty, self.maxiter = penalty, maxiter
        self.channel_rank, self.filters = channel_rank, filters

    def pack(self,u,v,h,b):
        return np.concatenate([z.ravel() for z in [u,v,h,b]])

    def unpack(self,theta):
        r,k = self.channel_rank,self.filters
        i,j,m = C*r, C*r+r*L*k, C*r+r*L*k+k*3
        return theta[:i].reshape(C,r),theta[i:j].reshape(r,L,k),theta[j:m].reshape(k,3),theta[m:]

    def objective(self,theta,x,y,w):
        u,v,h,bias = self.unpack(theta)
        b = np.einsum('cr,rlk->clk',u,v,optimize=True).reshape(C*L,self.filters)
        norm = np.sum(b*b,axis=0)+1e-20
        sb = (x.reshape(-1,C*L)@b).reshape(len(x),C*L,self.filters)
        e0 = np.einsum('dk,ndk->nk',b,sb,optimize=True)/norm
        e = e0+1e-8
        z = (np.log(e)-self.center_)/self.scale_
        logits = z@h+bias
        loss = np.sum(w*(logsumexp(logits,axis=1)-logits[np.arange(len(y)),y]))
        anch = 1e-3
        loss += .5*self.penalty*np.sum(h*h)+.5*anch*(np.sum((u-self.u0_)**2)+np.sum((v-self.v0_)**2))
        delta = softmax(logits,axis=1)
        delta[np.arange(len(y)),y] -= 1
        delta *= w[:,None]
        gh = z.T@delta+self.penalty*h
        gbias = delta.sum(0)
        de = (delta@h.T)/self.scale_/e
        gb = 2*(np.einsum('nk,ndk->dk',de/norm,sb,optimize=True)-b*np.sum(de*e0/norm,axis=0))
        gb = gb.reshape(C,L,self.filters)
        gu = np.einsum('clk,rlk->cr',gb,v,optimize=True)+anch*(u-self.u0_)
        gv = np.einsum('cr,clk->rlk',u,gb,optimize=True)+anch*(v-self.v0_)
        return float(loss),self.pack(gu,gv,gh,gbias)

    def fit(self,x,y):
        p = DelayProjection('tt',self.filters,self.channel_rank).fit(x,y)
        self.u0_ = p.cores_[0][0].copy()
        self.v0_ = p.cores_[1].copy()
        z = p.transform(x)
        w = weights(y); w /= w.sum()
        self.center_ = np.sum(w[:,None]*z,axis=0)
        self.scale_ = np.maximum(np.sqrt(np.sum(w[:,None]*(z-self.center_)**2,axis=0)),.1)
        theta = self.pack(self.u0_,self.v0_,np.zeros((self.filters,3)),np.zeros(3))
        start_loss = self.objective(theta,x,y,w)[0]
        result = minimize(self.objective,theta,args=(x,y,w),jac=True,method='L-BFGS-B',
                          options=dict(maxiter=self.maxiter,maxls=20,ftol=1e-7,gtol=1e-5))
        assert np.isfinite(result.fun) and result.fun <= start_loss+1e-8
        self.theta_ = result.x
        self.optimization_ = dict(success=bool(result.success),message=str(result.message),
                                  iterations=int(result.nit),evaluations=int(result.nfev),
                                  initial_loss=start_loss,final_loss=float(result.fun),
                                  gradient_inf_norm=float(np.max(np.abs(result.jac))))
        return self

    def decision_function(self,x):
        u,v,h,bias = self.unpack(self.theta_)
        b = np.einsum('cr,rlk->clk',u,v,optimize=True).reshape(C*L,self.filters)
        z = (np.log(energies(x,b)+1e-8)-self.center_)/self.scale_
        return z@h+bias

    def predict(self,x):
        return self.decision_function(x).argmax(1)


class WavePipeline:
    def __init__(self, cfg):
        self.cfg = cfg.copy()

    def fit(self,x,controls,y):
        cfg = self.cfg
        self.fit_labels_ = np.array(y)
        if cfg['family'] == 'supervised_tt':
            self.model_ = SupervisedTT(penalty=cfg['penalty']).fit(x,y)
            return self
        if cfg['family'] == 'spectral_control':
            z = controls
        else:
            if cfg['family'] == 'wave_tt':
                self.projector_ = DelayProjection('tt', rank=cfg['rank']).fit(x,y)
            else:
                self.projector_ = DelayProjection('tucker',channel_rank=cfg['rank'],lag_rank=cfg['rank']).fit(x,y)
            z = self.projector_.transform(x)
        steps = []
        if cfg['family'] == 'spectral_control': steps.append(('select',SelectKBest(f_classif,k=cfg['rank'])))
        steps.append(('scale',StandardScaler()))
        model = (LogisticRegression(C=cfg['C'],max_iter=4000,tol=1e-5)
                 if cfg['kind']=='logistic' else SVC(C=cfg['C'],kernel='rbf',gamma='scale',probability=False))
        steps.append(('clf',model))
        self.model_ = Pipeline(steps)
        w = weights(y)
        self.model_.fit(z,y,scale__sample_weight=w,clf__sample_weight=w)
        return self

    def matrix(self,x,controls):
        return controls if self.cfg['family']=='spectral_control' else self.projector_.transform(x)

    def predict(self,x,controls):
        return self.model_.predict(x if self.cfg['family']=='supervised_tt' else self.matrix(x,controls))

    def scores(self,x,controls):
        return self.model_.decision_function(x if self.cfg['family']=='supervised_tt' else self.matrix(x,controls))


def candidates():
    specs = []
    for family,ranks in [('wave_tt',[8,16]),('wave_tucker',[4,6]),('spectral_control',[16,32])]:
        for r in ranks:
            for kind,c in [('logistic',.1),('logistic',1.),('svm_rbf',1.),('svm_rbf',10.)]:
                specs.append(dict(family=family,rank=r,kind=kind,C=c))
    specs.extend([dict(family='supervised_tt',penalty=p) for p in [.01,.1]])
    return specs
