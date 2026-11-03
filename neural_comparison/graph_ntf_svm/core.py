"""Inductive non-negative CP with an explicitly scale-constrained graph objective."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, nnls
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def unit_columns(v):
    return v / np.maximum(np.linalg.norm(v, axis=0), 1e-15)


def electrode_graph(positions, neighbours=4):
    p = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    distances = np.arccos(np.clip(p @ p.T, -1, 1))
    np.fill_diagonal(distances, np.inf)
    adjacency = np.zeros_like(distances)
    closest = np.argsort(distances, axis=1)[:, :neighbours]
    width = np.median(np.take_along_axis(distances, closest, axis=1))
    for i in range(len(p)):
        adjacency[i, closest[i]] = np.exp(-distances[i, closest[i]] ** 2 / (2 * width**2))
    adjacency = np.maximum(adjacency, adjacency.T)
    laplacian = np.diag(adjacency.sum(1)) - adjacency
    laplacian /= np.linalg.eigvalsh(laplacian).max()
    return adjacency, laplacian


class GraphNTF:
    """X[n,c,f,s] = sum_r A[n,r] B[c,r] F[f,r] S[s,r].

    Optional Y[n,c,d,b] shares A and BOTH electrode factors B. Y's diagonal
    is masked, since self-connectivity is undefined. All factors are >=0;
    B,F,S,H columns have unit L2 norm (eliminates graph scaling degeneracy).
    Loss = .5*||X-Xhat||^2/||X||^2 + graph_lambda/(2R)*tr(B.T L B)
           + coupling*.5*||mask*(Y-Yhat)||^2/||mask*Y||^2.
    Positive training-only statistic scaling is saved and applied at inference.
    L-BFGS-B optimizes raw positive shape factors; objective and exact gradient
    both include their column normalization. No post-update penalty mismatch.
    """
    def __init__(self, rank=5, graph_lambda=.01, coupling=0., maxiter=250, starts=3, seed=802):
        self.rank, self.graph_lambda, self.coupling = rank, graph_lambda, coupling
        self.maxiter, self.starts, self.seed = maxiter, starts, seed

    def _unpack(self, vector):
        arrays, cursor = [], 0
        for shape in self.shapes_:
            size = int(np.prod(shape))
            arrays.append(vector[cursor:cursor+size].reshape(shape))
            cursor += size
        return arrays

    def _loss_grad(self, vector, x, y, laplacian):
        raw = self._unpack(vector)
        a, b, f, s = [raw[0]] + [unit_columns(v) for v in raw[1:4]]
        basis = np.einsum('cr,fr,sr->cfsr', b, f, s, optimize=True)
        residual = (a @ basis.reshape(-1, self.rank).T).reshape(x.shape) - x
        energy = max(float(np.sum(x*x)), 1e-15)
        loss = .5 * np.sum(residual**2) / energy
        ga = residual.reshape(len(x), -1) @ basis.reshape(-1, self.rank) / energy
        gb = np.einsum('ncfs,nr,fr,sr->cr', residual, a, f, s, optimize=True) / energy
        gf = np.einsum('ncfs,nr,cr,sr->fr', residual, a, b, s, optimize=True) / energy
        gs = np.einsum('ncfs,nr,cr,fr->sr', residual, a, b, f, optimize=True) / energy
        loss += self.graph_lambda / (2*self.rank) * np.sum(b * (laplacian @ b))
        gb += self.graph_lambda / self.rank * (laplacian @ b)
        if y is not None and self.coupling:
            h = unit_columns(raw[4])
            cbasis = np.einsum('ir,jr,kr->ijkr', b, b, h, optimize=True)
            resy = (a @ cbasis.reshape(-1, self.rank).T).reshape(y.shape) - y
            mask = (1-np.eye(y.shape[1]))[None, :, :, None]
            resy *= mask
            en = max(float(np.sum((y*mask)**2)), 1e-15)
            weight = self.coupling/en
            loss += .5*weight*np.sum(resy**2)
            ga += weight * (resy.reshape(len(y), -1) @ cbasis.reshape(-1, self.rank))
            gb += weight * (np.einsum('nijk,nr,jr,kr->ir', resy,a,b,h,optimize=True)
                            + np.einsum('nijk,nr,ir,kr->jr', resy,a,b,h,optimize=True))
            gh = weight*np.einsum('nijk,nr,ir,jr->kr',resy,a,b,b,optimize=True)
            effective = [gb,gf,gs,gh]
        else:
            effective = [gb,gf,gs]
        grads = [ga.ravel()]
        for v, gradient in zip(raw[1:], effective):
            norm = np.maximum(np.linalg.norm(v, axis=0), 1e-15)
            u = v/norm
            grads.append(((gradient-u*np.sum(u*gradient,axis=0))/norm).ravel())
        return float(loss), np.concatenate(grads)

    def fit(self, x, laplacian, y=None):
        x = np.asarray(x, float)
        if not np.isfinite(x).all() or x.min() < 0:
            raise ValueError('G-NTF requires finite non-negative input')
        self.stat_scale_ = np.maximum(np.sqrt(np.mean(x*x,axis=(0,1,2))),1e-12)
        x = x/self.stat_scale_
        self.y_scale_ = 1.
        if y is not None and self.coupling:
            if not np.isfinite(y).all() or np.min(y)<0:
                raise ValueError('Connectivity must be non-negative and finite')
            self.y_scale_ = max(float(np.sqrt(np.mean(y*y))),1e-12)
            y = np.asarray(y,float)/self.y_scale_
        else:
            y = None
        self.shapes_ = [(d,self.rank) for d in x.shape]
        if y is not None:
            self.shapes_.append((y.shape[-1],self.rank))
        best = None
        diagnostics=[]
        for start in range(self.starts):
            rng=np.random.default_rng(self.seed+1009*start)
            raw=[rng.uniform(.1,1,size=shape) for shape in self.shapes_]
            b,f,s=[unit_columns(v) for v in raw[1:4]]
            basis=np.einsum('cr,fr,sr->cfsr',b,f,s,optimize=True).reshape(-1,self.rank)
            raw[0]=np.stack([nnls(basis,row.ravel(),maxiter=100*self.rank)[0] for row in x])
            v=np.concatenate([r.ravel() for r in raw])
            n_a=raw[0].size
            bounds=[(0,None)]*n_a+[(1e-8,1.)]*(len(v)-n_a)
            history=[self._loss_grad(v,x,y,laplacian)[0]]
            result=minimize(self._loss_grad,v,args=(x,y,laplacian),jac=True,method='L-BFGS-B',bounds=bounds,
                            options={'maxiter':self.maxiter,'ftol':1e-10,'gtol':1e-7,'maxls':35},
                            callback=lambda z: history.append(self._loss_grad(z,x,y,laplacian)[0]))
            diagnostics.append({'start':start,'objective':float(result.fun),'iterations':int(result.nit),
                                'converged':bool(result.success),'message':str(result.message),'history':history})
            if best is None or result.fun<best.fun:
                best=result
        factors=self._unpack(best.x)
        factors=[factors[0]]+[unit_columns(v) for v in factors[1:]]
        order=np.argsort(np.linalg.norm(factors[0],axis=0))[::-1]
        self.factors_=[v[:,order].copy() for v in factors[1:]]
        b,f,s=self.factors_[:3]
        self.basis_=np.einsum('cr,fr,sr->cfsr',b,f,s,optimize=True).reshape(-1,self.rank)
        self.weight_x_=1/max(np.sum(x*x),1e-15)
        self.conn_basis_=None
        if y is not None:
            cb=np.einsum('ir,jr,kr->ijkr',b,b,self.factors_[3],optimize=True)
            self.conn_mask_=np.broadcast_to((1-np.eye(y.shape[1]))[:,:,None],y.shape[1:]).ravel().astype(bool)
            self.conn_basis_=cb.reshape(-1,self.rank)[self.conn_mask_]
            self.weight_y_=self.coupling/max(np.sum(y.reshape(len(y),-1)[:,self.conn_mask_]**2),1e-15)
        self.diagnostics_=diagnostics
        self.objective_=float(best.fun)
        self.train_embeddings_=self.transform_unscaled(x,y)
        self.reconstruction_error_=float(np.linalg.norm(x.reshape(len(x),-1)-self.train_embeddings_@self.basis_.T)/np.linalg.norm(x))
        self.graph_roughness_=float(np.sum(b*(laplacian@b))/self.rank)
        return self

    def transform_unscaled(self,x,y=None):
        design=self.basis_*np.sqrt(self.weight_x_)
        target=x.reshape(len(x),-1)*np.sqrt(self.weight_x_)
        if self.conn_basis_ is not None:
            if y is None:
                raise ValueError('Coupled model requires connectivity for inference')
            design=np.vstack([design,self.conn_basis_*np.sqrt(self.weight_y_)])
            target=np.hstack([target,y.reshape(len(y),-1)[:,self.conn_mask_]*np.sqrt(self.weight_y_)])
        # Rescaling both sides preserves NNLS and improves stopping precision.
        normalizer=max(float(np.linalg.norm(design)),1e-12)
        return np.stack([nnls(design/normalizer,row/normalizer,maxiter=1000*self.rank)[0] for row in target])

    def transform(self,x,y=None):
        return self.transform_unscaled(np.asarray(x,float)/self.stat_scale_,None if y is None else np.asarray(y,float)/self.y_scale_)


class FeatureModel:
    def __init__(self, config, laplacian, maxiter, starts, seed):
        self.config=config
        self.laplacian=laplacian
        self.maxiter,self.starts,self.seed=maxiter,starts,seed

    def fit(self,x,sites,direct=None,connectivity=None):
        cfg=self.config
        self.decomposition=None
        if cfg['method']=='direct':
            values=np.asarray(direct,float)
        elif cfg['method']=='tt':
            from tensorly.decomposition import tensor_train
            # Common TT feature basis, sample mode LAST; fixed train-only mean.
            self.mean_=x.mean(0)
            self.scale_=max(float(np.std(x-self.mean_)),1e-12)
            z=((x-self.mean_)/self.scale_).transpose(1,2,3,0)
            rank=cfg['rank']
            cores=tensor_train(z,rank=[1,min(8,x.shape[1]),min(24,x.shape[1]*x.shape[2]),rank,1])
            basis=cores[0]
            for core in cores[1:-1]:
                basis=np.tensordot(basis,core,axes=(-1,0))
            self.tt_basis_=basis.reshape(-1,cores[-1].shape[0])
            values=((x-self.mean_)/self.scale_).reshape(len(x),-1)@self.tt_basis_
        else:
            lap=self.laplacian.copy()
            if cfg['method']=='shuffled':
                perm=np.random.default_rng(4801).permutation(len(lap))
                lap=lap[np.ix_(perm,perm)]
            self.decomposition=GraphNTF(cfg['rank'],cfg['lam'],cfg.get('coupling',0),self.maxiter,self.starts,self.seed)
            self.decomposition.fit(x,lap,connectivity)
            values=self.decomposition.transform(x,connectivity)
        # Training-only site centering AFTER decomposition preserves NTF positivity.
        self.global_mean_=values.mean(0)
        self.site_means_={s:values[np.asarray(sites)==s].mean(0) for s in set(sites)}
        centered=self.center(values,sites)
        self.scaler_=StandardScaler().fit(centered)
        scaled=self.scaler_.transform(centered)
        self.pca_=None
        if cfg['method']=='direct' and cfg.get('pca'):
            self.pca_=PCA(n_components=min(cfg['pca'],len(x)-1,scaled.shape[1]),svd_solver='full').fit(scaled)
        return self

    def center(self,values,sites):
        return values-np.stack([self.site_means_.get(s,self.global_mean_) for s in sites])

    def embedding(self,x,direct=None,connectivity=None):
        if self.config['method']=='direct':
            return np.asarray(direct,float)
        if self.config['method']=='tt':
            return ((x-self.mean_)/self.scale_).reshape(len(x),-1)@self.tt_basis_
        return self.decomposition.transform(x,connectivity)

    def transform(self,x,sites,direct=None,connectivity=None):
        values=self.scaler_.transform(self.center(self.embedding(x,direct,connectivity),sites))
        return self.pca_.transform(values) if self.pca_ is not None else values


def make_svm(config):
    return SVC(C=config['C'],kernel=config['kernel'],gamma=config.get('gamma','scale'),
               class_weight='balanced',decision_function_shape='ovr',random_state=892,probability=False)


class ProbabilityModel:
    """Sigmoid calibration fitted to out-of-fold FULL-pipeline decision scores."""
    def __init__(self,features,svm,calibrators):
        self.features,self.svm,self.calibrators=features,svm,calibrators
        self.classes_ = np.array([0,1,2])

    def predict_proba(self,x,sites,direct=None,connectivity=None):
        z=self.features.transform(x,sites,direct,connectivity)
        scores=self.svm.decision_function(z)
        p=np.column_stack([c.predict_proba(scores[:,i:i+1])[:,1] for i,c in enumerate(self.calibrators)])
        p=np.maximum(p,1e-12)
        return p/p.sum(1,keepdims=True)


def calibrators_from_oof(scores,labels):
    # Unweighted calibrators retain observed training prevalences.
    return [LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000).fit(scores[:,i:i+1],labels==i) for i in range(3)]
