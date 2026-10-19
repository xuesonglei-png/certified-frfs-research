"""Training search with subset-specific transferred metric weights."""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .validated_pool import preprocess_fit
from .softvote_metric import learn_softvote_weights
from .core import _inputs
from .utility_search import UtilityWorkspace
from .majority_metric import MajorityTerms
from .adaptive_witness import adaptive_bounds
from .redistributed_loss import transfer_weights,redistributed_loss_bound


class RedistributedUtilityWorkspace:
    def __init__(self,X,y,w_full,*,eps=.01,seed=101,ranking_workspace=None):
        self.X,self.y,self.full,self.w=_inputs(X,y,None,w_full)
        self.eps=eps;self.seed=seed;self.cache={};self.ranking_workspace=ranking_workspace
        self.fallback=UtilityWorkspace(self.X,self.y,self.w,eps=eps,seed=seed)

    def evaluate(self,features):
        fs=tuple(features)
        if fs not in self.cache:
            wr=transfer_weights(self.X,fs,self.w)
            # Recompute the two LOO utilities under this subset's metric.
            d=[]
            rng=np.random.default_rng(self.seed)
            tables=[self.X,np.clip(self.X+rng.uniform(-self.eps,self.eps,self.X.shape),0,1)]
            codes=self.fallback.codes;balance=self.fallback.balance;k=self.fallback.k
            for Z in tables:
                D=np.abs(Z[:,None,fs]-Z[None,:,fs])@wr;np.fill_diagonal(D,np.inf)
                nn=np.argsort(D,axis=1,kind='stable')[:,:k]
                votes=np.stack([(codes[nn]==label).sum(axis=1) for label in range(len(self.fallback.labels))],axis=1)
                d.append(float(balance@(votes.argmax(axis=1)==codes)))
            self.cache[fs]={'clean_LOO':d[0],'shared_noise_LOO':d[1],
                            'utility':float(np.mean(d)),'reduced_weights':wr.tolist(),
                            'score_scope':'training_LOO_subset_specific_transferred_metric'}
        return self.cache[fs]


class CrossfitRedistributedUtilityWorkspace:
    """Subset-specific transfer and 5NN utility fit only inside training folds."""
    def __init__(self,X,y,*,family='uniform',eps=.01,seed=101):
        self.X=np.asarray(X,float);self.y=np.asarray(y);self.eps=eps;self.seed=seed
        if family not in ('uniform','softvote'):raise ValueError('unknown metric family')
        labels,counts=np.unique(self.y,return_counts=True);ids=np.flatnonzero(np.isin(self.y,labels[counts>1]))
        single=np.flatnonzero(np.isin(self.y,labels[counts==1]))
        if len(ids)==0:raise ValueError('crossfit transfer utility needs repeated classes')
        nfold=min(3,int(counts[counts>1].min()));self.folds=[];self.cache={};self.splits=[]
        for index,(ii,jj) in enumerate(StratifiedKFold(nfold,shuffle=True,random_state=seed+41).split(ids,self.y[ids])):
            tr=np.r_[ids[ii],single];va=ids[jj];A,B=preprocess_fit(self.X[tr],self.X[va])
            w=np.ones(A.shape[1])/np.sqrt(A.shape[1]) if family=='uniform' else learn_softvote_weights(A,self.y[tr])
            self.folds.append((A,B,self.y[tr],self.y[va],w));self.splits.append({'train':tr.tolist(),'validation':va.tolist(),'weights':w.tolist()})

    def evaluate(self,features):
        fs=tuple(features)
        if not fs:raise ValueError('empty distance subset')
        if fs not in self.cache:
            clean=[];noisy=[];weights=[]
            for index,(A,B,yt,yv,w) in enumerate(self.folds):
                wr=transfer_weights(A,fs,w);D=np.abs(B[:,None,fs]-A[None,:,fs])@wr
                nn=np.argsort(D,axis=1,kind='stable')[:,:min(5,len(yt))];labels=np.unique(yt)
                pred=np.stack([(yt[nn]==q).sum(axis=1) for q in labels],axis=1).argmax(axis=1)
                clean.append(float(np.mean([(labels[pred][yv==q]==q).mean() for q in np.unique(yv)])))
                rng=np.random.default_rng(self.seed+43+index);Z=np.clip(A+rng.uniform(-self.eps,self.eps,A.shape),0,1);Q=np.clip(B+rng.uniform(-self.eps,self.eps,B.shape),0,1)
                DD=np.abs(Q[:,None,fs]-Z[None,:,fs])@wr;nn=np.argsort(DD,axis=1,kind='stable')[:,:min(5,len(yt))]
                pred=np.stack([(yt[nn]==q).sum(axis=1) for q in labels],axis=1).argmax(axis=1)
                noisy.append(float(np.mean([(labels[pred][yv==q]==q).mean() for q in np.unique(yv)])));weights.append(wr)
            self.cache[fs]={'clean_LOO':float(np.mean(clean)),'shared_noise_LOO':float(np.mean(noisy)),
                'utility':float((np.mean(clean)+np.mean(noisy))/2),'reduced_weights':np.mean(weights,axis=0).tolist(),
                'score_scope':'adaptive_training_crossfit_subset_specific_transfer'}
        return self.cache[fs]


def redistributed_path(X,y,w_full,*,certified=True,retention=.9,eps=.01,seed=101,dependency_eta=.1,ranking_workspace=None):
    X,y,full,w=_inputs(X,y,None,w_full);d=X.shape[1]
    if not 0<retention<=1 or not 0<=dependency_eta<=1:raise ValueError('invalid constraints')
    utility=ranking_workspace or RedistributedUtilityWorkspace(X,y,w,eps=eps,seed=seed)
    fixed=MajorityTerms(X,y,weights=w,eps=eps) if certified else None
    protected=fixed.bounds(range(d),allowed_enemies=2)>1e-10 if certified else np.zeros(len(y),bool)
    groups=[(y==label)&protected for label in np.unique(y)]
    required=[int(np.ceil(retention*g.sum()-1e-12)) for g in groups]
    current=list(range(d));candidates=[current.copy()];trace=[];cache={}
    def certify(fs):
        key=tuple(fs)
        if key not in cache:
            wr=transfer_weights(X,fs,w)
            if not certified:
                cache[key]={'retained':[],'reduced_weights':wr.tolist(),'majority_feasible':True,'feasible':True}
                return cache[key]
            reduced=MajorityTerms(X[:,fs],y,weights=wr,eps=eps).bounds(range(len(fs)),allowed_enemies=2)
            counts=[int(((reduced>1e-10)&g).sum()) for g in groups]
            majority=all(c>=r for c,r in zip(counts,required))
            detail={'retained':counts,'reduced_weights':wr.tolist(),'majority_feasible':majority,'feasible':False}
            if majority:
                dep=redistributed_loss_bound(X,y,fs,w,wr,eps=eps)
                detail['dependency_certificate']=dep;detail['feasible']=dep['upper_bound']<=dependency_eta
            cache[key]=detail
        return cache[key]
    while len(current)>1:
        moves=[]
        for a in current:
            fs=[b for b in current if b!=a];scores=utility.evaluate(fs)
            moves.append({'feature':a,'features':fs,**scores})
        moves.sort(key=lambda r:(-r['utility'],-r['clean_LOO'],r['feature']))
        accepted=None;attempts=[]
        for move in moves:
            detail=certify(move['features']);attempts.append({**move,**detail})
            if detail['feasible']:accepted=move;break
        trace.append({'current':current.copy(),'ranked_moves':moves,'certification_attempts':attempts,
                      'accepted':accepted['features'] if accepted else None})
        if accepted is None:break
        current=accepted['features'];candidates.append(current.copy())
    result={'candidates':candidates,'trace':trace,'required':required,'protected_counts':[int(g.sum()) for g in groups],
        'training_utility_evaluations':len(utility.cache),'certificate_evaluations':len(cache),
        'scope':'training_subset_specific_metric_redistribution_and_finite_dependency_certificate_not_test_risk',
        'certified':certified,'eps':eps,'seed':seed,'dependency_eta':dependency_eta,
        'metric_transfer':'correlation_power2_preserve_full_weight_sum'}
    if ranking_workspace is not None:
        result['ranking_scope']='adaptive_training_crossfit_not_validation_or_test';result['ranking_splits']=ranking_workspace.splits
    return result


def redistributed_candidates(X,y,w,**kwargs):
    from .crossfit_majority import FRACTIONS
    path=redistributed_path(X,y,w,**kwargs);d=np.asarray(X).shape[1];rows=[]
    for fraction in FRACTIONS:
        target=max(1,int(np.ceil(fraction*d)))
        fs=min(path['candidates'],key=lambda f:(abs(len(f)-target),-len(f),tuple(f)))
        rows.append({'config':{'fraction':fraction},'features':fs,
                     'reduced_weights':transfer_weights(X,fs,w).tolist()})
    return rows,path
