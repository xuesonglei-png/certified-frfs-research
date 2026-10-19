"""Independent training holdout selection over frozen certified paths.

Validation selects features and a metric family; it does not certify risk.
Final refit must obtain a fresh empirical dependency certificate.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from .triplet_metric import learn_triplet_weights
from .graph_recent import certified_backward_candidates
from .core import certificate
from . import baselines


def preprocess_fit(reference, query, *, geometry='minmax'):
    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    if geometry == 'minmax':
        scaler=MinMaxScaler(clip=True)
    elif geometry == 'quantile':
        scaler=QuantileTransformer(n_quantiles=min(64,len(reference)),
            output_distribution='uniform',subsample=100000,random_state=0)
    else:
        raise ValueError('unknown geometry')
    A=scaler.fit_transform(imputer.fit_transform(reference))
    B=scaler.transform(imputer.transform(query))
    return A,B


def metric_weights(A,y,family,seed):
    if family=='uniform':return np.ones(A.shape[1])/np.sqrt(A.shape[1])
    if family=='triplet':return learn_triplet_weights(A,y,seed=seed)
    raise ValueError('unknown metric family')


def validation_scores(A,y,B,truth,fs,w,seed,*,joint=False):
    model=KNeighborsClassifier(n_neighbors=min(5,len(y)),metric='manhattan',algorithm='brute')
    model.fit(A[:,fs]*w[fs],y)
    rng=np.random.default_rng(seed)
    clean=float(balanced_accuracy_score(truth,model.predict(B[:,fs]*w[fs])))
    perturbed=[]
    for _ in range(3):
        if joint:
            ref=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
            model.fit(ref[:,fs]*w[fs],y)
        Z=np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
        perturbed.append(balanced_accuracy_score(truth,model.predict(Z[:,fs]*w[fs])))
    return {'clean':clean,'noise01':float(np.mean(perturbed))}


def select_validated_pool(X,y,*,method='pool',mode='robust',seed=101,joint=False):
    X,y=np.asarray(X,float),np.asarray(y)
    stability='exact' if method=='pool_ldfrs_exact' else 'interval'
    if method=='pool_ldfrs_exact':method='pool_ldfrs'
    if mode not in ['nominal','robust','compression']:
        raise ValueError('unknown mode')
    discovery,validation=train_test_split(np.arange(len(y)),test_size=.3,
                                          stratify=y,random_state=seed+41)
    A,B=preprocess_fit(X[discovery],X[validation]);d=A.shape[1]
    trace=[];paths=[]
    for family in ['uniform','triplet']:
        w=metric_weights(A,y[discovery],family,seed)
        if method in ('pool','pool_ldfrs'):
            candidates={tuple(range(d))}
            for rank in ['mi','relief']:
                relevance=(baselines.mi_scores(A*w,y[discovery],random_state=seed) if rank=='mi'
                           else baselines.relieff_scores(A*w,y[discovery],random_state=seed))
                ranking=np.argsort(relevance,kind='stable').tolist()
                path=certified_backward_candidates(A,y[discovery],ranking,weights=w,eta=.1,eps=.01,stability=stability)
                candidates.update(tuple(fs) for fs in path['candidates'])
                paths.append({'family':family,'rank':rank,**path})
            if method=='pool_ldfrs':
                from .ldfrs import rank_ldfrs
                k=min(10,int(np.unique(y[discovery],return_counts=True)[1].min())-1)
                for convention in ['inclusive_formula','toy_reconciled']:
                    ranking=rank_ldfrs(A*w,y[discovery],k=k,convention=convention)['ranking']
                    path=certified_backward_candidates(A,y[discovery],ranking[::-1],weights=w,eta=.1,eps=.01,stability=stability)
                    candidates.update(tuple(fs) for fs in path['candidates'])
                    paths.append({'family':family,'rank':'ldfrs_'+convention,**path})
            candidates=sorted(candidates,key=lambda fs:(len(fs),fs))
        elif method=='all':candidates=[tuple(range(d))]
        elif method in ['mi','relief','mrmr']:
            function={'mi':baselines.select_mi,'relief':baselines.select_relieff,'mrmr':baselines.select_mrmr}[method]
            candidates=[tuple(function(A*w,y[discovery],k=max(1,int(np.ceil(f*d)))))
                        for f in [.2,.35,.5,.65,.8,1.]]
        elif method=='ldfrs':
            from .ldfrs import rank_ldfrs
            k=min(10,int(np.unique(y[discovery],return_counts=True)[1].min())-1)
            candidates=[]
            for convention in ['inclusive_formula','toy_reconciled']:
                ranking=rank_ldfrs(A*w,y[discovery],k=k,convention=convention)['ranking']
                candidates.extend(tuple(ranking[:max(1,int(np.ceil(f*d)))])
                                  for f in [.2,.35,.5,.65,.8,1.])
        elif method=='sfrip':
            from .recent import select_sfrip
            candidates=[tuple(select_sfrip(A*w,y[discovery],beta=b,confidence=c)['features'])
                        for b in [2.,3.,5.] for c in [.6,.8]]
        else:raise ValueError('unknown method')
        for fs in sorted(set(candidates)):
            if not fs:continue
            metrics=validation_scores(A,y[discovery],B,y[validation],list(fs),w,seed+43,joint=joint)
            score=metrics['clean'] if mode=='nominal' else (metrics['clean']+metrics['noise01'])/2
            trace.append({'family':family,'features':list(fs),'score':score,'validation':metrics})
    if not trace:raise ValueError('no nonempty candidates')
    if mode=='compression':
        # A fixed empirical tolerance is a tuning rule, not a risk guarantee.
        maximum=max(row['score'] for row in trace)
        eligible=[row for row in trace if row['score']>=maximum-.01-1e-12]
        chosen=max(eligible,key=lambda row:(-len(row['features']),row['score'],row['family']=='uniform'))
    else:
        chosen=max(trace,key=lambda row:(row['score'],-len(row['features']),row['family']=='uniform'))
    full,_=preprocess_fit(X,X)
    w=metric_weights(full,y,chosen['family'],seed)
    fs=list(chosen['features']);restoration=[]
    final_cert=None
    if method in ('pool','pool_ldfrs'):
        # Refit changes the metric and table: discovery certificates cannot
        # transfer. Restore in decreasing learned weight until freshly feasible.
        order=np.argsort(-w,kind='stable').tolist()
        while True:
            final_cert=certificate(full,y,fs,weights=w,eps=.01,stability=stability)
            if final_cert['upper_bound']<=.1+1e-10:break
            feature=next(a for a in order if a not in fs)
            fs=sorted(fs+[feature]);restoration.append(feature)
    return {'features':fs,'weights':w.tolist(),'family':chosen['family'],
            'discovery_indices':discovery.tolist(),'validation_indices':validation.tolist(),
            'chosen':chosen,'trace':trace,'paths':paths,'restoration':restoration,
            'certificate':final_cert,'preprocessed_training':full,
            'joint_validation':bool(joint),
            'stability':stability,
            'scope':'training_holdout_selection_fresh_fixed_metric_dependency_not_test_risk'}
