"""Geometric plurality ties, original query certificate and16 noise draws."""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score
from label_equivariant_vote import predict
from certified_frfs.knn_query import certify_knn_queries


def evaluate(A,y,B,truth,metric):
    A,y,B,truth=np.asarray(A,float),np.asarray(y),np.asarray(B,float),np.asarray(truth)
    fs,w=metric['features'],metric['weights'];classes=np.unique(y)
    if fs:
        distance=cdist(B[:,fs]*w,A[:,fs]*w,'cityblock')
        report=predict(distance,y)
    else:
        probabilities=np.tile([(sum(y==c)+1)/(len(y)+len(classes)) for c in classes],(len(B),1))
        maximum=probabilities[0].max();tied=classes[probabilities[0]==maximum]
        winner=min(tied,key=lambda c:int(np.flatnonzero(y==c)[0]))
        report={'predictions':[winner.item()]*len(B),'probabilities':probabilities.tolist(),
            'neighbor_indices':[[] for _ in B],'tied_queries':[len(tied)>1]*len(B),
            'rule':'empty_prior_then_reference_identity'}
    probabilities=np.array(report['probabilities']);pred=np.array(report['predictions'])
    def balanced(value):return float(np.mean([np.mean(value[truth==c]) for c in np.unique(truth)]))
    own=np.searchsorted(classes,truth)
    report['metrics']={'clean_BA':balanced(pred==truth),
        'balanced_logloss':balanced(-np.log(probabilities[np.arange(len(B)),own])),
        'macro_F1':float(f1_score(truth,pred,average='macro',zero_division=0))}
    return report


def final_evaluation(A,y,B,truth,metric):
    report=evaluate(A,y,B,truth,metric);fs,w=metric['features'],metric['weights']
    if fs:
        embedded=np.zeros(A.shape[1]);embedded[fs]=w
        cert=certify_knn_queries(A,y,B,np.array(report['predictions']),fs,weights=embedded,k=5,allowed_enemies=2)
    else:cert={'certified':[False]*len(B),'scope':'no_neighborhood_certificate_for_empty_prior'}
    correct=(np.array(report['predictions'])==truth)&np.array(cert['certified'])
    report['metrics']['query_certified_BA']=float(np.mean([np.mean(correct[truth==c]) for c in np.unique(truth)]))
    noisy=[]
    for draw in range(16):
        rng=np.random.default_rng(20261060+draw)
        na=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1);nb=np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
        fresh=evaluate(na,y,nb,truth,metric)
        assert not np.any((np.array(fresh['predictions'])!=np.array(report['predictions']))&np.array(cert['certified']))
        noisy.append(fresh)
    report['metrics']['mean_perturbed_BA']=float(np.mean([r['metrics']['clean_BA'] for r in noisy]))
    return {'evaluation':report,'query_certificate':cert,'perturbation_evaluations':noisy}
