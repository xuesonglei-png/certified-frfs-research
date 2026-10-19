"""Independent coordinate distances, direct class counters and query replay."""
import numpy as np
from sklearn.metrics import f1_score
from audit_certified_metric_knn_pool import query_certificate
from audit_forest_compact_transfer import equivalent


def independent_evaluate(A,y,B,truth,metric):
    A,y,B,truth=np.asarray(A,float),np.asarray(y),np.asarray(B,float),np.asarray(truth)
    classes=np.unique(y);fs,w=metric['features'],metric['weights']
    predictions=[];probabilities=[];neighbors=[];ties=[]
    if fs:
        distance=np.zeros((len(B),len(A)))
        for a,value in zip(fs,w):distance+=abs(B[:,a,None]*value-A[None,:,a]*value)
        for row in distance:
            ids=np.argsort(row,kind='stable')[:5].tolist();neighbors.append(ids)
            counter={c:sum(y[j]==c for j in ids) for c in classes}
            top=max(counter.values());contenders=[c for c in classes if counter[c]==top]
            score={c:(sum(float(row[j]) for j in ids if y[j]==c)/counter[c],
                      min(j for j in ids if y[j]==c)) for c in contenders}
            predictions.append(min(contenders,key=lambda c:score[c]).item())
            probabilities.append([(counter[c]+1)/(5+len(classes)) for c in classes])
            ties.append(len(contenders)>1)
        rule='strict_plurality_then_mean_neighbor_distance_then_reference_id'
    else:
        counts={c:int(sum(y==c)) for c in classes};top=max(counts.values())
        contenders=[c for c in classes if counts[c]==top]
        winner=min(contenders,key=lambda c:next(j for j in range(len(y)) if y[j]==c))
        predictions=[winner.item()]*len(B);neighbors=[[] for _ in B]
        probabilities=[[(counts[c]+1)/(len(y)+len(classes)) for c in classes] for _ in B]
        ties=[len(contenders)>1]*len(B);rule='empty_prior_then_reference_identity'
    pred=np.array(predictions);prob=np.array(probabilities)
    def balanced(v):return float(np.mean([np.mean(v[truth==c]) for c in np.unique(truth)]))
    own=np.searchsorted(classes,truth)
    metrics={'clean_BA':balanced(pred==truth),
             'balanced_logloss':balanced(-np.log(prob[np.arange(len(B)),own])),
             'macro_F1':float(f1_score(truth,pred,average='macro',zero_division=0))}
    return {'predictions':predictions,'probabilities':probabilities,'neighbor_indices':neighbors,
            'tied_queries':ties,'rule':rule,'metrics':metrics}


def replay_final(A,y,B,truth,stored,counts):
    metric=stored['metric'];report=independent_evaluate(A,y,B,truth,metric)
    cert=query_certificate(A,y,B,np.array(report['predictions']),metric['features'],metric['weights'])
    equivalent(cert,stored['query_certificate'])
    correct=(np.array(report['predictions'])==truth)&np.array(cert['certified'])
    report['metrics']['query_certified_BA']=float(np.mean([np.mean(correct[truth==c]) for c in np.unique(truth)]))
    assert len(stored['perturbation_evaluations'])==16;noisy=[]
    for draw,frozen in enumerate(stored['perturbation_evaluations']):
        rng=np.random.default_rng(20261060+draw)
        ref=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
        query=np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
        fresh=independent_evaluate(ref,y,query,truth,metric);equivalent(fresh,frozen)
        assert not np.any((np.array(fresh['predictions'])!=np.array(report['predictions']))&np.array(cert['certified']))
        noisy.append(fresh['metrics']['clean_BA']);counts['independent_noise_predictions']+=1
    report['metrics']['mean_perturbed_BA']=float(np.mean(noisy))
    equivalent(report,stored['evaluation']);counts['independent_outer_predictions']+=1
    return report['metrics']
