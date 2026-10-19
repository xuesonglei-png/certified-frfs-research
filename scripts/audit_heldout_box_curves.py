"""Verify exact margins by explicit adversarial pairwise vertices."""
import hashlib,json
import numpy as np
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

FOLDER=ROOT/'results/heldout-box-curves'

def main():
    summary=json.loads((FOLDER/'complete.json').read_text());counts={'splits':0,'models':0,'object_radius_checks':0,'strict_negative_witnesses':0}
    for row in summary['rows']:
        name=row['dataset'];p=json.loads((FOLDER/f'{name}.json').read_text());X,y=load_dataset(name)
        for rel,digest in p['fingerprints'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
        for s in p['splits']:
            ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);assert set(ri).isdisjoint(vi)
            _,B=preprocess_fit(X[ri],X[vi]);truth=y[vi]
            for method,r in s['scores'].items():
                W,b,classes=np.array(r['coef']),np.array(r['intercept']),np.array(r['classes'])
                pred=classes[(B@W.T+b).argmax(axis=1)]
                clean=np.mean([np.mean(pred[truth==c]==c) for c in np.unique(truth)])
                np.testing.assert_allclose(clean,r['clean'],atol=1e-12)
                for eps,value in r['radii'].items():
                    radius=float(eps);expected=[]
                    for x,label in zip(B,truth):
                        own=int(np.flatnonzero(classes==label)[0]);lo=np.maximum(0,x-radius);hi=np.minimum(1,x+radius)
                        candidates=[]
                        for enemy in range(len(classes)):
                            if enemy==own:continue
                            difference=W[own]-W[enemy];vertex=np.where(difference>=0,lo,hi)
                            ownscore=W[own]@vertex+b[own];enemyscore=W[enemy]@vertex+b[enemy]
                            candidates.append(ownscore-enemyscore)
                        margin=min(candidates);expected.append(margin)
                        if margin < -1e-10:counts['strict_negative_witnesses']+=1
                    np.testing.assert_allclose(expected,value['margins'],atol=1e-10)
                    certified=np.array(expected)>1e-10
                    rate=np.mean([np.mean(certified[truth==c]) for c in np.unique(truth)])
                    np.testing.assert_allclose(rate,value['balanced_certified'],atol=1e-12)
                    counts['object_radius_checks']+=len(truth)
                counts['models']+=1
            counts['splits']+=1
        print(name,'adversarial vertices verified',flush=True)
    assert counts['splits']==44 and counts['models']==264
    counts['scope']='heldout_explicit_pairwise_vertices_strict_ties_uncertified_not_population_risk'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(counts)

if __name__=='__main__':main()
