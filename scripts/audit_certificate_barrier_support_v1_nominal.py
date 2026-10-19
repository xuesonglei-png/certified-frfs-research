"""Separate nominal primitive replay for every screened training support.

Shares the frozen deployed weight recipe, and independently rebuilds distances,
class losses, nominal protected deficits and hard-vote BA. This is explicit
numerical replay, not formal directed-rounding or an independent optimizer.
"""
import json
from pathlib import Path
import numpy as np
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certificate_barrier_support_v1 import metric_weights
from audit_relation_support_selector_v1 import independent_protected
from run_repair_relation_selector_v4 import ROOT,NAMES

FOLDER=ROOT/'results/certificate-barrier-support-v1'


def distance(A,fs,w):
    D=np.zeros((len(A),len(A)))
    for a,v in zip(fs,w,strict=True):D+=v*abs(A[:,a,None]-A[None,:,a])
    return D


def main():
    checked=0;rows=[]
    for name in NAMES:
        result=json.loads((FOLDER/(name+'.json')).read_text())
        X,y=load_dataset(name);ids=np.asarray(result['enclosing_training_indices'])[result['inner_training_indices']]
        A,_=preprocess_fit(X[ids],X[ids]);labels=y[ids];classes=np.unique(labels)
        ref=np.ones(A.shape[1])/np.sqrt(A.shape[1]);protected=independent_protected(A,labels,ref)
        cross=labels[:,None]!=labels[None,:]
        oldgamma=np.minimum(1.,np.min(np.where(cross,distance(A,range(A.shape[1]),ref),np.inf),axis=1))
        recorded={tuple(x['features']):x['proxy'] for x in result['checks']}
        for step in result['proposals']:
            for item in step['screened']:
                fs=tuple(item['features'])
                if fs in recorded:assert recorded[fs]==item['proxy']
                recorded[fs]=item['proxy']
        for fs,p in recorded.items():
            w=metric_weights(A,fs,ref);D=distance(A,fs,w)
            gamma=np.minimum(1.,np.min(np.where(cross,D,np.inf),axis=1))
            loss=np.maximum(0.,oldgamma-gamma)
            expected=[float(loss[labels==c].mean()) for c in classes]
            np.testing.assert_allclose(expected,p['nominal_class_loss'],rtol=0,atol=2e-14)
            np.fill_diagonal(D,np.inf);neighbors=np.argsort(D,axis=1,kind='stable')[:,:min(5,len(labels)-1)]
            votes=np.column_stack([np.sum(labels[neighbors]==c,axis=1) for c in classes])
            correct=classes[votes.argmax(axis=1)]==labels
            ba=float(np.mean([correct[labels==c].mean() for c in classes]))
            assert abs(ba-p['training_LOO_BA'])<1e-12,(name,fs,ba,p['training_LOO_BA'])
            deficits=[]
            for c in classes:
                mask=(labels==c)&protected;required=int(np.ceil(.9*mask.sum()-1e-12))
                deficits.append(max(0.,(required-int(sum(correct&mask)))/max(1,required)))
            assert deficits==p['nominal_protected_deficits']
        assert len(recorded)==result['proxy_checks'];checked+=len(recorded)
        rows.append({'dataset':name,'screened_supports_replayed':len(recorded)})
        print(name,'all nominal primitive replayed',len(recorded),flush=True)
    report={'passed':True,'datasets':6,'supports':checked,'rows':rows,
            'weight_recipe_shared':True,'distance_and_protected_flags_separate':True,
            'new_outer_scoring':False,'no_new_model_selection':True}
    (FOLDER/'nominal-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
