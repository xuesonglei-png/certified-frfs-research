"""Held-out exact pairwise clipped-box classification certificates."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.linear_retention import linear_parameters,box_class_margins
from review_box_iterations import paired

FOLDER=ROOT/'results/heldout-box-curves';RADII=(0.,.005,.01,.02,.05)

def rates(A,truth,classes,W,b):
    values={};prediction=classes[(A@W.T+b).argmax(axis=1)]
    clean=float(balanced_accuracy_score(truth,prediction))
    for eps in RADII:
        margins=box_class_margins(A,truth,W,b,classes,eps=eps)
        robust=margins>1e-10
        values[str(eps)]={'balanced_certified':float(np.mean([np.mean(robust[truth==c]) for c in np.unique(truth)])),
            'margins':margins.tolist(),'ambiguous_ties':int(np.sum(abs(margins)<=1e-10))}
    assert values['0.0']['balanced_certified']<=clean+1e-12
    sequence=[values[str(e)]['balanced_certified'] for e in RADII]
    assert np.all(np.diff(sequence)<=1e-12)
    return {'clean':clean,'radii':values,'coef':W.tolist(),'intercept':b.tolist(),'classes':classes.tolist()}

def main():
    threadpool_limits(1);FOLDER.mkdir(exist_ok=True)
    names=[r['dataset'] for r in json.loads((ROOT/'results/nested-teacher-regularized/complete.json').read_text())['rows']];rows=[]
    for name in names:
        target=FOLDER/f'{name}.json'
        if target.exists():raise FileExistsError(target)
        paths=[ROOT/'results'/folder/f'{name}.json' for folder in ('teacher-support-refit','nested-teacher-regularized','nested-classifier-pipelines')]
        old,new,baseline=[json.loads(p.read_text()) for p in paths];X,y=load_dataset(name);splits=[]
        for i,s in enumerate(new['splits']):
            ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices'])
            assert ri.tolist()==old['splits'][i]['reference_indices']==baseline['splits'][i]['reference_indices']
            assert vi.tolist()==old['splits'][i]['validation_indices']==baseline['splits'][i]['validation_indices']
            A,B=preprocess_fit(X[ri],X[vi]);teacher=json.loads((ROOT/'results/linear-retention'/f'{name}.json').read_text())['splits'][i]
            configs={'teacher_support':(old['splits'][i]['selected']['fit'],np.array(teacher['teacher_classes'])),
                'nested_regularized':(s['stage']['policies'][s['chosen']]['fit'],np.array(s['stage']['teacher_classes'])),
                'nested_full':(s['outer_stages'][str(s['full_config']['C'])]['policies']['1.0']['fit'],np.array(s['stage']['teacher_classes']))}
            scores={}
            for method,(fit,classes) in configs.items():
                scores[method]=rates(B,y[vi],classes,np.array(fit['coef']),np.array(fit['intercept']))
            for method in ('full','fgs2023','sfrip2025'):
                bs=baseline['splits'][i];r=bs['results']['logistic'][method];fs=r['features'];weights=np.array(r['weights'])
                C=bs['trace']['logistic'][method][r['chosen_index']]['C']
                if fs:
                    model=LogisticRegression(C=C,max_iter=2000,random_state=0).fit(A[:,fs]*weights,y[ri])
                    coef,intercept,classes=linear_parameters(model);W=np.zeros((len(classes),X.shape[1]));W[:,fs]=coef*weights
                else:
                    classes,counts=np.unique(y[ri],return_counts=True);W=np.zeros((len(classes),X.shape[1]));intercept=np.log(counts/counts.sum())
                scores['archived_'+method]=rates(B,y[vi],classes,W,intercept)
                np.testing.assert_allclose(scores['archived_'+method]['clean'],r['metrics']['clean'],atol=1e-12)
            np.testing.assert_allclose(scores['nested_regularized']['clean'],s['metrics']['clean'],atol=1e-12)
            np.testing.assert_allclose(scores['nested_full']['clean'],s['full_metrics']['clean'],atol=1e-12)
            splits.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'scores':scores})
        paths += [Path(__file__),ROOT/'research/heldout-box-evaluation-protocol.md',ROOT/'src/certified_frfs/linear_retention.py',ROOT/'src/certified_frfs/validated_pool.py',ROOT/'data/processed'/f'{name}.npz']
        target.write_text(json.dumps({'dataset':name,'splits':splits,'fingerprints':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2),encoding='utf-8')
        means={m:{str(e):float(np.mean([s['scores'][m]['radii'][str(e)]['balanced_certified'] for s in splits])) for e in RADII} for m in splits[0]['scores']}
        rows.append({'dataset':name,'means':means});print(name,'held-out box curves scored',flush=True)
    methods=list(rows[0]['means']);means={m:{str(e):float(np.mean([r['means'][m][str(e)] for r in rows])) for e in RADII} for m in methods}
    comparisons={m:{str(e):paired([r['means']['nested_regularized'][str(e)]-r['means'][m][str(e)] for r in rows]) for e in RADII}
                 for m in methods if m!='nested_regularized'}
    result={'rows':rows,'means':means,'comparisons':comparisons,'scope':'heldout_box_strict_margin_coverage_repeatedly_exposed_development_not_population_guarantee'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(means,flush=True)

if __name__=='__main__':main()
