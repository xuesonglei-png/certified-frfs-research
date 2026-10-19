"""All audited inner guard metrics, class-star loss only; no classifier fit."""
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT,digest
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.core import positive_region
from classwise_star_dependency import classwise_star_dependency
from audit_forest_compact_transfer import equivalent

SOURCE=ROOT/'results/majority-hinge-inner-all15'
FOLDER=ROOT/'results/classwise-star-admission'


def run(name):
    threadpool_limits(1);FOLDER.mkdir(exist_ok=True)
    freeze=json.loads((ROOT/'research/classwise-star-admission-source-freeze.json').read_text())
    for rel,sha in freeze['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    audit=verify(SOURCE);assert audit['artifact_hashes'][name]==digest(SOURCE/(name+'.json'))
    hashes={**freeze['fingerprints'],(SOURCE/'complete-audit.json').relative_to(ROOT).as_posix():digest(SOURCE/'complete-audit.json')}
    out=FOLDER/(name+'.json')
    if out.exists():
        saved=json.loads(out.read_text());assert saved['fingerprints']==hashes
        if saved['complete']:return
    source=json.loads((SOURCE/(name+'.json')).read_text());X,y=load_dataset(name);ri=np.array(source['reference_indices'])
    raw,labels=X[ri],y[ri];inner=[]
    for fi,b in enumerate(source['inner']):
        ca=np.array(b['reference_indices']);A,_=preprocess_fit(raw[ca],raw[ca]);train_y=labels[ca]
        full=np.ones(A.shape[1])/np.sqrt(A.shape[1]);classes=np.unique(train_y);checks=[]
        for guard in b['stage']['certificate_checks']:
            fs,w=guard['features'],guard['weights']
            report=classwise_star_dependency(A,train_y,fs,full,w,base_dependency=guard['certificate']['dependency_certificate'])
            equivalent(report['original_class_certificate'],guard['class_certificate'])
            new_feasible=bool(guard['certificate']['majority_feasible'] and report['global_upper_bound']<=.1
                              and report['class_certificate']['feasible_all_classes'])
            assert not guard['accepted'] or new_feasible
            wide=np.zeros(A.shape[1]);wide[fs]=w;draws=[]
            for draw in range(3):
                rng=np.random.default_rng(20261111+draw);Z=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
                loss=positive_region(Z,train_y,weights=full)-positive_region(Z,train_y,fs,weights=wide)
                conditional=[float(loss[train_y==c].mean()) for c in classes]
                assert np.all(np.array(conditional)<=np.array(report['class_certificate']['upper_bounds'])+1e-10)
                draws.append(conditional)
            checks.append({'guard_id':guard['id'],'features':fs,'weights':w,'old_accepted':guard['accepted'],
                'new_feasible':new_feasible,'new_admission':bool(new_feasible and not guard['accepted']),
                'majority_feasible_unchanged':guard['certificate']['majority_feasible'],'certificate':report,
                'shared_table_sampled_class_losses':draws,'classification_scored':False})
        inner.append({'reference_indices':ca.tolist(),'checks':checks})
        print(name,'inner',fi+1,'ALL unique class-star guard checks',len(checks),'new admissions',sum(r['new_admission'] for r in checks),flush=True)
    saved={'dataset':name,'reference_indices':ri.tolist(),'fingerprints':hashes,'inner':inner,'complete':True,
        'goal_achieved':False,'classification_scored':False,'scope':'all15_inner_classwise_star_certificate_admission_only'}
    out.write_text(json.dumps(saved,separators=(',',':')),encoding='utf-8')


def main():
    FOLDER.mkdir(exist_ok=True);names=sorted(verify(SOURCE)['artifact_hashes'])
    with ProcessPoolExecutor(max_workers=2) as executor:
        for f in as_completed([executor.submit(run,n) for n in names]):f.result()
    rows=[]
    for n in names:
        s=json.loads((FOLDER/(n+'.json')).read_text());checks=[r for b in s['inner'] for r in b['checks']]
        rows.append({'dataset':n,'guard_checks':len(checks),'new_admissions':sum(r['new_admission'] for r in checks),
            'class_bounds_strictly_tightened':sum(sum(np.array(r['certificate']['class_certificate']['upper_bounds'])<
                np.array(r['certificate']['original_class_certificate']['upper_bounds'])-1e-12) for r in checks),
            'total_class_bounds':sum(len(r['certificate']['class_certificate']['upper_bounds']) for r in checks)})
    summary={'rows':rows,'counts':{k:sum(r[k] for r in rows) for k in ('guard_checks','new_admissions','class_bounds_strictly_tightened','total_class_bounds')},
        'datasets':len(names),'inner_folds':sum(len(json.loads((FOLDER/(n+'.json')).read_text())['inner']) for n in names),
        'artifact_hashes':{n:digest(FOLDER/(n+'.json')) for n in names},'goal_achieved':False,
        'independent_audit_required':True,'classification_scored':False,'scope':'certificate_tightness_and_admission_not_prediction_superiority'}
    (FOLDER/'complete.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
