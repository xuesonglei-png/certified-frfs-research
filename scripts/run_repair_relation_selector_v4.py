"""Checkpointed six-dataset nested development runner; no LaTeX output."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.relation_support_selector_v1 import weighted_knn
from certified_frfs.selective_support_selector_v3 import query_scores
from repair_relation_selector_v4 import fit_outer

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/repair-relation-selector-v4'
NAMES=('banknote','glass','heart_statlog','ionosphere','seeds','wdbc')

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def freeze():
    files=list((ROOT/'src/certified_frfs').glob('*.py'))
    files += [ROOT/'scripts'/p for p in ('repair_relation_selector_v4.py',
        'run_repair_relation_selector_v4.py','audit_repair_relation_selector_v4.py',
        'run_repair_relation_controls_v4.py','summarize_repair_relation_v4.py',
        'run_selective_recent_controls_v2.py','expanded_fgs_grid.py')]
    files += [ROOT/'research/repair-relation-selector-v4-protocol.md',
              ROOT/'tests/test_repair_relation_selector_v4.py']
    files += [ROOT/'data/processed'/(n+'.npz') for n in NAMES]
    current={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(files)}
    OUT.mkdir(exist_ok=True);p=OUT/'source-freeze.json'
    if p.exists():assert json.loads(p.read_text())['hashes']==current,'new version needed'
    else:p.write_text(json.dumps({'hashes':current,'created_unix':time.time(),
        'datasets':NAMES,'eps':.01,'eta':.1,'retention':.9,'cap':'2/sqrt(d)'},indent=2))
    return current

def run(name):
    threadpool_limits(1);hashes=freeze();p=OUT/(name+'.json')
    prior=json.loads(p.read_text()) if p.exists() else None
    if prior:
        assert prior['source_hashes']==hashes
        if prior['complete']:return
    rows=prior['rows'] if prior else [];X,y=load_dataset(name)
    elapsed=prior['seconds'] if prior else 0.;start=time.perf_counter()
    for fold,(ii,jj) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261011).split(X,y)):
        if fold<len(rows):continue
        inner=min(3,int(np.unique(y[ii],return_counts=True)[1].min()))
        fit=fit_outer(X[ii],y[ii],seed=20261011+fold,inner_splits=inner)
        A,B=preprocess_fit(X[ii],X[jj]);d=A.shape[1];c=fit['selected']
        models={'proposed':c,'full':{'features':list(range(d)),'weights':np.ones(d)/np.sqrt(d)},
            'matched_uniform':{'features':c['features'],'weights':np.ones(len(c['features']))/np.sqrt(len(c['features']))}}
        scores={m:query_scores(A,y[ii],B,y[jj],v) for m,v in models.items()}
        noise={m:[] for m in models}
        for draw in range(8):
            rng=np.random.default_rng(20262000+10*fold+draw)
            na=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
            nb=np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
            for m,v in models.items():
                s=weighted_knn(na,y[ii],nb,y[jj],v['features'],v['weights'])
                flags=np.array(scores[m]['query_certificate']['certified'])
                assert not np.any(flags & (np.array(s['predictions'])!=scores[m]['predictions']))
                noise[m].append(s)
        rows.append({'fold':fold,'train_indices':ii.tolist(),'test_indices':jj.tolist(),
            'fit':fit,**scores,'shared_perturbations':noise,
            'perturbed_BA':{m:float(np.mean([s['clean_BA'] for s in vals])) for m,vals in noise.items()}})
        assert hashes==freeze()
        saved={'dataset':name,'d':d,'rows':rows,'complete':len(rows)==3,'source_hashes':hashes,
            'seconds':elapsed+time.perf_counter()-start,
            'scope':'nested_exposed_development_not_confirmatory'}
        temporary=p.with_suffix('.tmp');temporary.write_text(json.dumps(saved,separators=(',',':')))
        temporary.replace(p)
        print(name,fold,'BA',round(100*scores['proposed']['clean_BA'],3),
            'full',round(100*scores['full']['clean_BA'],3),'certBA',round(100*scores['proposed']['query_certified_BA'],3),
            'support',len(c['features']),'/',d,'family',fit['chosen']['family'],
            'restored',c['restored_count'],'fallback',c['uniform_fallback'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--only',choices=NAMES);args=p.parse_args()
    for n in ((args.only,) if args.only else NAMES):run(n)
