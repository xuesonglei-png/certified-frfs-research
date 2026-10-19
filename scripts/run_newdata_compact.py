"""Frozen end-to-end experiment on eight additional numeric UCI tasks."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import time
from urllib.request import urlopen
import numpy as np
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split, StratifiedKFold
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from compare_trajectory_recent import GRIDS, SELECTORS, score, utility
from certified_frfs.feasible_query_descent import fit_feasible_query
from certified_frfs.trajectory_certificate import certify_capped_trajectory
from certified_frfs.capped_sparse import sparse_trajectory_extension
from certified_frfs.validated_pool import preprocess_fit, validation_scores

FOLDER = ROOT/'results/newdata-compact'; DATA = ROOT/'data/newdata-compact'
BASE = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
SOURCES = {
    'balance_scale': (['balance-scale/balance-scale.data'], 'first'),
    'breast_original': (['breast-cancer-wisconsin/breast-cancer-wisconsin.data'], 'last_drop_id'),
    'dermatology': (['dermatology/dermatology.data'], 'last'),
    'lymphography': (['lymphography/lymphography.data'], 'first'),
    'libras': (['libras/movement_libras.data'], 'last'),
    'parkinsons': (['parkinsons/parkinsons.data'], 'header'),
    'spectf': (['spect/SPECTF.train', 'spect/SPECTF.test'], 'first'),
    'spambase': (['spambase/spambase.data'], 'last')}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(name):
    paths, convention = SOURCES[name]; raw_paths = []; rows = []
    for endpoint in paths:
        path = DATA/(name+'-'+Path(endpoint).name)
        if not path.exists():
            with urlopen(BASE+endpoint, timeout=60) as response: payload = response.read()
            if b'<html' in payload.lower(): raise ValueError(f'HTML instead of numeric data: {endpoint}')
            path.write_bytes(payload)
        raw_paths.append(path); text = path.read_text(encoding='utf-8-sig')
        if convention == 'header':
            for record in csv.DictReader(io.StringIO(text)):
                rows.append(([record[k] for k in record if k not in ('name', 'status')], record['status']))
        else:
            for record in csv.reader(io.StringIO(text)):
                if not record: continue
                if convention=='first': predictors, label = record[1:], record[0]
                elif convention=='last_drop_id': predictors, label = record[1:-1], record[-1]
                else: predictors, label = record[:-1], record[-1]
                rows.append((predictors, label.strip()))
    X = np.array([[np.nan if v.strip()=='?' else float(v) for v in r[0]] for r in rows], float)
    classes, y = np.unique([r[1] for r in rows], return_inverse=True)
    assert X.ndim==2 and len(X)==len(y) and np.all(np.isfinite(X)|np.isnan(X))
    ids = np.arange(len(X))
    if len(ids)>600: ids,_ = train_test_split(ids,train_size=600,stratify=y,random_state=13100)
    payload = {'X':X[ids], 'y':y[ids], 'source_indices':ids}
    processed = DATA/f'{name}.npz'
    if processed.exists():
        with np.load(processed) as cached:
            for key,value in payload.items(): assert np.array_equal(cached[key],value,equal_nan=True)
    else: np.savez_compressed(processed,**payload)
    metadata = {'source_urls':[BASE+p for p in paths], 'raw_sha256':{p.name:digest(p) for p in raw_paths},
        'processed_sha256':digest(processed), 'original_n':len(X),'sampled_n':len(ids),'d':X.shape[1],
        'classes':classes.tolist(),'parsing':convention,'source_indices':ids.tolist()}
    (DATA/f'{name}-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(name, 'downloaded', len(ids), X.shape[1], flush=True)


def sources(name):
    paths = [Path(__file__),ROOT/'scripts/compare_trajectory_recent.py',ROOT/'research/newdata-compact-protocol.md',
        DATA/f'{name}.npz',DATA/f'{name}-metadata.json']
    paths.extend(sorted((ROOT/'src/certified_frfs').glob('*.py')))
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def run(name):
    threadpool_limits(1); fingerprints = sources(name); out = FOLDER/f'{name}.json'
    if out.exists():
        assert json.loads(out.read_text())['fingerprints']==fingerprints
        return
    with np.load(DATA/f'{name}.npz') as data: X,y=data['X'],data['y']
    splits = []
    for fold,(tr,te) in enumerate(StratifiedKFold(3,shuffle=True,random_state=13101).split(X,y)):
        start = time.perf_counter()
        trajectory=fit_feasible_query(X[tr],y[tr],certified=False,weight_cap_ratio=2)
        certified=certify_capped_trajectory(X[tr],y[tr],trajectory)
        fit=sparse_trajectory_extension(X[tr],y[tr],certified)
        elapsed=time.perf_counter()-start; selected=fit['compact_selected']; A,B=preprocess_fit(X[tr],X[te])
        proposed=validation_scores(A,y[tr],B,y[te],selected['features'],np.array(selected['weights']),144+fold,joint=True)
        full=validation_scores(A,y[tr],B,y[te],list(range(X.shape[1])),np.ones(X.shape[1])/np.sqrt(X.shape[1]),144+fold,joint=True)
        baselines={}
        for method,grid in GRIDS.items():
            start=time.perf_counter();trace=[{'config':c,'folds':[]} for c in grid]
            for j,inner in enumerate(fit['training_splits']):
                ti,qi=np.array(inner['reference']),np.array(inner['query']); P,Q=preprocess_fit(X[tr][ti],X[tr][qi])
                for c in trace:
                    selection=SELECTORS[method](P,y[tr][ti],**c['config']); metrics=score(P,y[tr][ti],Q,y[tr][qi],selection['features'],5000+j)
                    c['folds'].append({'features':selection['features'],'metrics':metrics})
            for c in trace:
                c['utility']=float(np.mean([utility(r['metrics']) for r in c['folds']]))
                c['mean_size']=float(np.mean([len(r['features']) for r in c['folds']]))
            chosen=max(range(len(grid)),key=lambda j:(trace[j]['utility'],-trace[j]['mean_size'],-j))
            selection=SELECTORS[method](A,y[tr],**grid[chosen]); seconds=time.perf_counter()-start
            metrics=score(A,y[tr],B,y[te],selection['features'],144+fold)
            baselines[method]={'trace':trace,'chosen_index':chosen,'selection':selection,'metrics':metrics,'seconds':seconds}
        splits.append({'train_indices':tr.tolist(),'test_indices':te.tolist(),'trajectory':trajectory,'certified_trajectory':certified,
            'fit':fit,'proposed_metrics':proposed,'full_metrics':full,'proposed_seconds':elapsed,'baselines':baselines})
        print(name,fold,'frozen pipeline completed',flush=True)
    assert sources(name)==fingerprints
    out.write_text(json.dumps({'dataset':name,'d':X.shape[1],'n':len(X),'fingerprints':fingerprints,'splits':splits},indent=2),encoding='utf-8')


def analyze():
    rows=[]
    for name in SOURCES:
        p=json.loads((FOLDER/f'{name}.json').read_text());assert p['fingerprints']==sources(name)
        metrics={'proposed':[s['proposed_metrics'] for s in p['splits']], 'full':[s['full_metrics'] for s in p['splits']]}
        metrics.update({m:[s['baselines'][m]['metrics'] for s in p['splits']] for m in GRIDS})
        sizes={'proposed':[len(s['fit']['compact_selected']['features']) for s in p['splits']]}
        sizes.update({m:[len(s['baselines'][m]['selection']['features']) for s in p['splits']] for m in GRIDS})
        rows.append({'dataset':name,'utility':{m:float(np.mean([utility(v) for v in values])) for m,values in metrics.items()},
            'metrics':{m:{k:float(np.mean([v[k] for v in values])) for k in ('clean','noise01')} for m,values in metrics.items()},
            'removed':{m:1-float(np.mean(values))/p['d'] for m,values in sizes.items()},
            'seconds':{'proposed':float(np.mean([s['proposed_seconds'] for s in p['splits']])),
                **{m:float(np.mean([s['baselines'][m]['seconds'] for s in p['splits']])) for m in GRIDS}}})
    comparisons={}
    for m in GRIDS:
        delta=np.array([r['utility']['proposed']-r['utility'][m] for r in rows])
        comparisons[m]={'gain':float(delta.mean()),'wins':int(sum(delta>1e-12)),'losses':int(sum(delta < -1e-12)),
            'wilcoxon_raw':float(wilcoxon(delta).pvalue) if np.any(delta!=0) else 1.}
    running=0.
    for rank,m in enumerate(sorted(comparisons,key=lambda m:comparisons[m]['wilcoxon_raw'])):
        running=max(running,min(1.,(len(comparisons)-rank)*comparisons[m]['wilcoxon_raw']));comparisons[m]['wilcoxon_holm']=running
    result={'rows':rows,'comparisons':comparisons,'means':{m:float(np.mean([r['utility'][m] for r in rows])) for m in ('proposed','full',*GRIDS)},
        'removed':{m:float(np.mean([r['removed'][m] for r in rows])) for m in ('proposed',*GRIDS)},
        'scope':'frozen_local_newdata_assessment_eight_sampled_tasks_not_universal_superiority'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':
    DATA.mkdir(exist_ok=True);FOLDER.mkdir(exist_ok=True)
    for name in SOURCES:download(name)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run,n) for n in SOURCES]):future.result()
    analyze()
