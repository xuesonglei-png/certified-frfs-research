"""Freeze eight-seed extension of the audited historical iteration-5 method."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/iteration5-reproduction-v1'
DEST=ROOT/'output/iteration5-eight-seeds-v1'
SEEDS=[101,202,303,404,505,606,707,808]


def write_json(path,obj):
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False)


def main():
    if DEST.exists():raise FileExistsError(DEST)
    audit=json.loads((BASE/'runs/rerun-001/audit.json').read_text())
    assert audit['passed']
    manifest=json.loads((BASE/'package-sha256.json').read_text())
    assert all(hashlib.sha256((BASE/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
    DEST.mkdir()
    for name in ['src','scripts','data','reference','tests']:
        shutil.copytree(BASE/name,DEST/name,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    for name in ['source-audit.json','environment.json','requirements.txt','pytest.ini','original-results-sha256.json']:
        shutil.copy2(BASE/name,DEST/name)
    shutil.copy2(BASE/'reproduce.py',DEST/'engine.py')
    shutil.copy2(BASE/'audit_reproduction.py',DEST/'historical_comparison.py')
    shutil.copy2(BASE/'runs/rerun-001/audit.json',DEST/'reference/base-reproduction-audit.json')
    shutil.copy2(ROOT/'scripts/run_iteration5_eight_seeds.py',DEST/'run_experiment.py')
    sys.path.insert(0,str(DEST/'src'))
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from certified_frfs.datasets import load_dataset,available_datasets
    datasets=available_datasets();splits=[]
    for seed in SEEDS:
        for name in datasets:
            X,y=load_dataset(name)
            for fold,(tr,te) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed).split(X,y)):
                inner=StratifiedKFold(min(3,np.unique(y[tr],return_counts=True)[1].min()),shuffle=True,random_state=seed+1)
                s={'dataset':name,'seed':seed,'fold':fold,'train_indices':tr.tolist(),'test_indices':te.tolist(),
                   'inner_indices_relative_to_train':[{'train':a.tolist(),'validation':b.tolist()} for a,b in inner.split(X[tr],y[tr])]}
                if seed==101:
                    old=json.loads((DEST/'reference/results'/f'{name}-s101-f{fold}.json').read_text())
                    assert old['train_indices']==s['train_indices'] and old['test_indices']==s['test_indices']
                splits.append(s)
    assert len(splits)==336 and len(datasets)==14
    write_json(DEST/'splits.json',splits)
    write_json(DEST/'protocol.json',{
        'frozen_unix':time.time(),'seeds':SEEDS,'datasets':datasets,'outer_folds':3,
        'inner_folds':3,'expected_splits':336,'expected_method_records':2688,
        'methods':['all','decision_margin','decision_nominal_margin','sfrip','mi','relief','mrmr','maf_instance'],
        'engine_sha256':hashlib.sha256((DEST/'engine.py').read_bytes()).hexdigest(),
        'base_package_manifest_sha256':hashlib.sha256((BASE/'package-sha256.json').read_bytes()).hexdigest(),
        'primary_metrics':['clean_balanced_accuracy','removed_fraction'],
        'secondary_metrics':['clean_macro_f1','noise01_BA','noise05_BA','balanced_training_coverage','minimum_class_training_coverage'],
        'aggregation':'mean over folds within seed/dataset, then seeds within dataset, then equal-weight datasets; SD across seeds reported separately',
        'accuracy_comparisons':'decision_margin vs seven comparators; paired two-sided Wilcoxon over 14 dataset averages; Holm correction within these seven comparisons',
        'descriptive_failures':'report all dataset deltas and highlight dataset mean BA loss >= 0.02 vs full; not a success gate',
        'seed_choice':'fixed before observing new results; include historical seed 101, recomputed, never suppress any seed',
        'scope':'repeated development stability evaluation, not independent confirmation or noninferiority test',
        'runtime_scope':'concurrently measured elapsed times are diagnostics, not fair efficiency evidence',
        'algorithm_changes':'none; byte-identical engine and method modules from audited bundle',
    })
    write_json(DEST/'frozen-inputs-sha256.json',{
        p.relative_to(DEST).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(DEST.rglob('*')) if p.is_file()})
    print(json.dumps({'directory':str(DEST),'splits':len(splits),'seeds':SEEDS},ensure_ascii=False))


if __name__=='__main__':main()
