"""Complete ALL15 training certificates at existing audited FGS120 weights."""
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from mixture_dependency_certificate_v1 import mixture_dependency_certificate
from run_mixture_enemy_diagnostic_v1 import digest,ROOT
from run_robustaware_projection_v1 import DATASETS,BASE

OUT=ROOT/'results/mixture-dependency-development-v1'
FREEZE=ROOT/'research/mixture-dependency-development-v1-source-freeze.json'


def main():
    threadpool_limits(1);OUT.mkdir(exist_ok=True)
    files=[Path(__file__),ROOT/'scripts/mixture_dependency_certificate_v1.py',ROOT/'scripts/mixture_enemy_envelope_v1.py',
        ROOT/'tests/test_mixture_dependency_certificate_v1.py',ROOT/'tests/test_mixture_enemy_envelope_v1.py']
    files+=sorted((ROOT/'src/certified_frfs').glob('*.py'))
    fingerprints={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    if not FREEZE.exists():FREEZE.write_text(json.dumps({'created_unix':time.time(),'fingerprints':fingerprints,
        'datasets':list(DATASETS),'pair_budget':2,'scope':'complete_training_tables_not_prediction'},indent=2),encoding='utf-8')
    assert json.loads(FREEZE.read_text())['fingerprints']==fingerprints
    summaries=[]
    for name in DATASETS:
        source=json.loads((BASE/(name+'.json')).read_text());X,y=load_dataset(name);ri=np.asarray(source['reference_indices'])
        A,_=preprocess_fit(X[ri],X[ri]);labels=y[ri]
        metric=source['outer']['results']['fgs_weighted120_atomic_count']['metric'];full=np.ones(A.shape[1])/np.sqrt(A.shape[1])
        old=json.loads((ROOT/'results/robustaware-projection-v1'/(name+'.json')).read_text())['fit']['anchor_guard']
        start=time.perf_counter()
        result=mixture_dependency_certificate(A,labels,metric['features'],full,metric['weights'],pair_budget=2)
        dep=result['upper_bound']<=.1 and max(result.get('class_upper_bounds',[0.]))<=.1
        accepted=bool(old['certificate']['majority_feasible'] and dep)
        row={'dataset':name,'reference_indices':ri.tolist(),'source_sha256':digest(BASE/(name+'.json')),
            'features':metric['features'],'weights':metric['weights'],'certificate':result,
            'original_guard':old,'mixture_dependency_feasible':bool(dep),'mixture_joint_feasible':accepted,
            'freeze_sha256':digest(FREEZE),'heldout_labels_used':False,'classification_scored':False,
            'seconds':time.perf_counter()-start,'goal_achieved':False}
        (OUT/(name+'.json')).write_text(json.dumps(row,indent=2),encoding='utf-8')
        summary={'dataset':name,'LPs':result['LPs'],'old_global':result['original_upper_bound'],
            'new_global':result['upper_bound'],'old_joint_feasible':old['accepted'],'new_joint_feasible':accepted,
            'old_max_class':old['class_certificate']['max_upper_bound'],
            'new_max_class':max(result.get('class_upper_bounds',[0.])),'seconds':row['seconds']}
        summaries.append(summary);print(json.dumps(summary),flush=True)
    result={'rows':summaries,'datasets':15,'LPs':sum(r['LPs'] for r in summaries),
        'old_joint_feasible':sum(r['old_joint_feasible'] for r in summaries),
        'new_joint_feasible':sum(r['new_joint_feasible'] for r in summaries),
        'mean_global_tightening':float(np.mean([r['old_global']-r['new_global'] for r in summaries])),
        'goal_achieved':False,'classification_scored':False,'independent_audit_required':True}
    (OUT/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':main()
