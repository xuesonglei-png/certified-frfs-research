"""All references, internal-only nonlinear teacher criterion and fixed counts."""
import json,time
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from certified_frfs.datasets import load_dataset
from certified_frfs.geometry_only_forest_stage import GeometryStageContext
from certified_frfs.crossfit_kernel_projection import CrossfitKernelProjection,certified_kernel_swaps

SOURCE = ROOT/'results/nested-geometry-forest'; FOLDER = ROOT/'results/crossfit-kernel-screen'

def run(name):
    threadpool_limits(1); source = source_verified(name); X,y = load_dataset(name)
    hashes = dict(source['fingerprints'])
    for p in (Path(__file__),SOURCE/(name+'.json'),ROOT/'src/certified_frfs/crossfit_kernel_projection.py',
        ROOT/'research/crossfit-kernel-screen-protocol.md',ROOT/'src/certified_frfs/nested_teacher_regularized.py'):
        hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    artifact = FOLDER/(name+'.json'); rows = []
    if artifact.exists():
        prior = json.loads(artifact.read_text()); assert prior['fingerprints'] == hashes
        if prior['complete']: return prior
        rows = prior['rows']
    index = 0
    for i,s in enumerate(source['splits']):
        ri = np.array(s['reference_indices'])
        for j,inner in enumerate(s['inner']):
            index += 1
            if index <= len(rows): continue
            a = np.array(inner['reference_indices']); raw,labels = X[ri[a]],y[ri[a]]; start = time.perf_counter()
            context = GeometryStageContext(raw,labels); context.geometry = {tuple(g['features']):g for g in inner['geometry']}
            initial = inner['models'][inner['stages'][7]['policies']['0.8']['model_id']]['features']
            risk = CrossfitKernelProjection(raw,labels); path = certified_kernel_swaps(initial,risk,context.candidate)
            folds = [{key:fold[key].tolist() if isinstance(fold[key],np.ndarray) else fold[key]
                for key in ('reference_indices','validation_indices','gamma','center','original_scores','balance')}
                for fold in risk.folds]
            evaluations = [{'features':list(fs),'risk':value} for fs,value in sorted(risk.cache.items())]
            rows.append({'outer':i,'inner':j,'outer_reference_indices':ri.tolist(),'reference_indices':a.tolist(),
                'folds':folds,'path':path,'risk_evaluations':evaluations,'seconds':time.perf_counter()-start})
            result = {'dataset':name,'fingerprints':hashes,'rows':rows,'complete':len(rows) == sum(len(s['inner']) for s in source['splits']),
                'query_labels_used':False,'scope':'training_adaptive_kernel_task_screen_not_prediction_superiority'}
            artifact.write_text(json.dumps(result,separators=(',',':')),encoding='utf-8')
            print(name,i,j,'kernel crossfit improvement_pp',100*(path['initial_risk']['balanced_error']-path['final_risk']['balanced_error']),
                'changed',path['features'] != initial,'loss_evaluations',len(evaluations),flush=True)
    return result

if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True); names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = [f.result() for f in as_completed([pool.submit(run,n) for n in names])]
    counts = Counter(); datasets = []
    for saved in rows:
        assert saved['complete'] and not saved['query_labels_used']; gains = []
        for r in saved['rows']:
            path = r['path']; gains.append(path['initial_risk']['balanced_error']-path['final_risk']['balanced_error'])
            counts['references'] += 1; counts['error_improves'] += round(path['final_risk']['balanced_error'],12) < round(path['initial_risk']['balanced_error'],12)
            counts['changed_supports'] += path['features'] != path['initial_features']; counts['count_mismatches'] += len(path['features']) != len(path['initial_features'])
            counts['loss_evaluations'] += len(r['risk_evaluations']); counts['attempts'] += len(path['attempts']); counts['rejections'] += sum(not a['accepted'] for a in path['attempts'])
            assert path['candidate']['certificate']['feasible']
        datasets.append({'dataset':saved['dataset'],'mean_internal_error_gain_pp':100*float(np.mean(gains))})
    assert counts['references'] == 128 and len(datasets) == 15
    mean = float(np.mean([r['mean_internal_error_gain_pp'] for r in datasets]))
    result = {'counts':dict(counts),'rows':datasets,'mean_internal_error_gain_pp':mean,
        'numeric_resource_gate':bool(mean >= .5 and counts['error_improves'] >= 20 and counts['count_mismatches'] == 0),
        'independent_audit_required':True,'goal_achieved':False,'query_labels_used':False,
        'scope':'optimistically_adaptive_training_screen_not_test_accuracy_or_SOTA'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)
