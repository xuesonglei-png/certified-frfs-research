"""Frozen reference-change development test; no query labels enter search."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from certified_frfs.task_reference_geometry import TaskReferenceGeometryContext
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.orthogonal_forest_logistic import orthogonal_scores
from certified_frfs.forest_penalty_selection import balanced_accuracy
from review_box_iterations import paired

SOURCE = ROOT/'results/nested-geometry-forest'
FOLDER = ROOT/'results/task-reference-geometry-pilot'
METHODS = ('old','full','uniform_native','task_native','task_matched')


def run(name):
    threadpool_limits(1); source = source_verified(name); hashes = dict(source['fingerprints'])
    for p in (Path(__file__),ROOT/'research/task-reference-geometry-pilot-protocol.md',SOURCE/(name+'.json'),
        ROOT/'src/certified_frfs/task_reference_geometry.py',ROOT/'src/certified_frfs/forest_lipschitz_metric.py',
        ROOT/'scripts/run_nested_joint_path.py',ROOT/'scripts/audit_nested_geometry_forest.py',
        ROOT/'research/experiment-source/run_task_reference_geometry_pilot_v1.py'):
        hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    out = FOLDER/(name+'.json'); records = []
    if out.exists():
        saved = json.loads(out.read_text()); assert saved['fingerprints'] == hashes
        if saved['complete']: return
        records = saved['records']
    X,y = load_dataset(name); n = 0
    for i,s in enumerate(source['splits']):
        ri = np.array(s['reference_indices'])
        for j,inner in enumerate(s['inner']):
            n += 1
            if n <= len(records): continue
            a,b = np.array(inner['reference_indices']),np.array(inner['validation_indices'])
            A,B = preprocess_fit(X[ri[a]],X[ri[b]]); labels = y[ri[a]]; old_stage = inner['stages'][7]
            assert old_stage['C'] == old_stage['pair_penalty'] == 1.
            old = inner['models'][old_stage['policies']['0.8']['model_id']]
            full = inner['models'][old_stage['policies']['1.0']['model_id']]; target = len(old['features'])
            contexts = {family:TaskReferenceGeometryContext(X[ri[a]],labels,family=family) for family in ('uniform_unit','task_unit')}
            stages = {family:context.fit_policy(fraction=.8) for family,context in contexts.items()}
            for stage in stages.values(): assert stage['reference_teacher'] == full
            task = stages['task_unit']; point = next(p for p in reversed(task['path']['trace']) if len(p['features']) >= target)
            matched = contexts['task_unit'].classifier(point['features'],1.,1.)
            fits = {'old':old,'full':full,'uniform_native':stages['uniform_unit']['fit'],
                'task_native':task['fit'],'task_matched':matched}
            rng = np.random.default_rng(144+i); queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
            metrics = {}
            for m,fit in fits.items():
                values = [balanced_accuracy(orthogonal_scores(Q,fit),fit['classes'],y[ri[b]]) for Q in queries]
                metrics[m] = {'clean':values[0],'noise':float(np.mean(values[1:])),'removed':1-len(fit['features'])/X.shape[1]}
            records.append({'outer_index':i,'inner_index':j,'outer_reference_indices':ri.tolist(),
                'reference_indices':a.tolist(),'validation_indices':b.tolist(),'query_seed':144+i,
                'target_count':target,'count_exact':len(matched['features']) == target,'matched_candidate':point,
                'stages':stages,'fits':fits,'metrics':metrics})
            out.write_text(json.dumps({'dataset':name,'d':X.shape[1],'fingerprints':hashes,'records':records,
                'complete':len(records) == sum(len(s['inner']) for s in source['splits']),
                'scope':'changed_task_informed_unit_mass_reference_development_pilot_not_old_relation_or_population_guarantee'},separators=(',',':')),encoding='utf-8')
            print(name,i,j,'task reference',{m:round(v['clean'],5) for m,v in metrics.items()},
                'counts',{m:len(f['features']) for m,f in fits.items()},flush=True)


def summarize(names):
    rows = []; counts = {'records':0,'final_fit_caps':0,'count_mismatches':0,'matched_unchanged_supports':0,
        'task_native_unchanged_supports':0,'uniform_native_unchanged_supports':0,'task_uniform_same_supports':0,'teacher_fallbacks':0}
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete']
        counts['records'] += len(saved['records'])
        for r in saved['records']:
            counts['final_fit_caps'] += sum(not f['optimizer_converged'] for f in r['fits'].values())
            counts['count_mismatches'] += not r['count_exact']
            for m in ('task_native','uniform_native'): counts[m+'_unchanged_supports'] += r['fits'][m]['features'] == r['fits']['old']['features']
            counts['matched_unchanged_supports'] += r['fits']['task_matched']['features'] == r['fits']['old']['features']
            counts['task_uniform_same_supports'] += r['fits']['task_native']['features'] == r['fits']['uniform_native']['features']
            counts['teacher_fallbacks'] += r['stages']['task_unit']['sensitivity']['uniform_fallback']
            for stage in r['stages'].values(): assert stage['candidate']['certificate']['feasible']
        rows.append({'dataset':name,'methods':{m:{metric:float(np.mean([r['metrics'][m][metric] for r in saved['records']]))
            for metric in ('clean','noise','removed')} for m in METHODS}})
    assert counts['records'] == 128 and len(rows) == 15
    result = {'rows':rows,'counts':counts,'means':{m:{metric:float(np.mean([r['methods'][m][metric] for r in rows]))
        for metric in ('clean','noise','removed')} for m in METHODS},
        'contrasts':{metric:{m:paired([r['methods'][m][metric]-r['methods']['old'][metric] for r in rows])
            for m in METHODS[1:]} for metric in ('clean','noise')},
        'task_vs_uniform':{metric:paired([r['methods']['task_native'][metric]-r['methods']['uniform_native'][metric] for r in rows])
            for metric in ('clean','noise')},
        'scope':'all15_all128_changed_full_reference_fixed_config_exposed_development_no_modern_SOTA_claim'}
    result['gate_passed'] = (result['contrasts']['clean']['task_native']['mean_pp'] >= .5
        and result['contrasts']['clean']['task_native']['sign_flip_two_sided_p'] < .05
        and result['contrasts']['clean']['task_matched']['mean_pp'] >= .5
        and counts['count_mismatches'] == counts['final_fit_caps'] == 0)
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k != 'rows'},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args(); FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in ([args.only] if args.only else names)]): f.result()
    if not args.only: summarize(names)
