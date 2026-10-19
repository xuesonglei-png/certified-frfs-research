"""Reference-internal full pipeline tuning of invariant score regularization."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from run_nested_geometry_forest import GRIDS,verified_native,holm
from run_spline_capacity_pilot import as_spline3
from certified_frfs.functional_pipeline_cv import METHODS,CONFIGS,tune_functional_pipelines,fit_candidate
from certified_frfs.geometry_only_forest_stage import GeometryStageContext
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.spline_forest import spline_scores,spline_box_margins
from certified_frfs.forest_penalty_selection import balanced_accuracy
from run_forest_robustness_curve import coverage
from review_box_iterations import paired

SOURCE = ROOT/'results/nested-geometry-forest'
FOLDER = ROOT/'results/functional-normalization-pilot'
ALL_METHODS = METHODS+('old3','full3')


def run(name):
    threadpool_limits(1); source = source_verified(name); native = verified_native(name,source)
    hashes = dict(source['fingerprints'])
    for p in (Path(__file__),ROOT/'research/functional-normalization-pilot-protocol.md',SOURCE/(name+'.json'),
        ROOT/'src/certified_frfs/functional_spline_forest.py',ROOT/'src/certified_frfs/functional_pipeline_cv.py',
        ROOT/'src/certified_frfs/spline_forest.py',ROOT/'src/certified_frfs/nested_teacher_regularized.py',
        ROOT/'scripts/run_spline_capacity_pilot.py',ROOT/'scripts/run_nested_joint_path.py',
        ROOT/'scripts/audit_nested_geometry_forest.py',ROOT/'research/functional-partition-refinement-lemma.md'):
        hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    out = FOLDER/(name+'.json'); records = []
    if out.exists():
        previous = json.loads(out.read_text()); assert previous['fingerprints'] == hashes
        if previous['complete']: return
        records = previous['records']
    X,y = load_dataset(name); n = 0
    for i,s in enumerate(source['splits']):
        ri = np.array(s['reference_indices'])
        for j,inner in enumerate(s['inner']):
            n += 1
            if n <= len(records): continue
            a,b = np.array(inner['reference_indices']),np.array(inner['validation_indices'])
            raw,labels = X[ri[a]],y[ri[a]]; A,B = preprocess_fit(raw,X[ri[b]])
            tuning = tune_functional_pipelines(raw,labels)
            oldstage = inner['stages'][7]; assert oldstage['C'] == oldstage['pair_penalty'] == 1.
            old = inner['models'][oldstage['policies']['0.8']['model_id']]
            full = inner['models'][oldstage['policies']['1.0']['model_id']]
            candidate = inner['geometry'][oldstage['policies']['0.8']['geometry_id']]
            assert candidate['features'] == old['features'] and candidate['certificate']['feasible']
            target = len(old['features']); d = A.shape[1]
            supports = {'certified':old['features'],'full':list(range(d))}; selectors = {}
            for method,ci,prefix in [('fgs2023',4,'fgs'),('sfrip2025',3,'sfrip')]:
                path = native['splits'][i]['inner'][j]['paths'][method][ci]
                supports[prefix+'_native'] = sorted(path['native']['features'])
                supports[prefix+'_matched'] = sorted(path['ranking'][:target])
                selectors[method] = {'selector_index':ci,'config':GRIDS[method][ci],'path':path}
            context = GeometryStageContext(raw,labels)
            context.trees = {tuple(t['features']):t for t in inner['trees']}
            fits = {}; choices = {}
            # Each hyperparameter choice came only from complete refolded
            # internal pipelines. All parent fits precede query evaluation.
            for method in METHODS:
                choice = tuning['choices'][method]; config = CONFIGS[choice['config_index']]
                fs = supports[method]; edges = [] if not fs else context.tree(fs)['edges']
                fits[method] = fit_candidate(A,labels,fs,edges,config); choices[method] = config
            fits['old3'],fits['full3'] = as_spline3(old),as_spline3(full)
            counts_exact = all(len(fits[m]['features']) == target for m in ('fgs_matched','sfrip_matched'))
            rng = np.random.default_rng(144+i); queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
            metrics,boxes = {},{}
            for method,fit in fits.items():
                values = [balanced_accuracy(spline_scores(Q,fit),fit['classes'],y[ri[b]]) for Q in queries]
                box = spline_box_margins(B,y[ri[b]],fit,eps=.01,witnesses=True)
                boxes[method] = {'margins':box['margins'].tolist(),'rivals':box['rivals'].tolist(),'witness_raw':box['witness_raw'].tolist()}
                metrics[method] = {'clean':values[0],'noise':float(np.mean(values[1:])),
                    'robust':coverage(box['margins'] > 1e-10,y[ri[b]]),'removed':1-len(fit['features'])/d}
            records.append({'outer_index':i,'inner_index':j,'outer_reference_indices':ri.tolist(),
                'reference_indices':a.tolist(),'validation_indices':b.tolist(),'query_seed':144+i,
                'target_count':target,'counts_exact':counts_exact,'candidate':candidate,'supports':supports,
                'selector_paths':selectors,'tuning':tuning,'selected_configs':choices,'fits':fits,'boxes':boxes,'metrics':metrics})
            out.write_text(json.dumps({'dataset':name,'d':d,'fingerprints':hashes,'records':records,
                'complete':len(records) == sum(len(s['inner']) for s in source['splits']),
                'scope':'functional_normalization_common_classifier_full_train_internal_pipeline_tuning_development'},separators=(',',':')),encoding='utf-8')
            print(name,i,j,'functional CV',{m:round(v['clean'],5) for m,v in metrics.items()},
                'configs',{m:tuning['choices'][m]['config_index'] for m in METHODS},flush=True)


def summarize(names):
    rows = []; counts = {'records':0,'count_mismatches':0,'parent_optimizer_caps':0,'child_optimizer_caps':0,'child_models':0,'child_folds':0}
    choices = {m:{'three_knots':0,'five_knots':0,'additive':0,'interaction':0} for m in METHODS}
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete']
        for r in saved['records']:
            counts['records'] += 1; counts['count_mismatches'] += not r['counts_exact']
            counts['parent_optimizer_caps'] += sum(not f['optimizer_converged'] for f in r['fits'].values())
            counts['child_folds'] += len(r['tuning']['folds'])
            for child in r['tuning']['folds']:
                counts['child_models'] += len(child['models']); counts['child_optimizer_caps'] += sum(not f['fit']['optimizer_converged'] for f in child['models'])
                assert child['support_block']['candidate']['certificate']['feasible']
            assert r['candidate']['certificate']['feasible']
            for m,cfg in r['selected_configs'].items():
                choices[m]['three_knots' if cfg['k'] == 3 else 'five_knots'] += 1
                choices[m]['additive' if cfg['pair_penalty'] is None else 'interaction'] += 1
        rows.append({'dataset':name,'methods':{m:{metric:float(np.mean([r['metrics'][m][metric] for r in saved['records']]))
            for metric in ('clean','noise','robust','removed')} for m in ALL_METHODS}})
    assert counts['records'] == 128 and len(rows) == 15
    result = {'rows':rows,'counts':counts,'choices':choices,
        'means':{m:{metric:float(np.mean([r['methods'][m][metric] for r in rows])) for metric in ('clean','noise','robust','removed')} for m in ALL_METHODS},
        'contrasts':{metric:{m:paired([r['methods']['certified'][metric]-r['methods'][m][metric] for r in rows]) for m in ALL_METHODS[1:]}
            for metric in ('clean','noise','robust')},
        'scope':'all15_all128_common_functional_capacity_refolded_training_CV_fixed_recent_selector_settings_exposed_development'}
    clean = result['contrasts']['clean']; result['matched_holm'] = holm({m:clean[m]['sign_flip_two_sided_p'] for m in ('full','fgs_matched','sfrip_matched')})
    result['method_gate_passed'] = (all(clean[m]['mean_pp'] >= .5 and result['matched_holm'][m] < .05 for m in ('fgs_matched','sfrip_matched'))
        and clean['full']['mean_pp'] >= 0 and counts['count_mismatches'] == counts['parent_optimizer_caps'] == counts['child_optimizer_caps'] == 0)
    result['goal_achieved'] = False
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k != 'rows'},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args(); FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in ([args.only] if args.only else names)]): f.result()
    if not args.only: summarize(names)
