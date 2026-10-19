"""Fresh exact inner coverage selects all methods under the same clean band."""
import argparse
import copy
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from run_nested_geometry_forest import MODELS,GRIDS,PATHS,Registry,verified_native,holm
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.geometry_only_forest_stage import GeometryStageContext
from certified_frfs.robust_policy_choice import choose_robust_policy
from certified_frfs.forest_fuzzy_classifier import forest_box_margins
from certified_frfs.orthogonal_forest_logistic import orthogonal_scores
from certified_frfs.forest_penalty_selection import balanced_accuracy
from run_forest_robustness_curve import RADII,area,coverage
from review_box_iterations import paired

SOURCE = ROOT/'results/nested-geometry-forest'
FOLDER = ROOT/'results/nested-robust-policy'
METHODS = ('proposed','full','fgs2023','sfrip2025')
ALL_METHODS = METHODS+tuple('old_'+m for m in METHODS)
SELECTION_RADIUS = .03


def box_margins(B,truth,fit,eps):
    q = len(fit['classes']); edges = fit['edges']
    return forest_box_margins(B,truth,fit['coef'],fit['intercept'],fit['classes'],edges,
        np.array(fit['pair_coef']).reshape(q,len(edges),3,3),eps=eps)


def seed_context(raw,y,block):
    context = GeometryStageContext(raw,y)
    context.geometry = {tuple(g['features']):g for g in block['geometry']}
    context.trees = {tuple(t['features']):t for t in block['trees']}
    for fit in block['models']:
        gamma = fit['pair_penalty'] if fit['edges'] else None
        context.classifiers[(tuple(fit['features']),float(fit['C']),gamma)] = fit
    stages = block['stages']; stages = list(stages.values()) if isinstance(stages,dict) else stages
    for stage in stages:
        for policy in stage['policies'].values():
            fit = block['models'][policy['model_id']]
            context.classifiers[(tuple(fit['features']),stage['C'],stage['pair_penalty'])] = fit
    return context


def run(name):
    threadpool_limits(1); source = source_verified(name); native = verified_native(name,source)
    hashes = dict(source['fingerprints'])
    for p in (Path(__file__),ROOT/'research/nested-robust-policy-protocol.md',SOURCE/(name+'.json'),
        ROOT/'src/certified_frfs/robust_policy_choice.py',ROOT/'scripts/run_forest_robustness_curve.py',
        ROOT/'scripts/audit_nested_geometry_forest.py',ROOT/'scripts/run_nested_joint_path.py'):
        hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    out = FOLDER/(name+'.json'); records = []
    if out.exists():
        previous = json.loads(out.read_text()); assert previous['fingerprints'] == hashes
        if previous['complete']: return
        records = previous['splits']
    X,y = load_dataset(name)
    for i,split in enumerate(source['splits']):
        if i < len(records): continue
        ri,vi = np.array(split['reference_indices']),np.array(split['validation_indices'])
        traces = copy.deepcopy(split['trace']); inner_records = []
        for j,inner in enumerate(split['inner']):
            a,b = np.array(inner['reference_indices']),np.array(inner['validation_indices'])
            _,B = preprocess_fit(X[ri[a]],X[ri[b]]); truth = y[ri[b]]
            mids = sorted({row['folds'][j]['model_id'] for rows in traces.values() for row in rows}); scores = {}
            for mid in mids:
                fit = inner['models'][mid]; margins = box_margins(B,truth,fit,SELECTION_RADIUS)
                scores[mid] = {'model_id':mid,'margins':margins.tolist(),
                    'robust':coverage(margins > 1e-10,truth),
                    'clean':balanced_accuracy(orthogonal_scores(B,fit),fit['classes'],truth)}
            for rows in traces.values():
                for row in rows:
                    fold = row['folds'][j]; measured = scores[fold['model_id']]
                    np.testing.assert_allclose(fold['clean'],measured['clean'],atol=1e-12)
                    fold['robust'] = measured['robust']
            inner_records.append({'reference_indices':a.tolist(),'validation_indices':b.tolist(),'scores':list(scores.values())})
            print(name,i,'robust inner',j,'models',len(scores),flush=True)
        choices = {m:choose_robust_policy(rows) for m,rows in traces.items()}
        context = seed_context(X[ri],y[ri],split); A,B = preprocess_fit(X[ri],X[vi]); registry = Registry(context)
        results = {}; stages = {}; paths = {}
        # Complete every new selection/fit before using any outer query labels.
        for m in METHODS:
            ci = choices[m]['chosen_index']; row = traces[m][ci]; cfg = MODELS[row['model_index']]
            old_result = split['results'][m]; reused = ci == old_result['chosen_index']
            if reused:
                fit = split['models'][old_result['model_id']]; fs = fit['features']; geometry_id = old_result['geometry_id']
                if m == 'proposed': stages[m] = copy.deepcopy(split['stages'][m])
                if m in GRIDS: paths[m] = split['native_paths'][m]
            elif m == 'proposed':
                stage = context.fit_stage(**cfg); stages[m] = registry.stage(stage)
                point = stage['policies'][str(row['fraction'])]; fit = point['fit']; fs = point['candidate']['features']
                geometry_id = stages[m]['policies'][str(row['fraction'])]['geometry_id']
            elif m == 'full':
                fs = list(range(X.shape[1])); fit = context.classifier(fs,**cfg); geometry_id = None
            else:
                config = GRIDS[m][row['selector_index']]; key = json.dumps(config,sort_keys=True)
                cache = native['splits'][i]['outer_paths'][m]
                path = cache[key] if key in cache else PATHS[m](A,y[ri],**config)
                paths[m] = {'config':config,'path':path,'cached':key in cache}
                fs = sorted(path['native']['features']); fit = context.classifier(fs,**cfg); geometry_id = None
            results[m] = {'chosen_index':ci,'features':fs,'model_index':row['model_index'],
                'selector_index':row['selector_index'],'model_id':registry.capture(fit),
                'geometry_id':geometry_id,'outer_fit_reused':reused}
        # Reused source stage indices belong to original source block, never
        # to this registry. Mark that identity explicitly instead of remapping.
        proposed_reused = results['proposed']['outer_fit_reused']
        if proposed_reused: stages.pop('proposed')
        for m in METHODS:
            fit = split['models'][split['results'][m]['model_id']]
            results['old_'+m] = {'features':fit['features'],'model_id':registry.capture(fit),'outer_fit_reused':True}
        rng = np.random.default_rng(144+i); queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        for m,row in results.items():
            fit = registry.models[row['model_id']]; truth = y[vi]
            noise = [balanced_accuracy(orthogonal_scores(Q,fit),fit['classes'],truth) for Q in queries]
            curves = [box_margins(B,truth,fit,eps) for eps in RADII]
            row['margins'] = [v.tolist() for v in curves]
            curve = [coverage(v > 1e-10,truth) for v in curves]
            row['metrics'] = {'clean':noise[0],'noise':float(np.mean(noise[1:])),
                'robust':curve[2],'area':area(curve),'removed':1-len(fit['features'])/X.shape[1],'curve':curve}
        records.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'inner':inner_records,
            'trace':traces,'choices':choices,'results':results,'stages':stages,'native_paths':paths,
            'proposed_stage_reused':proposed_reused,'query_seed':144+i,**registry.export()})
        out.write_text(json.dumps({'dataset':name,'d':X.shape[1],'fingerprints':hashes,'splits':records,
            'complete':len(records) == len(source['splits']),'selection_radius':SELECTION_RADIUS,
            'radii':RADII,'scope':'common_nested_exact_robust_selection_numeric_box_exposed_development'},separators=(',',':')),encoding='utf-8')
        print(name,i,'robust selected',{m:{k:round(v,5) for k,v in row['metrics'].items() if k in ('clean','robust','removed')} for m,row in results.items()},flush=True)


def summarize(names):
    rows = []; counts = {'outer_splits':0,'inner_splits':0,'inner_score_models':0,'optimizer_caps':0,
        'selected_proposed_changed':0,'all_selected_changed':0}
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete']
        for s in saved['splits']:
            counts['outer_splits'] += 1; counts['inner_splits'] += len(s['inner'])
            counts['inner_score_models'] += sum(len(b['scores']) for b in s['inner'])
            counts['optimizer_caps'] += sum(not f['optimizer_converged'] for f in s['models'])
            counts['selected_proposed_changed'] += not s['results']['proposed']['outer_fit_reused']
            counts['all_selected_changed'] += sum(not s['results'][m]['outer_fit_reused'] for m in METHODS)
        rows.append({'dataset':name,'methods':{m:{metric:float(np.mean([s['results'][m]['metrics'][metric] for s in saved['splits']]))
            for metric in ('clean','noise','robust','area','removed')} for m in ALL_METHODS}})
    assert len(rows) == 15 and counts['outer_splits'] == 44 and counts['inner_splits'] == 128
    result = {'rows':rows,'counts':counts,
        'means':{m:{metric:float(np.mean([r['methods'][m][metric] for r in rows])) for metric in ('clean','noise','robust','area','removed')} for m in ALL_METHODS},
        'selection_effects':{m:{metric:paired([r['methods'][m][metric]-r['methods']['old_'+m][metric] for r in rows])
            for metric in ('clean','noise','robust','area')} for m in METHODS},
        'new_method_contrasts':{metric:{m:paired([r['methods']['proposed'][metric]-r['methods'][m][metric] for r in rows])
            for m in METHODS[1:]} for metric in ('clean','robust','area')},
        'scope':'all15_all44_all128_common_train_only_robust_selection_native_budgets_exposed_development'}
    result['descriptive_holm'] = {metric:holm({m:v['sign_flip_two_sided_p'] for m,v in result['new_method_contrasts'][metric].items()}) for metric in ('clean','robust','area')}
    effect = result['selection_effects']['proposed']
    result['gate_passed'] = (effect['robust']['mean_pp'] >= 1 and effect['robust']['sign_flip_two_sided_p'] < .05
        and effect['clean']['mean_pp'] >= -.5 and counts['optimizer_caps'] == 0)
    result['goal_achieved'] = False
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k != 'rows'},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args(); FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in ([args.only] if args.only else names)]): f.result()
    if not args.only: summarize(names)
