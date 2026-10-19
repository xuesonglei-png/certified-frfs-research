"""Frozen five-knot capacity/search test with actual-count modern controls."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from run_nested_geometry_forest import GRIDS,verified_native,holm
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.spline_geometry_stage import SplineGeometryStageContext,KNOTS
from certified_frfs.spline_forest import fit_spline_forest,spline_scores,spline_box_margins
from certified_frfs.orthogonal_forest_logistic import fit_orthogonal_forest
from certified_frfs.fuzzy_interaction_tree import interaction_tree
from certified_frfs.forest_penalty_selection import balanced_accuracy
from run_forest_robustness_curve import coverage
from review_box_iterations import paired

SOURCE = ROOT/'results/nested-geometry-forest'
FOLDER = ROOT/'results/spline-capacity-pilot'
SELECTOR_INDICES = {'fgs2023':4,'sfrip2025':3}
METHODS = ('old3','full3','full5_fixed3','full5','old_support5','proposed5','same_new_support3',
    'fgs_native5','sfrip_native5','fgs_matched5','sfrip_matched5')


def as_spline3(fit):
    return {**fit,'knots':np.tile([0.,.5,1.],(len(fit['coef'][0]),1)).tolist()}


def run(name):
    threadpool_limits(1); source = source_verified(name); native = verified_native(name,source)
    hashes = dict(source['fingerprints'])
    for p in (Path(__file__),ROOT/'research/spline-capacity-pilot-protocol.md',SOURCE/(name+'.json'),
        ROOT/'src/certified_frfs/spline_forest.py',ROOT/'src/certified_frfs/spline_geometry_stage.py',
        ROOT/'scripts/run_nested_joint_path.py',ROOT/'scripts/audit_nested_geometry_forest.py'):
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
            A,B = preprocess_fit(X[ri[a]],X[ri[b]]); labels = y[ri[a]]; d = A.shape[1]
            oldstage = inner['stages'][7]; assert oldstage['C'] == oldstage['pair_penalty'] == 1.
            old = inner['models'][oldstage['policies']['0.8']['model_id']]
            full = inner['models'][oldstage['policies']['1.0']['model_id']]
            context = SplineGeometryStageContext(X[ri[a]],labels)
            context.geometry = {tuple(g['features']):g for g in inner['geometry']}
            stage = context.fit_policy(fraction=.8); target = len(stage['fit']['features'])
            fullfixed = fit_spline_forest(A,labels,list(range(d)),full['edges'],knots=KNOTS,C=1.,pair_penalty=1.)
            fs = stage['fit']['features']; tree3 = interaction_tree(A[:,fs],labels)
            same3 = fit_orthogonal_forest(A,labels,fs,[sorted([fs[u],fs[v]]) for u,v in tree3['edges']],C=1.,pair_penalty=1.)
            fits = {'old3':as_spline3(old),'full3':as_spline3(full),'full5_fixed3':fullfixed,
                'full5':stage['teacher'],'old_support5':context.classifier(old['features']),
                'proposed5':stage['fit'],'same_new_support3':as_spline3(same3)}
            paths = {}
            for method,ci in SELECTOR_INDICES.items():
                path = native['splits'][i]['inner'][j]['paths'][method][ci]
                assert sorted(path['ranking']) == list(range(d))
                native_features = sorted(path['native']['features']); matched = sorted(path['ranking'][:target])
                prefix = 'fgs' if method == 'fgs2023' else 'sfrip'
                fits[prefix+'_native5'] = context.classifier(native_features)
                fits[prefix+'_matched5'] = context.classifier(matched)
                paths[method] = {'selector_index':ci,'config':GRIDS[method][ci],
                    'ranking':path['ranking'],'native_features':native_features,'matched_features':matched}
            assert set(fits) == set(METHODS)
            # All fits, certificates and native/matched supports exist before
            # the query labels enter any metric computation.
            rng = np.random.default_rng(144+i); queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
            metrics = {}; boxes = {}
            for method,fit in fits.items():
                values = [balanced_accuracy(spline_scores(Q,fit),fit['classes'],y[ri[b]]) for Q in queries]
                box = spline_box_margins(B,y[ri[b]],fit,eps=.01,witnesses=True)
                boxes[method] = {'margins':box['margins'].tolist(),'rivals':box['rivals'].tolist(),'witness_raw':box['witness_raw'].tolist()}
                metrics[method] = {'clean':values[0],'noise':float(np.mean(values[1:])),
                    'robust':coverage(box['margins'] > 1e-10,y[ri[b]]),'removed':1-len(fit['features'])/d}
            counts_exact = all(len(fits[m]['features']) == target for m in ('fgs_matched5','sfrip_matched5'))
            records.append({'outer_index':i,'inner_index':j,'outer_reference_indices':ri.tolist(),
                'reference_indices':a.tolist(),'validation_indices':b.tolist(),'query_seed':144+i,
                'target_count':target,'counts_exact':counts_exact,'stage':stage,'selector_paths':paths,
                'fits':fits,'boxes':boxes,'metrics':metrics})
            out.write_text(json.dumps({'dataset':name,'d':d,'fingerprints':hashes,'records':records,
                'complete':len(records) == sum(len(s['inner']) for s in source['splits']),
                'scope':'fixed_five_knot_development_capacity_ablation_original_FRS_search_actual_count_controls'},separators=(',',':')),encoding='utf-8')
            print(name,i,j,'five-knot capacity',{m:round(v['clean'],5) for m,v in metrics.items()},
                'counts',{m:len(f['features']) for m,f in fits.items()},flush=True)


def summarize(names):
    rows = []; counts = {'records':0,'count_mismatches':0,'optimizer_caps':0,'old_new_supports_equal':0}
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete']
        for r in saved['records']:
            counts['records'] += 1; counts['count_mismatches'] += not r['counts_exact']
            counts['optimizer_caps'] += sum(not f['optimizer_converged'] for f in r['fits'].values())
            counts['old_new_supports_equal'] += r['fits']['old3']['features'] == r['fits']['proposed5']['features']
            assert r['stage']['candidate']['certificate']['feasible']
        rows.append({'dataset':name,'methods':{m:{metric:float(np.mean([r['metrics'][m][metric] for r in saved['records']]))
            for metric in ('clean','noise','robust','removed')} for m in METHODS}})
    assert len(rows) == 15 and counts['records'] == 128
    compare = lambda first,second,metric:paired([r['methods'][first][metric]-r['methods'][second][metric] for r in rows])
    result = {'rows':rows,'counts':counts,
        'means':{m:{metric:float(np.mean([r['methods'][m][metric] for r in rows])) for metric in ('clean','noise','robust','removed')} for m in METHODS},
        'ablations':{metric:{'basis_only':compare('full5_fixed3','full3',metric),
            'five_state_tree':compare('full5','full5_fixed3',metric),
            'new_support_capacity':compare('proposed5','same_new_support3',metric),
            'path_change':compare('proposed5','old_support5',metric),
            'new_vs_old':compare('proposed5','old3',metric)} for metric in ('clean','noise','robust')},
        'method_contrasts':{metric:{m:compare('proposed5',m,metric) for m in ('full5','fgs_native5','sfrip_native5','fgs_matched5','sfrip_matched5')}
            for metric in ('clean','noise','robust')},
        'scope':'fixed_config_all15_all128_development_basis_tree_path_ablations_native_and_actual_count_controls'}
    clean = result['method_contrasts']['clean']
    result['matched_holm'] = holm({m:clean[m]['sign_flip_two_sided_p'] for m in ('full5','fgs_matched5','sfrip_matched5')})
    basis = result['ablations']['clean']['basis_only']
    result['representation_gate_passed'] = basis['mean_pp'] >= .5 and basis['sign_flip_two_sided_p'] < .05
    result['method_gate_passed'] = (all(clean[m]['mean_pp'] >= .5 and result['matched_holm'][m] < .05 for m in ('fgs_matched5','sfrip_matched5'))
        and clean['full5']['mean_pp'] >= 0 and counts['count_mismatches'] == counts['optimizer_caps'] == 0)
    result['goal_achieved'] = False
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k != 'rows'},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args(); FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in ([args.only] if args.only else names)]): f.result()
    if not args.only: summarize(names)
