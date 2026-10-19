"""Training-marginal reconstruction; unchanged certificates and actual count."""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from run_nested_geometry_forest import GRIDS,verified_native,holm
from run_spline_capacity_pilot import as_spline3
from certified_frfs.empirical_product_reduction import marginal_product_anova,feasible_count_refinement
from certified_frfs.forest_anova_reduction import canonical_raw_anova,functional_score_energies
from certified_frfs.geometry_only_forest_stage import GeometryStageContext
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.spline_forest import spline_scores,spline_box_margins
from certified_frfs.forest_penalty_selection import balanced_accuracy
from run_forest_robustness_curve import coverage
from review_box_iterations import paired

SOURCE = ROOT/'results/nested-geometry-forest'
FOLDER = ROOT/'results/empirical-product-fixed-count'
METHODS = ('old','full','empirical_refit','uniform_refit','fgs_native','sfrip_native','fgs_matched','sfrip_matched')

def run(name):
    threadpool_limits(1); source = source_verified(name); native = verified_native(name,source)
    hashes = dict(source['fingerprints'])
    for p in (Path(__file__),ROOT/'research/empirical-product-fixed-count-protocol.md',SOURCE/(name+'.json'),
        ROOT/'src/certified_frfs/empirical_product_reduction.py',ROOT/'src/certified_frfs/forest_anova_reduction.py',
        ROOT/'src/certified_frfs/functional_spline_forest.py',ROOT/'src/certified_frfs/spline_forest.py',
        ROOT/'scripts/run_nested_joint_path.py',ROOT/'scripts/run_spline_capacity_pilot.py'):
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
            stage = inner['stages'][7]; assert stage['C'] == stage['pair_penalty'] == 1.
            old = inner['models'][stage['policies']['0.8']['model_id']]; full = inner['models'][stage['policies']['1.0']['model_id']]
            context = GeometryStageContext(X[ri[a]],labels)
            context.geometry = {tuple(g['features']):g for g in inner['geometry']}
            context.trees = {tuple(t['features']):t for t in inner['trees']}
            for fit in inner['models']:
                if fit['C'] == fit['pair_penalty'] == 1. and (fit['edges'] or len(fit['features']) <= 1):
                    context.classifiers[(tuple(fit['features']),1.,1.)] = fit
            weights = np.array([1/(len(full['classes'])*sum(labels == c)) for c in labels])
            empirical_teacher,empirical_energy,law = marginal_product_anova(as_spline3(full),A,weights)
            uniform_teacher = canonical_raw_anova(as_spline3(full)); uniform_energy = functional_score_energies(uniform_teacher)
            np.testing.assert_allclose(spline_scores(A,empirical_teacher),spline_scores(A,as_spline3(full)),atol=1e-10)
            paths,times = {},{}
            for key,energy in [('empirical',empirical_energy),('uniform',uniform_energy)]:
                start = time.perf_counter(); paths[key] = feasible_count_refinement(energy,old['features'],context.candidate)
                times[key] = time.perf_counter()-start
            target = len(old['features'])
            fits = {'old':as_spline3(old),'full':as_spline3(full)}
            for key in ('empirical','uniform'):
                fits[key+'_refit'] = as_spline3(context.classifier(paths[key]['features'],1.,1.))
            selector_paths = {}
            for method,ci,prefix in [('fgs2023',4,'fgs'),('sfrip2025',3,'sfrip')]:
                path = native['splits'][i]['inner'][j]['paths'][method][ci]
                nf,mf = sorted(path['native']['features']),sorted(path['ranking'][:target])
                fits[prefix+'_native'] = as_spline3(context.classifier(nf,1.,1.))
                fits[prefix+'_matched'] = as_spline3(context.classifier(mf,1.,1.))
                selector_paths[method] = {'selector_index':ci,'config':GRIDS[method][ci],
                    'ranking':path['ranking'],'native_features':nf,'matched_features':mf}
            # All support choices and fits are complete before any query metric.
            exact = all(len(fits[m]['features']) == target for m in ('empirical_refit','uniform_refit','fgs_matched','sfrip_matched'))
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
                'target_count':target,'counts_exact':exact,'empirical_teacher':empirical_teacher,
                'empirical_energy':empirical_energy,'marginal_law':law,'uniform_teacher':uniform_teacher,
                'uniform_energy':uniform_energy,'paths':paths,'selection_seconds':times,
                'selector_paths':selector_paths,'fits':fits,'metrics':metrics,'boxes':boxes})
            out.write_text(json.dumps({'dataset':name,'d':d,'fingerprints':hashes,'records':records,
                'complete':len(records) == sum(len(s['inner']) for s in source['splits']),
                'scope':'training_marginal_product_fixed_count_certified_surrogate_development'},separators=(',',':')),encoding='utf-8')
            print(name,i,j,{m:round(v['clean'],5) for m,v in metrics.items()},
                'changed',paths['empirical']['features'] != old['features'],flush=True)

def summarize(names):
    rows = []; counts = {'records':0,'count_mismatches':0,'optimizer_caps':0,
        'empirical_changed':0,'uniform_changed':0,'empirical_attempts':0,'uniform_attempts':0,
        'empirical_rejections':0,'uniform_rejections':0}
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete']
        for r in saved['records']:
            counts['records'] += 1; counts['count_mismatches'] += not r['counts_exact']
            counts['optimizer_caps'] += sum(not f['optimizer_converged'] for f in r['fits'].values())
            for key in ('empirical','uniform'):
                path = r['paths'][key]; counts[key+'_changed'] += path['features'] != r['fits']['old']['features']
                counts[key+'_attempts'] += len(path['attempts']); counts[key+'_rejections'] += sum(not a['accepted'] for a in path['attempts'])
                assert path['candidate']['certificate']['feasible'] and path['final_loss'] <= path['initial_loss']+1e-10
        rows.append({'dataset':name,'methods':{m:{metric:float(np.mean([r['metrics'][m][metric] for r in saved['records']]))
            for metric in ('clean','noise','robust','removed')} for m in METHODS}})
    assert len(rows) == 15 and counts['records'] == 128
    result = {'rows':rows,'counts':counts,
        'means':{m:{metric:float(np.mean([r['methods'][m][metric] for r in rows])) for metric in ('clean','noise','robust','removed')} for m in METHODS},
        'contrasts':{metric:{m:paired([r['methods']['empirical_refit'][metric]-r['methods'][m][metric] for r in rows])
            for m in METHODS if m != 'empirical_refit'} for metric in ('clean','noise','robust')},
        'scope':'all15_all128_exposed_development_frozen_same_actual_count'}
    clean = result['contrasts']['clean']; corrected = holm({m:clean[m]['sign_flip_two_sided_p'] for m in ('full','fgs_matched','sfrip_matched')})
    result['matched_holm'] = corrected
    result['development_numeric_gate'] = bool(all(clean[m]['mean_pp'] >= .5 and corrected[m] < .05 for m in ('fgs_matched','sfrip_matched'))
        and clean['full']['mean_pp'] >= 0 and result['means']['empirical_refit']['removed'] >= .1
        and counts['count_mismatches'] == counts['optimizer_caps'] == 0)
    result['independent_audit_required'] = True; result['goal_achieved'] = False
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k != 'rows'},flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args(); FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in ([args.only] if args.only else names)]): f.result()
    if not args.only: summarize(names)
