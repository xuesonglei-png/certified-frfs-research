"""Fixed original teacher, exact surrogate DP, separately screened FRS."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from run_nested_geometry_forest import GRIDS,verified_native,holm
from run_spline_capacity_pilot import as_spline3
from certified_frfs.forest_anova_reduction import (canonical_raw_anova,functional_score_energies,
    forest_energy_optimal_support,projected_support_fit,omission_energy)
from certified_frfs.geometry_only_forest_stage import GeometryStageContext
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.spline_forest import spline_scores,spline_box_margins
from certified_frfs.forest_penalty_selection import balanced_accuracy
from run_forest_robustness_curve import coverage
from review_box_iterations import paired

SOURCE = ROOT/'results/nested-geometry-forest'
FOLDER = ROOT/'results/forest-anova-dp-pilot'
METHODS = ('old','full','dp_refit','dp_projected','fgs_native','sfrip_native','fgs_matched','sfrip_matched')


def run(name):
    threadpool_limits(1); source = source_verified(name); native = verified_native(name,source)
    hashes = dict(source['fingerprints'])
    for p in (Path(__file__),ROOT/'research/forest-anova-dp-pilot-protocol.md',SOURCE/(name+'.json'),
        ROOT/'src/certified_frfs/forest_anova_reduction.py',ROOT/'src/certified_frfs/functional_spline_forest.py',
        ROOT/'src/certified_frfs/spline_forest.py',ROOT/'scripts/run_nested_joint_path.py',
        ROOT/'scripts/run_spline_capacity_pilot.py',ROOT/'scripts/audit_nested_geometry_forest.py'):
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
            teacher = canonical_raw_anova(as_spline3(full)); energies = functional_score_energies(teacher)
            np.testing.assert_allclose(spline_scores(A,teacher),spline_scores(A,as_spline3(full)),atol=1e-10)
            context = GeometryStageContext(X[ri[a]],labels)
            context.geometry = {tuple(g['features']):g for g in inner['geometry']}
            context.trees = {tuple(t['features']):t for t in inner['trees']}
            for fit in inner['models']:
                if fit['C'] == fit['pair_penalty'] == 1. and (fit['edges'] or len(fit['features']) <= 1):
                    context.classifiers[(tuple(fit['features']),1.,1.)] = fit
            candidates = []
            for count in range(int(np.ceil(.8*d)),d+1):
                result = forest_energy_optimal_support(energies['unary_energy'],energies['edges'],energies['pair_energy'],count)
                candidate = context.candidate(result['features'])
                candidates.append({'dp':result,'candidate':candidate})
                if candidate['certificate']['feasible']: break
            selected = candidates[-1]; assert selected['candidate']['certificate']['feasible']
            fs = selected['candidate']['features']; target = len(fs)
            fits = {'old':as_spline3(old),'full':as_spline3(full),
                'dp_refit':as_spline3(context.classifier(fs,1.,1.)),'dp_projected':projected_support_fit(teacher,fs)}
            paths = {}
            for method,ci,prefix in [('fgs2023',4,'fgs'),('sfrip2025',3,'sfrip')]:
                path = native['splits'][i]['inner'][j]['paths'][method][ci]
                native_features = sorted(path['native']['features']); matched = sorted(path['ranking'][:target])
                fits[prefix+'_native'] = as_spline3(context.classifier(native_features,1.,1.))
                fits[prefix+'_matched'] = as_spline3(context.classifier(matched,1.,1.))
                paths[method] = {'selector_index':ci,'config':GRIDS[method][ci],'ranking':path['ranking'],
                    'native_features':native_features,'matched_features':matched}
            exact = all(len(fits[m]['features']) == target for m in ('fgs_matched','sfrip_matched'))
            rng = np.random.default_rng(144+i); queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
            metrics,boxes = {},{}
            for method,fit in fits.items():
                values = [balanced_accuracy(spline_scores(Q,fit),fit['classes'],y[ri[b]]) for Q in queries]
                box = spline_box_margins(B,y[ri[b]],fit,eps=.01,witnesses=True)
                boxes[method] = {'margins':box['margins'].tolist(),'rivals':box['rivals'].tolist(),'witness_raw':box['witness_raw'].tolist()}
                metrics[method] = {'clean':values[0],'noise':float(np.mean(values[1:])),
                    'robust':coverage(box['margins'] > 1e-10,y[ri[b]]),'removed':1-len(fit['features'])/d}
            records.append({'outer_index':i,'inner_index':j,'outer_reference_indices':ri.tolist(),
                'reference_indices':a.tolist(),'validation_indices':b.tolist(),'query_seed':144+i,'target_count':target,
                'counts_exact':exact,'teacher':teacher,'energies':energies,'candidates':candidates,
                'uniform_omission_losses':{'old':omission_energy(energies,old['features']),
                    'selected_dp':omission_energy(energies,fs)},
                'selector_paths':paths,'fits':fits,'metrics':metrics,'boxes':boxes})
            out.write_text(json.dumps({'dataset':name,'d':d,'fingerprints':hashes,'records':records,
                'complete':len(records) == sum(len(s['inner']) for s in source['splits']),
                'scope':'fixed_original_teacher_uniform_ANOVA_surrogate_optimal_support_independent_FRS_screening_development'},separators=(',',':')),encoding='utf-8')
            print(name,i,j,'anova DP',{m:round(v['clean'],5) for m,v in metrics.items()},
                'attempt_counts',[c['dp']['count'] for c in candidates],flush=True)


def summarize(names):
    rows = []; counts = {'records':0,'count_mismatches':0,'optimizer_caps':0,'same_old_supports':0,
        'full_fallbacks':0,'restored_count_references':0,'total_dp_candidates':0}
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete']
        for r in saved['records']:
            counts['records'] += 1; counts['count_mismatches'] += not r['counts_exact']
            counts['optimizer_caps'] += sum(not f['optimizer_converged'] for m,f in r['fits'].items() if m != 'dp_projected')
            counts['same_old_supports'] += r['fits']['dp_refit']['features'] == r['fits']['old']['features']
            counts['full_fallbacks'] += len(r['fits']['dp_refit']['features']) == saved['d']
            counts['restored_count_references'] += len(r['candidates']) > 1; counts['total_dp_candidates'] += len(r['candidates'])
            assert r['candidates'][-1]['candidate']['certificate']['feasible']
        rows.append({'dataset':name,'methods':{m:{metric:float(np.mean([r['metrics'][m][metric] for r in saved['records']]))
            for metric in ('clean','noise','robust','removed')} for m in METHODS}})
    assert len(rows) == 15 and counts['records'] == 128
    result = {'rows':rows,'counts':counts,
        'means':{m:{metric:float(np.mean([r['methods'][m][metric] for r in rows])) for metric in ('clean','noise','robust','removed')} for m in METHODS},
        'contrasts':{metric:{m:paired([r['methods']['dp_refit'][metric]-r['methods'][m][metric] for r in rows]) for m in METHODS if m != 'dp_refit'}
            for metric in ('clean','noise','robust')},
        'scope':'uniform_model_reconstruction_optimum_FRS_screened_all15_all128_exposed_development'}
    clean = result['contrasts']['clean']; corrected = holm({m:clean[m]['sign_flip_two_sided_p'] for m in ('full','fgs_matched','sfrip_matched')})
    result['matched_holm'] = corrected
    result['gate_passed'] = (all(clean[m]['mean_pp'] >= .5 and corrected[m] < .05 for m in ('fgs_matched','sfrip_matched'))
        and clean['full']['mean_pp'] >= 0 and result['means']['dp_refit']['removed'] >= .1
        and counts['count_mismatches'] == counts['optimizer_caps'] == 0)
    result['goal_achieved'] = False
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k != 'rows'},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args(); FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in ([args.only] if args.only else names)]): f.result()
    if not args.only: summarize(names)
