"""Fixed audited outer classifiers, exact numeric-box robustness curves."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from run_nested_geometry_forest import holm
from review_box_iterations import paired
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.forest_certificate_comparison import compare_forest_certificates
from certified_frfs.orthogonal_forest_logistic import orthogonal_scores
from certified_frfs.forest_penalty_selection import balanced_accuracy

SOURCE = ROOT/'results/nested-geometry-forest'
MATCHED = ROOT/'results/geometry-forest-matched-count'
FOLDER = ROOT/'results/forest-robustness-curve'
METHODS = ('proposed','full','fgs2023','sfrip2025','fgs2023_matched','sfrip2025_matched')
RADII = (0.,.01,.03,.05,.10)
CERTIFICATES = ('exact','independent','lipschitz')


def coverage(mask,y):
    return float(np.mean([np.mean(mask[y == c]) for c in np.unique(y)]))


def area(curve):
    return float(np.trapezoid(curve,RADII)/RADII[-1])


def matched_verified(name):
    artifact = MATCHED/(name+'.json'); saved = json.loads(artifact.read_text()); assert saved['complete']
    assert json.loads((MATCHED/'complete-audit.json').read_text())['passed']
    per = json.loads((MATCHED/(name+'-audit.json')).read_text())
    for rel,value in saved['fingerprints'].items(): assert digest(ROOT/rel) == value
    for rel,value in per['fingerprints'].items(): assert digest(ROOT/rel) == value
    assert per['fingerprints'][str(artifact.relative_to(ROOT))] == digest(artifact)
    return saved


def run(name):
    threadpool_limits(1); source = source_verified(name); matched = matched_verified(name)
    hashes = {**source['fingerprints'],**matched['fingerprints']}
    for p in (Path(__file__),ROOT/'research/forest-robustness-curve-protocol.md',SOURCE/(name+'.json'),MATCHED/(name+'.json'),
        ROOT/'src/certified_frfs/forest_certificate_comparison.py',ROOT/'src/certified_frfs/forest_lipschitz_metric.py',
        ROOT/'scripts/audit_nested_geometry_forest.py',ROOT/'scripts/audit_geometry_forest_matched_count.py'):
        hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    out = FOLDER/(name+'.json'); records = []
    if out.exists():
        previous = json.loads(out.read_text()); assert previous['fingerprints'] == hashes
        if previous['complete']: return
        records = previous['splits']
    X,y = load_dataset(name)
    assert len(source['splits']) == len(matched['splits'])
    for i,(split,control) in enumerate(zip(source['splits'],matched['splits'])):
        if i < len(records): continue
        for key in ('reference_indices','validation_indices'): assert split[key] == control[key]
        ri,vi = np.array(split['reference_indices']),np.array(split['validation_indices'])
        _,B = preprocess_fit(X[ri],X[vi]); truth = y[vi]
        fits = {m:split['models'][split['results'][m]['model_id']] for m in METHODS[:4]}
        fits.update({m+'_matched':control['results'][m]['fit'] for m in ('fgs2023','sfrip2025')})
        assert len(fits['proposed']['features']) == control['target_count']
        results = {}
        for method,fit in fits.items():
            curves = []
            for eps in RADII:
                values = compare_forest_certificates(B,truth,fit,eps=eps); exact = values['exact']
                vectors = {'exact':np.array(exact['margins']),'independent':values['independent_termwise_margins'],
                    'lipschitz':values['lipschitz_margins']}
                assert all(np.all(v <= vectors['exact']+1e-9) for v in vectors.values())
                curves.append({'radius':eps,'margins':{k:v.tolist() for k,v in vectors.items()},
                    'coverage':{k:coverage(v > 1e-10,truth) for k,v in vectors.items()},
                    'negative_witness_rate':coverage(vectors['exact'] < -1e-10,truth),
                    'rivals':np.array(exact['rivals']).tolist(),'witness_raw':np.array(exact['witness_raw']).tolist(),
                    'pair_margins':[[float(v) if np.isfinite(v) else None for v in row] for row in exact['pair_margins']]})
            clean = balanced_accuracy(orthogonal_scores(B,fit),fit['classes'],truth)
            results[method] = {'fit':fit,'features':fit['features'],'clean':clean,'curves':curves,
                'area':{k:area([row['coverage'][k] for row in curves]) for k in CERTIFICATES},
                'radius_zero_tie_gap':clean-curves[0]['coverage']['exact']}
        records.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'results':results})
        out.write_text(json.dumps({'dataset':name,'d':X.shape[1],'fingerprints':hashes,'splits':records,
            'complete':len(records) == len(source['splits']),'radii':RADII,
            'scope':'unchanged_classifiers_numeric_box_certification_development_native_and_conditional_matched_controls'},separators=(',',':')),encoding='utf-8')
        print(name,i,'robustness curve',{m:{k:round(v,5) for k,v in row['area'].items()} for m,row in results.items()},flush=True)


def summarize(names):
    rows = []; counts = {'outer_splits':0,'models':0,'radius_evaluations':0,'optimizer_caps':0,'zero_radius_nonzero_tie_gaps':0}
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete']
        counts['outer_splits'] += len(saved['splits'])
        for s in saved['splits']:
            for m,r in s['results'].items():
                counts['models'] += 1; counts['radius_evaluations'] += len(r['curves'])
                counts['optimizer_caps'] += not r['fit']['optimizer_converged']
                counts['zero_radius_nonzero_tie_gaps'] += abs(r['radius_zero_tie_gap']) > 1e-12
        rows.append({'dataset':name,'methods':{m:{'clean':float(np.mean([s['results'][m]['clean'] for s in saved['splits']])),
            'removed':float(np.mean([1-len(s['results'][m]['features'])/saved['d'] for s in saved['splits']])),
            'area':{k:float(np.mean([s['results'][m]['area'][k] for s in saved['splits']])) for k in CERTIFICATES},
            'curves':{k:[float(np.mean([s['results'][m]['curves'][i]['coverage'][k] for s in saved['splits']])) for i in range(len(RADII))]
                for k in CERTIFICATES}} for m in METHODS}})
    assert len(rows) == 15 and counts['outer_splits'] == 44
    exact_gain = {m:paired([r['methods']['proposed']['area']['exact']-r['methods'][m]['area']['exact'] for r in rows]) for m in METHODS[1:]}
    result = {'rows':rows,'counts':counts,'radii':RADII,
        'means':{m:{'clean':float(np.mean([r['methods'][m]['clean'] for r in rows])),
            'removed':float(np.mean([r['methods'][m]['removed'] for r in rows])),
            'area':{k:float(np.mean([r['methods'][m]['area'][k] for r in rows])) for k in CERTIFICATES},
            'curves':{k:np.mean([r['methods'][m]['curves'][k] for r in rows],axis=0).tolist() for k in CERTIFICATES}} for m in METHODS},
        'same_classifier_certificate_gain':{m:{k:paired([r['methods'][m]['area']['exact']-r['methods'][m]['area'][k] for r in rows])
            for k in CERTIFICATES[1:]} for m in METHODS},
        'exact_method_gain':exact_gain,
        'exact_method_gain_holm':holm({m:r['sign_flip_two_sided_p'] for m,r in exact_gain.items()}),
        'scope':'exposed_development_fixed_numeric_box_certification_not_clean_prediction_SOTA_or_population_effect'}
    primary = result['same_classifier_certificate_gain']['proposed']['independent']
    result['diagnostic_gate_passed'] = primary['mean_pp'] >= 1 and primary['sign_flip_two_sided_p'] < .05
    result['goal_achieved'] = False
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k != 'rows'},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args(); FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in ([args.only] if args.only else names)]): f.result()
    if not args.only: summarize(names)
