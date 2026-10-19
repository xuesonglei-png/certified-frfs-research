"""ALL15 FIRST training-fold fixed-cell deletion bottleneck diagnostic."""
import argparse
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT, digest
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from frs_coordinate_minimum import solve_diagnostic, MODES

SOURCE = ROOT/'results/capped-frs-training'
FOLDER = ROOT/'results/frs-coordinate-diagnostic-v2'
FREEZE = ROOT/'research/frs-coordinate-diagnostic-v2-source-freeze.json'


def builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def frozen():
    hashes = json.loads(FREEZE.read_text())['fingerprints']
    for rel, sha in hashes.items():
        assert digest(ROOT/rel) == sha, rel
    return hashes


def prerequisites():
    audit = verify(SOURCE); review = json.loads((ROOT/'research/capped-frs-training-review.json').read_text())
    assert review['audit_sha256'] == digest(SOURCE/'complete-audit.json')
    assert len(audit['artifact_hashes']) == 15 and not review['classification_scored']
    return audit


def run(name):
    threadpool_limits(1); prerequisites(); hashes = frozen(); FOLDER.mkdir(exist_ok=True)
    path = FOLDER/(name+'.json')
    if path.exists():
        saved = json.loads(path.read_text()); assert saved['fingerprints'] == hashes
        if saved['complete']:
            return
    old_folder = ROOT/'results/frs-coordinate-diagnostic'
    old_case, old_audit = old_folder/(name+'.json'), old_folder/(name+'.smoke-audit.json')
    if old_case.exists() and old_audit.exists():
        replay = json.loads(old_audit.read_text())
        assert replay['passed'] and replay['sha256'] == digest(old_case)
        previous = json.loads(old_case.read_text())
        for rel, sha in previous['fingerprints'].items():
            assert digest(ROOT/rel) == sha, rel
        previous['fingerprints'] = hashes
        previous['audited_v1_smoke_reuse_sha256'] = digest(old_case)
        path.write_text(json.dumps(previous, default=builtin, separators=(',', ':')), encoding='utf-8')
        print(name, 'fully audited v1 smoke reused without recomputation', flush=True)
        return
    source = json.loads((SOURCE/(name+'.json')).read_text()); old = source['inner_training'][0]
    ri, ca = np.asarray(source['reference_indices']), np.asarray(old['reference_indices'])
    X, y = load_dataset(name); raw, labels = X[ri], y[ri]
    A, _ = preprocess_fit(raw[ca], raw[ca]); timer = time.perf_counter()
    saved = {'dataset':name, 'd':X.shape[1], 'fold':0, 'fingerprints':hashes,
        'reference_indices':ri.tolist(), 'training_indices':ca.tolist(),
        'original_support':old['features'], 'original_anchor':old['anchor'],
        'strict_cell_available':old['strict_cell_available']}
    if old['strict_cell_available']:
        saved['diagnostic'] = solve_diagnostic(A, labels[ca], old['features'], old['anchor'])
    else:
        saved['diagnostic'] = None
    saved.update({'complete':True, 'fit_seconds':time.perf_counter()-timer,
        'classification_scored':False, 'goal_achieved':False,
        'scope':'all15_first_training_fold_individual_coordinate_minimum_not_classifier_or_joint_reduct'})
    path.write_text(json.dumps(saved, default=builtin, separators=(',', ':')), encoding='utf-8')
    counts = None if saved['diagnostic'] is None else {m:sum(r['accepted_individual_zero_witness'] for r in
        saved['diagnostic']['regimes'][m]['coordinates'] if r['success']) for m in MODES}
    print(name, 'first-training-fold raw accepted individual zero witnesses', counts, flush=True)


def row(name):
    saved = json.loads((FOLDER/(name+'.json')).read_text()); out = {'dataset':name,
        'available':saved['strict_cell_available'], 'support_size':len(saved['original_support']),
        'fit_seconds':saved['fit_seconds'], 'regimes':{}}
    if saved['diagnostic'] is None:
        return out
    for mode in MODES:
        r = saved['diagnostic']['regimes'][mode]; successes = [v for v in r['coordinates'] if v['success']]
        out['regimes'][mode] = {'coordinate_LPs':len(r['coordinates']), 'failures':r['failures'],
            'accepted_individual_zero_witnesses':sum(v['accepted_individual_zero_witness'] for v in successes),
            'raw_exact_zero_candidates':sum(v['exact_zero_candidate'] for v in successes),
            'actual_coordinate_rejections':sum(not v['original_constraints_passed_at_actual_coordinates'] for v in successes),
            'numerically_positive_minimum_coordinates':sum(v['actual_minimum_weight'] > 1e-6 for v in successes),
            'actual_minima':{str(v['feature']):v['actual_minimum_weight'] for v in successes}}
    return out


def summarize(names):
    rows = [row(n) for n in names]
    sums = {mode:{k:sum(r['regimes'][mode][k] for r in rows if r['available']) for k in
        ('coordinate_LPs', 'failures', 'accepted_individual_zero_witnesses', 'raw_exact_zero_candidates',
         'actual_coordinate_rejections', 'numerically_positive_minimum_coordinates')} for mode in MODES}
    summary = {'rows':rows, 'datasets':len(names), 'training_folds':len(names),
        'available_cells':sum(r['available'] for r in rows), 'counts':sums,
        'artifact_hashes':{n:digest(FOLDER/(n+'.json')) for n in names}, 'independent_audit_required':True,
        'goal_achieved':False, 'classification_scored':False,
        'limits':['A positive minimum only concerns this fixed sufficient cell, not all valid reducts.',
            'Individually feasible deletions cannot be combined without common feasibility rechecking.',
            'Numerical LP primal/duals are not directed interval or exact infeasibility certificates.',
            'First fold of every exposed dataset is a diagnostic, not confirmation or an accuracy experiment.']}
    (FOLDER/'complete.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows', 'artifact_hashes')}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--dataset'); args = p.parse_args()
    if args.dataset:
        run(args.dataset)
    else:
        names = sorted(prerequisites()['artifact_hashes'])
        for name in names:
            run(name)
        summarize(names)
