"""ALL15FIRSTfold prospective geometry/sparsity attribution; training only."""
import argparse
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT, digest
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from sparse_global_frs_training_ablation import fit_block, ARMS

CAPPED = ROOT/'results/capped-frs-training'
DIAGNOSTIC = ROOT/'results/frs-coordinate-diagnostic-v2'
FOLDER = ROOT/'results/sparse-global-frs-training'
FREEZE = ROOT/'research/sparse-global-frs-training-source-freeze.json'


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
    audits = {}
    for folder, review_name in ((CAPPED, 'capped-frs-training-review.json'),
        (DIAGNOSTIC, 'FRS-coordinate-minimum-review-v2.json'),
        (ROOT/'results/pair-transition-frs-training', 'pair-transition-FRS-training-review.json'),
        (ROOT/'results/latent-frs-training', 'latent-FRS-training-review.json')):
        a = verify(folder); r = json.loads((ROOT/'research'/review_name).read_text())
        assert r['audit_sha256'] == digest(folder/'complete-audit.json')
        assert len(a['artifact_hashes']) == 15 and not r['goal_achieved'] and not r['classification_scored']
        audits[folder.name] = a
    names = set(audits[CAPPED.name]['artifact_hashes'])
    assert all(set(a['artifact_hashes']) == names for a in audits.values())
    return audits


def run(name):
    threadpool_limits(1); prerequisites(); hashes = frozen(); FOLDER.mkdir(exist_ok=True)
    path = FOLDER/(name+'.json')
    if path.exists():
        saved = json.loads(path.read_text()); assert saved['fingerprints'] == hashes
        if saved['complete']:
            return
    source = json.loads((CAPPED/(name+'.json')).read_text()); diag = json.loads((DIAGNOSTIC/(name+'.json')).read_text())
    old = source['inner_training'][0]
    assert diag['reference_indices'] == source['reference_indices'] and diag['training_indices'] == old['reference_indices']
    assert diag['original_support'] == old['features'] and diag['original_anchor'] == old['anchor']
    ri, ca = np.asarray(source['reference_indices']), np.asarray(old['reference_indices'])
    X, y = load_dataset(name); A, _ = preprocess_fit(X[ri][ca], X[ri][ca]); labels = y[ri][ca]
    block = None
    if old['strict_cell_available']:
        endpoint = old['fits'][old['training_selected_start']]['target']
        block = fit_block(A, labels, old['features'], old['anchor'], endpoint, diag['diagnostic'])
    saved = {'dataset':name, 'fingerprints':hashes, 'reference_indices':ri.tolist(),
        'training_indices':ca.tolist(), 'fold':0, 'strict_cell_available':old['strict_cell_available'],
        'block':block, 'complete':True, 'classification_scored':False, 'goal_achieved':False,
        'scope':'all15firstfold_matched_three_arm_three_start_training_sparsity_geometry_ablation'}
    path.write_text(json.dumps(saved, default=builtin, separators=(',', ':')), encoding='utf-8')
    print(name, 'matched global/sparse training complete', None if block is None else block['evaluation'], flush=True)


def row(name):
    s = json.loads((FOLDER/(name+'.json')).read_text())
    return {'dataset':name, 'available_original_cell':s['strict_cell_available'],
        'evaluation':None if s['block'] is None else s['block']['evaluation'],
        'common_zero_feature':None if s['block'] is None else s['block']['common_zero_feature']}


def summarize(names):
    rows = [row(n) for n in names]; available = [r for r in rows if r['evaluation'] is not None]
    metrics = {k:float(np.mean([r['evaluation'][k] for r in available])) for k in
        ('global_region_bound_decrease_vs_fixed_pp', 'sparse_bound_decrease_vs_global_pp',
         'sparse_additional_exact_zeros_vs_global')}
    failures = {a:sum(r['evaluation']['optimizer_failures'][a] for r in available) for a in ARMS}
    summary = {'rows':rows, 'datasets':len(names), 'training_folds':len(names),
        'dataset_balanced_metrics':metrics, 'optimizer_failures':failures,
        'available_regions':{a:sum(r['evaluation']['available'][a] for r in available) for a in ARMS},
        'folds_with_exact_zero_weights':{a:sum(r['evaluation']['exact_zero_weights'][a] > 0 for r in available) for a in ARMS},
        'artifact_hashes':{n:digest(FOLDER/(n+'.json')) for n in names},
        'independent_audit_required':True, 'goal_achieved':False, 'classification_scored':False,
        'limits':['Training error bounds and weights are not validation/test accuracy.',
            'Sparsity objectives differ by arm; compare risk-bound and actual zeros separately.',
            'Unavailable global region falls back explicitly, without a claimed successful fit.',
            'Only first training fold of every exposed dataset; no confirmation or official baseline replication.']}
    (FOLDER/'complete.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows', 'artifact_hashes')}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--dataset'); args = p.parse_args()
    if args.dataset:
        run(args.dataset)
    else:
        names = sorted(prerequisites()[CAPPED.name]['artifact_hashes'])
        for name in names:
            run(name)
        summarize(names)
