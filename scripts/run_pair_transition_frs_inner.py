"""35-choice exposed INNER development, with full audited native controls."""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT, digest
from continue_soft_topk_knn_pilot import verify
from pair_transition_inner_design import (METHODS, STRONG, alignment, append_block,
    select_refined, strong_selection, cohort_summary)
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

SOURCE = ROOT/'results/atomic-pruning-classifier-v2'
PREVIOUS = ROOT/'results/monotone-frs-inner'
CANONICAL = ROOT/'results/majority-hinge-canonical-benchmark'
TRAINING = ROOT/'results/pair-transition-frs-training'
FOLDER = ROOT/'results/pair-transition-frs-inner'
FREEZE = ROOT/'research/pair-transition-frs-inner-source-freeze.json'


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
    audits = {folder.name:verify(folder) for folder in (SOURCE, PREVIOUS, CANONICAL, TRAINING)}
    capped = ROOT/'results/capped-frs-training'
    capped_audit = verify(capped)
    capped_review = json.loads((ROOT/'research/capped-frs-training-review.json').read_text())
    assert capped_review['audit_sha256'] == digest(capped/'complete-audit.json')
    assert len(capped_audit['artifact_hashes']) == 15 and not capped_review['classification_scored']
    names = set(audits[SOURCE.name]['artifact_hashes'])
    assert len(names) == 15
    for a in audits.values():
        assert set(a['artifact_hashes']) == names
    for folder, review_name in ((PREVIOUS, 'monotone-frs-inner-review.json'),
                                (TRAINING, 'pair-transition-FRS-training-review.json')):
        review = json.loads((ROOT/'research'/review_name).read_text())
        assert review['audit_sha256'] == digest(folder/'complete-audit.json')
        assert not review['goal_achieved'] and not review['classification_superiority_established']
    proof = json.loads((ROOT/'research/pair-transition-native-control-alignment.json').read_text())
    assert proof['passed'] and proof['datasets'] == 15 and proof['inner_folds'] == 44
    for r in proof['rows']:
        assert digest(SOURCE/(r['dataset']+'.json')) == r['atomic_sha256']
        assert digest(CANONICAL/(r['dataset']+'.json')) == r['canonical_sha256']
    return audits


def run(name):
    threadpool_limits(1); prerequisites(); hashes = frozen(); FOLDER.mkdir(exist_ok=True)
    out = FOLDER/(name+'.json')
    if out.exists():
        saved = json.loads(out.read_text()); assert saved['fingerprints'] == hashes
        if saved['complete']:
            return
    atomic = json.loads((SOURCE/(name+'.json')).read_text())
    prior = json.loads((PREVIOUS/(name+'.json')).read_text())
    canonical = json.loads((CANONICAL/(name+'.json')).read_text())
    alignment(atomic, canonical, prior)
    controls = strong_selection(canonical)
    X, y = load_dataset(name); ri = np.asarray(prior['reference_indices']); raw, labels = X[ri], y[ri]
    blocks = []
    for fi, old in enumerate(prior['inner']):
        part = FOLDER/(name+f'.inner{fi}.partial.json')
        if part.exists():
            saved = json.loads(part.read_text()); assert saved['fingerprints'] == hashes
            blocks.append(saved['block']); continue
        ca, cb = np.asarray(old['reference_indices']), np.asarray(old['validation_indices'])
        assert not set(ca)&set(cb)
        A, B = preprocess_fit(raw[ca], raw[cb]); start = time.perf_counter()
        b = append_block(A, labels[ca], B, labels[cb], old)
        b['new_fit_score_seconds'] = time.perf_counter()-start
        part.write_text(json.dumps({'fingerprints':hashes, 'block':b}, default=builtin, separators=(',', ':')), encoding='utf-8')
        blocks.append(b)
        new = b['stage']['learners'][b['previous_audited_learners']:]
        print(name, 'inner', fi+1, 'new pair fits', len(new), 'failures',
              sum(l['fit']['subproblem_failures'] for l in new), flush=True)
    saved = {'dataset':name, 'd':X.shape[1], 'fingerprints':hashes, 'inner':blocks,
        'reference_indices':ri.tolist(), 'selection':select_refined(blocks), 'strong_controls':controls,
        'canonical_inner_source_sha256':digest(CANONICAL/(name+'.json')),
        'complete':True, 'goal_achieved':False, 'new_outer_evaluation_performed':False,
        'scope':'append_only_35_choices_all_six_refined_arms_entire_native_controls_inner_development'}
    out.write_text(json.dumps(saved, default=builtin, separators=(',', ':')), encoding='utf-8')
    print(name, '35-choice INNER complete; not confirmatory', flush=True)


def dataset_row(name):
    saved = json.loads((FOLDER/(name+'.json')).read_text())
    prior = json.loads((PREVIOUS/(name+'.json')).read_text())
    pick = lambda s, m:s['selection'][m]['options'][s['selection'][m]['selected']]['mean_BA']
    refined = {m:pick(saved, m) for m in METHODS}
    strong = {m:saved['strong_controls'][m]['selected_recipe']['mean_BA'] for m in STRONG}
    return {'dataset':name,
        'anchor_gain_pp':{m:saved['selection'][m]['selected_inner_gain_pp'] for m in METHODS},
        'vs_previous_DC_pp':{m:100*(refined[m]-pick(prior, m)) for m in METHODS},
        'selected_inner_BA':{**refined, **strong},
        'selected_new_pair_family':{m:saved['selection'][m]['selected'] >= 28 for m in METHODS}}


def summarize(names):
    rows = [dataset_row(n) for n in names]; failures = folds = 0
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); folds += len(saved['inner'])
        failures += sum(l['fit']['subproblem_failures'] for b in saved['inner']
                        for l in b['stage']['learners'][b['previous_audited_learners']:])
    summary = {**cohort_summary(rows, failures), 'inner_folds':folds,
        'artifact_hashes':{n:digest(FOLDER/(n+'.json')) for n in names}, 'independent_audit_required':True}
    (FOLDER/'complete.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows', 'artifact_hashes')}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--dataset'); args = p.parse_args()
    if args.dataset:
        run(args.dataset)
    else:
        names = sorted(prerequisites()[SOURCE.name]['artifact_hashes'])
        with ProcessPoolExecutor(max_workers=2) as pool:
            for f in as_completed([pool.submit(run, n) for n in names]):
                f.result()
        summarize(names)
