"""Same-anchor bound comparison from fully audited inputs, not accuracy gains."""
import json
import sys
import time
import traceback
import numpy as np
from run_third_enemy_inner import ROOT, FOLDER, PREVIOUS, METHODS, digest
from continue_hardaware_soft5_knn_pilot import wait_verified
from continue_soft_topk_knn_pilot import verify
from audit_forest_compact_transfer import equivalent

STATUS = ROOT/'results/third-enemy-bound-tightness-status.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase': phase, 'updated_unix': time.time(), 'goal_achieved': False, **values}, indent=2), encoding='utf-8')


def main(pid):
    status('waiting_for_complete_DC_independent_audit', prerequisite_pid=pid)
    current_audit = wait_verified(FOLDER, pid); previous_audit = verify(PREVIOUS)
    rows = []
    families = [('third_enemy_unconstrained', 'majority_hinge_unconstrained'),
                ('third_enemy_constrained', 'majority_hinge_constrained')]
    counts = {'datasets': 0, 'matched_anchor_bounds': 0, 'previous_target_failures': 0, 'current_subproblem_failures': 0}
    for name in sorted(current_audit['artifact_hashes']):
        cur = json.loads((FOLDER/(name+'.json')).read_text()); prev = json.loads((PREVIOUS/(name+'.json')).read_text())
        assert current_audit['artifact_hashes'][name] == digest(FOLDER/(name+'.json'))
        assert previous_audit['artifact_hashes'][name] == digest(PREVIOUS/(name+'.json'))
        dataset = {m: [] for m in METHODS}
        for b, old in zip(cur['inner'], prev['inner'], strict=True):
            assert b['reference_indices'] == old['reference_indices']
            old_lookup = {(tuple(l['features']), tuple(l['anchor']), l['family']): l for l in old['stage']['learners']}
            new_lookup = {(tuple(l['features']), tuple(l['anchor']), l['family']): l for l in b['stage']['learners']}
            for m in METHODS:
                equivalent(b['bases'][m], old['bases'][m]); base = b['bases'][m]
                for family, old_family in families:
                    key = (tuple(base['features']), tuple(base['weights']))
                    before = old_lookup[(*key, old_family)]['fit']; after = new_lookup[(*key, family)]['fit']
                    equivalent(before['training_witnesses']['witnesses'], after['training_witnesses']['witnesses'])
                    equivalent(before['training_witnesses']['alpha'], after['training_witnesses']['alpha'])
                    convex = before['training_error_upper_bound_initial']; rank = after['training_error_upper_bound_initial']
                    assert rank <= convex+1e-10
                    dataset[m].append({'convex_anchor_bound': convex, 'rank_anchor_bound': rank, 'tightening_pp': 100*(convex-rank)})
                    counts['matched_anchor_bounds'] += 1
            counts['previous_target_failures'] += sum(not l['fit']['optimizer_success'] for l in old['stage']['learners'])
            counts['current_subproblem_failures'] += sum(l['fit']['subproblem_failures'] for l in b['stage']['learners'])
        rows.append({'dataset': name, 'same_anchor_bound_means': {m: {k: float(np.mean([r[k] for r in dataset[m]]))
                     for k in ('convex_anchor_bound', 'rank_anchor_bound', 'tightening_pp')} for m in METHODS}})
        counts['datasets'] += 1
    assert counts['datasets'] == 15
    report = {'rows': rows, 'dataset_balanced_means': {m: {k: float(np.mean([r['same_anchor_bound_means'][m][k] for r in rows]))
               for k in ('convex_anchor_bound', 'rank_anchor_bound', 'tightening_pp')} for m in METHODS},
        'counts': counts, 'input_audit_sha256': {str(FOLDER.relative_to(ROOT)): digest(FOLDER/'complete-audit.json'),
                                                 str(PREVIOUS.relative_to(ROOT)): digest(PREVIOUS/'complete-audit.json')},
        'reviewer_sha256': digest(ROOT/'scripts/review_third_enemy_bound_tightness.py'),
        'goal_achieved': False, 'classification_superiority_established': False,
        'scope': 'tighter_finite_training_shared_box_error_bound_on_IDENTICAL_anchors_and_witnesses_not_lower_test_error',
        'limits': ['The bound definitions differ; final weights are not used to confuse tighter certificate with better fitting.',
                   'All anchors, samples and witnesses are exactly matched; no failing fits or data are omitted.',
                   'Each input bound was independently numerically replayed; this report aggregates that audited evidence.',
                   'Finite-table training bound is not population risk, held-out performance or a novelty claim.']}
    (ROOT/'research/third-enemy-bound-tightness-review.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    status('complete_input_audits_verified_reviewed', counts=counts)


if __name__ == '__main__':
    try:
        main(int(sys.argv[1]))
    except Exception as error:
        status('failed', error=str(error)); traceback.print_exc(); raise
