"""Repeat every recorded Sonar FRS failure without any classifier scoring."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT, digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from fit_strict_frs_cell_hinge import fit_strict_frs_cell_target
from audit_frs_cell_fit_stored import check_cell_fit

FREEZE = ROOT/'research/strict-frs-solver-diagnostic-source-freeze.json'
FOLDER = ROOT/'results/strict-frs-solver-diagnostic'
SOURCE = ROOT/'results/frs-cell-inner-v2/sonar.json'


def builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    threadpool_limits(1); FOLDER.mkdir(exist_ok=True)
    hashes = json.loads(FREEZE.read_text())['fingerprints']
    for rel, sha in hashes.items():
        assert digest(ROOT/rel) == sha, rel
    saved = json.loads(SOURCE.read_text()); assert saved['complete']
    X, y = load_dataset('sonar'); ri = np.asarray(saved['reference_indices']); raw, labels = X[ri], y[ri]
    counts = Counter(); reports = []; source_keys = []
    for fi, b in enumerate(saved['inner']):
        ca = np.asarray(b['reference_indices']); A, _ = preprocess_fit(raw[ca], raw[ca])
        for li, learner in enumerate(b['stage']['learners']):
            if learner['family'] != 'frs_cell_constrained' or not learner['fit']['subproblem_failures']:
                continue
            source_keys.append([fi, li]); old = learner['fit']; anchor = np.asarray(learner['anchor'])
            assert not learner['certified_arm'] and not old['cell_unavailable']
            target, fit = fit_strict_frs_cell_target(A, labels[ca], learner['features'], anchor,
                full_reference=old['class_cell']['full_weights'], reference_scope=old['reference_scope'])
            record = {'features': learner['features'], 'anchor': anchor.tolist(), 'target': target.tolist(), 'fit': fit}
            check_cell_fit(A, labels[ca], record, counts)
            reports.append({'fold': fi, 'learner_id': li, 'original_failure_status': old['status'],
                'original_subproblem_failures': old['subproblem_failures'], 'new_record': record,
                'new_success': fit['optimizer_success'], 'original_fit_preserved': True,
                'new_classification_scored': False})
            print('Sonar failure replay', fi, li, 'new success', fit['optimizer_success'], 'status', fit['status'], flush=True)
    assert len(source_keys) == 4
    out = {'passed': True, 'fingerprints': hashes, 'source_sha256': digest(SOURCE), 'source_keys': source_keys,
        'rows': reports, 'independent_counts': dict(counts), 'new_successful_repeats': sum(r['new_success'] for r in reports),
        'goal_achieved': False, 'classification_scored': False,
        'scope': 'all_recorded_Sonar_control_failures_stricter_solver_training_only'}
    (FOLDER/'complete-audited.json').write_text(json.dumps(out, default=builtin, separators=(',', ':')), encoding='utf-8')


if __name__ == '__main__':
    main()
