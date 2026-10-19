"""Retry only previously failed group stages with fixed Clarabel->SCS fallback."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.joint_group_fuzzy import joint_group_scores


def main():
    threadpool_limits(1)
    source = ROOT/'results/nested-joint-group'
    out = ROOT/'results/joint-solver-fallback-diagnostic.json'
    summary = json.loads((source/'complete.json').read_text())
    rows, counts = [], {'old_failed': 0, 'retry_feasible': 0, 'retry_failed': 0,
                        'clarabel_solver_error': 0, 'scs_used': 0}
    for row in summary['rows']:
        name = row['dataset']; saved = json.loads((source/(name+'.json')).read_text())
        X, y = load_dataset(name)
        for split in saved['splits']:
            ri = np.array(split['reference_indices']); raw, labels = X[ri], y[ri]
            stages = []
            stages.extend(inner['by_C'][str(split['selected_config']['C'])]['stage'] for inner in split['inner'])
            stages.append(split['outer_stages'][str(split['selected_config']['C'])])
            for stage in stages:
                old = stage['joint_fit']
                if old['feasible']:
                    continue
                counts['old_failed'] += 1
                # Recover the exact stage's raw table from its enclosing inner fold.
                # The outer stage is always the complete reference; inner stages are
                # handled by the caller's saved index information below.
            # Rebuild every failed inner stage with exact fold references.
            for inner in split['inner']:
                a = np.array(inner['reference_indices']);
                stage = inner['by_C'][str(split['selected_config']['C'])]['stage']
                if stage['joint_fit']['feasible']:
                    continue
                A, _ = preprocess_fit(raw[a], raw[a])
                retry = joint_group_scores(A, labels[a], stage['teacher_coef'], stage['teacher_intercept'],
                                            stage['teacher_classes'], C=stage['teacher_C'])
                counts['retry_feasible'] += retry['feasible']; counts['retry_failed'] += not retry['feasible']
                counts['clarabel_solver_error'] += retry['solver_attempts'][0]['status'] == 'solver_error'
                counts['scs_used'] += len(retry['solver_attempts']) > 1
                rows.append({'dataset': name, 'kind': 'inner', 'old_status': stage['joint_fit']['status'],
                             'retry_status': retry['status'], 'retry_feasible': retry['feasible'],
                             'solver_attempts': retry['solver_attempts']})
            stage = split['outer_stages'][str(split['selected_config']['C'])]
            if not stage['joint_fit']['feasible']:
                A, _ = preprocess_fit(raw, raw)
                retry = joint_group_scores(A, labels, stage['teacher_coef'], stage['teacher_intercept'],
                                           stage['teacher_classes'], C=stage['teacher_C'])
                counts['retry_feasible'] += retry['feasible']; counts['retry_failed'] += not retry['feasible']
                counts['clarabel_solver_error'] += retry['solver_attempts'][0]['status'] == 'solver_error'
                counts['scs_used'] += len(retry['solver_attempts']) > 1
                rows.append({'dataset': name, 'kind': 'outer', 'old_status': stage['joint_fit']['status'],
                             'retry_status': retry['status'], 'retry_feasible': retry['feasible'],
                             'solver_attempts': retry['solver_attempts']})
        print(name, 'failed group stages retried', flush=True)
    result = {'rows': rows, 'counts': counts,
              'scope': 'failed_stages_only_solver_fallback_diagnostic_not_nested_reselection'}
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(counts)


if __name__ == '__main__':
    main()
