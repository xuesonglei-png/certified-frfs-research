"""Trace archived diagnostic references to their original raw-data split.

No predictive evaluation. Small references are nested development partitions,
not an undocumented sampling cap or proven experimental implementation bug.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from run_nested_joint_path import ROOT, source_verified, digest
from screen_multiknot_forest import FOLDER
from certified_frfs.datasets import load_dataset
from certified_frfs.nested_teacher_regularized import inner_splits


if __name__ == '__main__':
    names = [r['dataset'] for r in json.loads((FOLDER/'complete.json').read_text())['rows']]
    rows, hashes = [], {}
    for name in names:
        X, y = load_dataset(name)
        path = ROOT/'results/predictive-iteration41'/(name+'-s101-f0.json')
        parent = json.loads(path.read_text())
        source = source_verified(name)
        mechanism = json.loads((FOLDER/(name+'.json')).read_text())
        hashes[path.relative_to(ROOT).as_posix()] = digest(path)
        hashes[(FOLDER/(name+'.json')).relative_to(ROOT).as_posix()] = digest(FOLDER/(name+'.json'))
        hashes['data/processed/'+name+'.npz'] = digest(ROOT/'data/processed'/(name+'.npz'))
        tr = np.array(parent['train_indices'])
        # Historical outer benchmark fixes three folds even for rare classes.
        # Replay that actual convention, not a newly inferred fold count.
        expected_tr, expected_te = list(StratifiedKFold(3, shuffle=True, random_state=101).split(X, y))[0]
        np.testing.assert_array_equal(tr, expected_tr)
        np.testing.assert_array_equal(parent['test_indices'], expected_te)
        assert len(source['splits']) == len(parent['inner_splits'])
        ri_counts, reference_counts = [], []
        for split, old in zip(source['splits'], parent['inner_splits']):
            ri, vi = np.array(split['reference_indices']), np.array(split['validation_indices'])
            np.testing.assert_array_equal(ri, tr[np.array(old['train'])])
            np.testing.assert_array_equal(vi, tr[np.array(old['validation'])])
            assert set(ri).isdisjoint(vi) and set(ri)|set(vi) == set(tr)
            ri_counts.append(len(ri))
            reference_counts += [len(inner['reference_indices']) for inner in split['inner']]
        ri, a = np.array(mechanism['outer_reference_indices']), np.array(mechanism['reference_indices'])
        np.testing.assert_array_equal(ri, source['splits'][0]['reference_indices'])
        np.testing.assert_array_equal(a, source['splits'][0]['inner'][0]['reference_indices'])
        splits = inner_splits(y[ri[a]], seed=20261027)
        fit_counts = []
        for (ca, cb), fold in zip(splits, mechanism['folds']):
            assert fold['reference_indices'] == ca.tolist() and fold['validation_indices'] == cb.tolist()
            fit_counts.append(len(ca))
        row = {'dataset': name, 'raw_n': len(X), 'd': X.shape[1], 'original_outer_train_n': len(tr),
               'archived_diagnostic_reference_n_range': [min(ri_counts), max(ri_counts)],
               'nested_reference_n_range': [min(reference_counts), max(reference_counts)],
               'first_reference_n': len(a), 'mechanism_training_n_range': [min(fit_counts), max(fit_counts)],
               'mechanism_raw_training_fraction_range': [min(fit_counts)/len(X), max(fit_counts)/len(X)],
               'mechanism_training_class_min_count': min(int(np.unique(y[ri[a[ca]]], return_counts=True)[1].min()) for ca, _ in splits)}
        rows.append(row)
        print(name, 'raw', len(X), 'original training', len(tr), 'diagnostic reference', ri_counts,
              'mechanism training', fit_counts, flush=True)
    result = {'passed': True, 'rows': rows, 'fingerprints': hashes, 'auditor_sha256': digest(Path(__file__)),
              'predictions_scored': False, 'undocumented_sample_cap_found': False,
              'goal_achieved': False,
              'scope': 'original41_outer_training_then_inner_development_reference_then_nested_reference_then_mechanism_fold',
              'interpretation': 'Small mechanism fits result from four training fractions, not full-sized final evaluation; no bug established.'}
    (ROOT/'results/development-sample-depth-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
