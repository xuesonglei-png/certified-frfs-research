"""Guard audited baseline reuse against training/source/audit mismatches."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('label_loss_cache_test_runner', ROOT/'scripts/run_projected_label_loss_recent_pilot.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def audited_cache(tmp_path, monkeypatch):
    name = 'case'
    folder = tmp_path/'results/adaptive-multiknot-recent-pilot'
    folder.mkdir(parents=True)
    audit_source = tmp_path/'scripts/audit_adaptive_multiknot_recent_pilot.py'
    audit_source.parent.mkdir()
    audit_source.write_text('frozen mock auditor', encoding='utf-8')
    algorithm = tmp_path/'src/algorithm.py'
    algorithm.parent.mkdir()
    algorithm.write_text('frozen algorithm', encoding='utf-8')
    X = np.array([[0., 2.], [1., np.nan], [2., 5.], [3., 7.], [4., 8.], [5., 10.], [9., 3.]])
    y = np.array([0, 1, 0, 1, 0, 1, 0])
    paths = {'fgs2023': [{'ranking': [0, 1], 'native': {'features': [0]}}],
             'sfrip2025': [{'ranking': [1, 0], 'native': {'features': [1]}}]}
    costs = {'fgs_grid_seconds': 1., 'sfrip_grid_seconds': 2.}
    saved = {'complete': True, 'query_labels_used_for_selection': False,
             'reference_indices': [0, 1, 2, 3, 4, 5], 'validation_indices': [6],
             'fingerprints': {'src/algorithm.py': runner.digest(algorithm)},
             'inner': [{'reference_indices': [0, 1, 2, 3], 'validation_indices': [4, 5], 'paths': paths, 'costs': costs}],
             'outer': {'paths': paths, 'costs': costs}}
    artifact = folder/(name+'.json')
    artifact.write_text(json.dumps(saved), encoding='utf-8')
    sha = runner.digest(artifact)
    (folder/(name+'-audit.json')).write_text(json.dumps({'fingerprints': {
        'artifact_sha256': sha, 'auditor_sha256': runner.digest(audit_source)}}), encoding='utf-8')
    (folder/'complete-audit.json').write_text(json.dumps({'passed': True, 'artifact_hashes': {name: sha}}), encoding='utf-8')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'load_dataset', lambda requested: (X.copy(), y.copy()) if requested == name else None)
    return name, X, y, paths, algorithm, artifact, folder


@pytest.mark.parametrize('index,ids', [(0, [0, 1, 2, 3]), (None, [0, 1, 2, 3, 4, 5])])
def test_verified_cache_uses_exact_inner_and_outer_training_tables(audited_cache, index, ids):
    name, X, y, paths, *_ = audited_cache
    A, _ = runner.preprocess_fit(X[ids], X[ids])
    actual, costs = runner.verified_cached_paths(name, index, A, y[ids])
    assert actual == paths
    assert costs['baseline_paths_reused'] and not costs['standalone_selector_time_measured_here']


@pytest.mark.parametrize('changed', ['table', 'labels', 'algorithm', 'artifact', 'auditor'])
def test_mismatched_provenance_cannot_be_silently_reused(audited_cache, changed):
    name, X, y, paths, algorithm, artifact, folder = audited_cache
    A, _ = runner.preprocess_fit(X[:4], X[:4])
    labels = y[:4].copy()
    if changed == 'table':
        A[0, 0] += 1e-8
    elif changed == 'labels':
        labels = labels[::-1]
    elif changed == 'algorithm':
        algorithm.write_text('changed source', encoding='utf-8')
    elif changed == 'artifact':
        artifact.write_text(artifact.read_text()+'\n', encoding='utf-8')
    else:
        source = algorithm.parents[1]/'scripts/audit_adaptive_multiknot_recent_pilot.py'
        source.write_text('changed auditor', encoding='utf-8')
    with pytest.raises(AssertionError):
        runner.verified_cached_paths(name, 0, A, labels)
