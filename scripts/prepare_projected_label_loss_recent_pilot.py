"""Generate preserved-setting runner/auditor, with audited baseline caching."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CACHE_FUNCTION = '''
def verified_cached_paths(name, inner_index, A, train):
    base = ROOT/'results/adaptive-multiknot-recent-pilot'
    artifact = base/(name+'.json')
    saved = json.loads(artifact.read_text())
    audit = json.loads((base/(name+'-audit.json')).read_text())
    complete = json.loads((base/'complete-audit.json').read_text())
    assert complete['passed'] and complete['artifact_hashes'][name] == digest(artifact)
    assert audit['fingerprints']['artifact_sha256'] == digest(artifact)
    assert audit['fingerprints']['auditor_sha256'] == digest(ROOT/'scripts/audit_adaptive_multiknot_recent_pilot.py')
    for rel, sha in saved['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    X, y = load_dataset(name)
    ri = np.array(saved['reference_indices'])
    if inner_index is None:
        block, ref, labels = saved['outer'], X[ri], y[ri]
    else:
        block = saved['inner'][inner_index]
        a = np.array(block['reference_indices'])
        ref, labels = X[ri[a]], y[ri[a]]
    expected, _ = preprocess_fit(ref, ref)
    np.testing.assert_array_equal(A, expected)
    np.testing.assert_array_equal(train, labels)
    costs = {key: block['costs'][key] for key in ('fgs_grid_seconds', 'sfrip_grid_seconds')}
    costs.update({'baseline_paths_reused': True, 'standalone_selector_time_measured_here': False})
    return block['paths'], costs

'''


if __name__ == '__main__':
    source = (ROOT/'scripts/run_adaptive_multiknot_recent_pilot.py').read_text(encoding='utf-8')
    source = source.replace("FOLDER = ROOT/'results/adaptive-multiknot-recent-pilot'",
                            "FOLDER = ROOT/'results/projected-label-loss-recent-pilot'")
    source = source.replace('from certified_frfs.adaptive_multiknot_stage import fit_adaptive_certified_stage, MODEL_GRID',
        'from certified_frfs.adaptive_multiknot_stage import MODEL_GRID\nfrom certified_frfs.label_loss_certified_stage import fit_label_loss_certified_stage')
    source = source.replace('fit_adaptive_certified_stage(', 'fit_label_loss_certified_stage(')
    source = source.replace("ROOT/'research/adaptive-multiknot-recent-pilot-protocol.md',",
        "ROOT/'research/adaptive-multiknot-recent-pilot-protocol.md', ROOT/'research/projected-label-loss-next-protocol.md',\n             ROOT/'results/adaptive-multiknot-recent-pilot'/(name+'.json'),\n             ROOT/'results/adaptive-multiknot-recent-pilot'/(name+'-audit.json'),\n             ROOT/'results/adaptive-multiknot-recent-pilot/complete-audit.json',\n             ROOT/'scripts/prepare_projected_label_loss_recent_pilot.py',")
    source = source.replace('def run(name):', CACHE_FUNCTION+'def run(name):')
    assert source.count('paths, costs = path_grid(A, labels[ca])') == 1
    source = source.replace('paths, costs = path_grid(A, labels[ca])', 'paths, costs = verified_cached_paths(name, fi, A, labels[ca])')
    assert source.count('paths, costs = path_grid(A, labels)') == 1
    source = source.replace('paths, costs = path_grid(A, labels)', 'paths, costs = verified_cached_paths(name, None, A, labels)')
    source = source.replace('original_FRS_and_tuned_recent', 'original_FRS_label_loss_and_tuned_recent')
    target = ROOT/'scripts/run_projected_label_loss_recent_pilot.py'
    assert not target.exists(), 'preserve generated source'
    target.write_text(source, encoding='utf-8')
    auditor = (ROOT/'scripts/audit_adaptive_multiknot_recent_pilot.py').read_text(encoding='utf-8')
    auditor = auditor.replace('from run_adaptive_multiknot_recent_pilot import', 'from run_projected_label_loss_recent_pilot import')
    old = "    difference = independent_scores(A, teacher)-independent_scores(raw, teacher)\n    difference -= difference.mean(axis=1, keepdims=True)\n    weights = np.array([1/(len(np.unique(y))*sum(y == c)) for c in y])\n    return float(weights@np.mean(difference**2, axis=1))"
    new = "    value = independent_scores(raw, teacher)\n    own = np.searchsorted(teacher['classes'], y)\n    weights = np.array([1/(len(np.unique(y))*sum(y == c)) for c in y])\n    return float(weights@(logsumexp(value, axis=1)-value[np.arange(len(y)), own]))"
    assert auditor.count(old) == 1
    auditor = auditor.replace(old, new).replace('projection_loss', 'projected_balanced_logloss')
    # Each support's independently reconstructed tree is identical across
    # C/knots. Cache by exact training values and labels, not by data name.
    auditor = auditor.replace('import argparse\n', 'import argparse\nimport hashlib\n')
    auditor = auditor.replace('interaction_tree(', 'cached_tree(')
    marker = '\n\ndef fresh_model('
    helper = '\n\nTREE_CACHE = {}\n\ndef cached_tree(X, y):\n    key = (X.shape, hashlib.sha256(X.tobytes()).digest(), hashlib.sha256(y.tobytes()).digest())\n    if key not in TREE_CACHE:\n        TREE_CACHE[key] = interaction_tree(X, y)\n    return TREE_CACHE[key]\n'
    assert auditor.count(marker) == 1
    auditor = auditor.replace(marker, helper+marker)
    target = ROOT/'scripts/audit_projected_label_loss_recent_pilot.py'
    assert not target.exists(), 'preserve generated source'
    target.write_text(auditor, encoding='utf-8')
