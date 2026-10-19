"""Generate a versioned common-proper-loss runner and independent auditor."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def generated():
    runner = (ROOT/'scripts/run_certified_diverse_pool_pilot.py').read_text(encoding='utf-8')
    runner = replace_once(runner, 'Registry, metric_scores, choose, METRICS', 'Registry, metric_scores, METRICS')
    runner = runner.replace('certified-diverse-pool-pilot', 'proper-loss-diverse-pool-pilot')
    # The protocol filename uses no -pilot suffix in the new version.
    runner = runner.replace('research/proper-loss-diverse-pool-pilot-protocol.md', 'research/proper-loss-diverse-pool-protocol.md')
    insert = '''
PREDICTION_PARENT = ROOT/'results/certified-diverse-pool-pilot'


def verified_prediction_parent(name):
    artifact = PREDICTION_PARENT/(name+'.json')
    audit = json.loads((PREDICTION_PARENT/'complete-audit.json').read_text())
    assert audit['passed'] and audit['artifact_hashes'][name] == digest(artifact)
    saved = json.loads(artifact.read_text())
    for rel, sha in saved['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    return saved


def cached_registry(A, labels, block, B=None, truth=None):
    registry = Registry(A, labels, B, truth)
    registry.models = block['models'].copy()
    registry.trees = {tuple(t['features']): t for t in block['trees']}
    for i, fit in enumerate(registry.models):
        mi = next(j for j, config in enumerate(MODEL_GRID) if config['C'] == fit['C'] and config['knot_count'] == len(fit['knots']))
        registry.lookup[(tuple(fit['features']), mi)] = i
    if B is not None:
        registry.evaluations = block['evaluations'].copy()
    return registry


def choose(rows):
    for row in rows:
        row['mean_balanced_logloss'] = float(np.mean([v['balanced_logloss'] for v in row['folds']]))
        row['mean_clean_BA'] = float(np.mean([v['clean_BA'] for v in row['folds']]))
        row['mean_actual_count'] = float(np.mean([v['actual_count'] for v in row['folds']]))
    return min(range(len(rows)), key=lambda i: (round(rows[i]['mean_balanced_logloss'], 12),
        -round(rows[i]['mean_clean_BA'], 12), rows[i]['mean_actual_count'], rows[i]['selector_index'], rows[i]['model_index']))

'''
    runner = replace_once(runner, '\ndef options(method):', insert+'\ndef options(method):')
    runner = replace_once(runner, "files += sorted((ROOT/'src/certified_frfs').glob('*.py'))", "files += [PREDICTION_PARENT/(name+'.json'), PREDICTION_PARENT/'complete-audit.json', ROOT/'scripts/prepare_proper_loss_pool_pilot.py']\n    files += sorted((ROOT/'src/certified_frfs').glob('*.py'))")
    runner = replace_once(runner, "'clean_BA': registry.evaluations[str(mid)]['metrics']['clean_BA']}", "'clean_BA': registry.evaluations[str(mid)]['metrics']['clean_BA'],\n                         'balanced_logloss': registry.evaluations[str(mid)]['metrics']['balanced_logloss']}")
    runner = replace_once(runner, 'base, own = verified_parent(BASE, name), verified_parent(OOF, name)', 'base, own = verified_parent(BASE, name), verified_parent(OOF, name)\n    prediction_parent = verified_prediction_parent(name)')
    runner = replace_once(runner, "pool = admit(raw[ca], labels[ca], recipes)", "parent_block = prediction_parent['inner'][fi]\n        assert parent_block['reference_indices'] == ca.tolist() and parent_block['validation_indices'] == cb.tolist()\n        assert recipes == parent_block['recipes']\n        pool = parent_block['pool']")
    runner = replace_once(runner, 'registry = Registry(A, labels[ca], B, labels[cb])', 'registry = cached_registry(A, labels[ca], parent_block, B, labels[cb])')
    runner = replace_once(runner, 'registry = Registry(A, labels)', "registry = cached_registry(A, labels, prediction_parent['outer'])")
    auditor = (ROOT/'scripts/audit_certified_diverse_pool_pilot.py').read_text(encoding='utf-8')
    auditor = replace_once(auditor, 'from run_certified_diverse_pool_pilot import', 'from run_proper_loss_diverse_pool_pilot import')
    begin, end = auditor.index('def independent_choose(rows):'), auditor.index('\ndef audit(name):')
    independent = '''def independent_choose(rows):
    losses = [float(np.mean([f['balanced_logloss'] for f in r['folds']])) for r in rows]
    means = [(float(np.mean([f['clean_BA'] for f in r['folds']])), float(np.mean([f['actual_count'] for f in r['folds']]))) for r in rows]
    for r, loss, (ba, size) in zip(rows, losses, means):
        equivalent(r['mean_balanced_logloss'], loss)
        equivalent(r['mean_clean_BA'], ba); equivalent(r['mean_actual_count'], size)
    return min(range(len(rows)), key=lambda i: (round(losses[i], 12), -round(means[i][0], 12), means[i][1], rows[i]['selector_index'], rows[i]['model_index']))

'''
    auditor = auditor[:begin]+independent+auditor[end:]
    auditor = replace_once(auditor, "'clean_BA': evaluations[str(mid)]['metrics']['clean_BA']}", "'clean_BA': evaluations[str(mid)]['metrics']['clean_BA'], 'balanced_logloss': evaluations[str(mid)]['metrics']['balanced_logloss']}")
    return {'scripts/run_proper_loss_diverse_pool_pilot.py': runner, 'scripts/audit_proper_loss_diverse_pool_pilot.py': auditor}


if __name__ == '__main__':
    for name, source in generated().items():
        compile(source, name, 'exec')
        (ROOT/name).write_text(source, encoding='utf-8')
    print('Generated proper-loss runner and independent auditor.')
