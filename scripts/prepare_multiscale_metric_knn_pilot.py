"""Versioned47recipe runner/auditor; no change to frozen earlier experiments."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def generated():
    source = (ROOT/'scripts/run_certified_metric_knn_pool.py').read_text(encoding='utf-8')
    source = source.replace('certified-metric-knn-pool', 'multiscale-certified-metric-knn-pool')
    source = source.replace("'proposed': 17, 'unguarded_pool': 17", "'proposed': 47, 'unguarded_pool': 47")
    source = once(source, "'fgs_native', 'sfrip_native',\n           'fgs_matched'", "'fgs_native', 'sfrip_native', 'raw_repair_pool',\n           'fgs_matched'")
    source = once(source, "'fgs_native': 9, 'sfrip_native': 6,", "'fgs_native': 9, 'sfrip_native': 6, 'raw_repair_pool': 47,")
    source = source.replace('METHODS[:5]', 'METHODS[:6]').replace('METHODS[5:]', 'METHODS[6:]')
    source = once(source, "ordinary = r['ordinary']; fs = ordinary['features']", "ordinary = r['ordinary']; fs = ordinary['features']\n        info = r['repair']\n        rw = info['weights'] if info is not None and 'weights' in info else np.array(ordinary['weights'])[fs].tolist()\n        output['raw_repair_pool'].append({'features': fs, 'weights': rw, 'origin': 'raw_normalized_repair_metric' if info is not None and 'weights' in info else 'ordinary_uncensored_metric'})")
    insert = '''
from run_recent_multiscale_pool_admission import recipes as scale_recipes, admission as scale_admission
SCALE_POOL = ROOT/'results/recent-multiscale-pool-admission'


def multiscale_parent(name, old, raw, labels):
    artifact = SCALE_POOL/(name+'.json')
    audit = json.loads((SCALE_POOL/'complete-audit.json').read_text())
    assert audit['passed'] and audit['artifact_hashes'][name] == digest(artifact)
    outer = json.loads(artifact.read_text())
    assert outer['reference_indices'] == old['reference_indices']
    for rel, sha in outer['fingerprints'].items(): assert digest(ROOT/rel) == sha, rel
    own = json.loads((ROOT/'results/crossfit-label-loss-recent-pilot'/(name+'.json')).read_text())
    result = {**old, 'inner': []}
    for fi, block in enumerate(old['inner']):
        ca = np.array(block['reference_indices'])
        candidates = scale_recipes(block['paths'], own['inner'][fi]['stage']['candidate']['features'], raw.shape[1])
        pool = scale_admission(raw[ca], labels[ca], candidates, block['pool'])
        result['inner'].append({**block, 'recipes': candidates, 'pool': pool})
        print(name, 'inner', fi, 'multiscale pool', sum(r['accepted'] for r in pool), '/', len(pool), flush=True)
    result['outer'] = {**old['outer'], 'recipes': outer['recipes'], 'pool': outer['pool']}
    return result

'''
    source = once(source, '\ndef evaluate(A, y, B, truth, fs, weights):', insert+'\ndef evaluate(A, y, B, truth, fs, weights):')
    source = once(source, "files += sorted((ROOT/'src/certified_frfs').glob('*.py'))", "files += [SCALE_POOL/(name+'.json'), SCALE_POOL/'complete-audit.json', ROOT/'scripts/run_recent_multiscale_pool_admission.py', ROOT/'scripts/prepare_multiscale_metric_knn_pilot.py']\n    files += sorted((ROOT/'src/certified_frfs').glob('*.py'))")
    source = once(source, 'raw, labels = X[ri], y[ri]', 'raw, labels = X[ri], y[ri]\n    parent = multiscale_parent(name, parent, raw, labels)')
    source = once(source, "'inner': inners, 'selection':", "'candidate_reference_blocks': [{k: block[k] for k in ('reference_indices', 'validation_indices', 'recipes', 'pool', 'paths')} for block in parent['inner']]+[{k: parent['outer'][k] for k in ('recipes', 'pool', 'paths')}],\n             'inner': inners, 'selection':")
    auditor = (ROOT/'scripts/audit_certified_metric_knn_pool.py').read_text(encoding='utf-8')
    auditor = once(auditor, 'from run_certified_metric_knn_pool import', 'from run_multiscale_certified_metric_knn_pool import')
    auditor = once(auditor, 'ROOT, FOLDER, PARENT, METHODS, COUNTS, digest', 'ROOT, FOLDER, PARENT, METHODS, COUNTS, digest, SCALE_POOL')
    auditor = once(auditor, 'from audit_forest_compact_transfer import equivalent', 'from audit_forest_compact_transfer import equivalent\nfrom audit_certified_diverse_pool_pilot import check_pool')
    auditor = once(auditor, "out['unguarded_pool'].append({'features': fs, 'weights': np.array(r['ordinary']['weights'])[fs].tolist(), 'origin': 'ordinary_uncensored_metric'})", "out['unguarded_pool'].append({'features': fs, 'weights': np.array(r['ordinary']['weights'])[fs].tolist(), 'origin': 'ordinary_uncensored_metric'})\n        info = r['repair']\n        repair_weights = info['weights'] if info is not None and 'weights' in info else np.array(r['ordinary']['weights'])[fs].tolist()\n        out['raw_repair_pool'].append({'features': fs, 'weights': repair_weights, 'origin': 'raw_normalized_repair_metric' if info is not None and 'weights' in info else 'ordinary_uncensored_metric'})")
    validation = '''
def validated_multiscale_parent(name, old, saved, raw, labels, counts):
    own = json.loads((ROOT/'results/crossfit-label-loss-recent-pilot'/(name+'.json')).read_text())
    blocks = saved['candidate_reference_blocks']
    assert len(blocks) == len(old['inner'])+1
    outer_path = SCALE_POOL/(name+'.json'); outer = json.loads(outer_path.read_text())
    audit = json.loads((SCALE_POOL/'complete-audit.json').read_text())
    assert audit['passed'] and audit['artifact_hashes'][name] == digest(outer_path)
    result = {**old, 'inner': []}
    for fi, (block, source) in enumerate(zip(blocks, old['inner']+[old['outer']])):
        is_outer = fi == len(old['inner'])
        equivalent(block['paths'], source['paths'])
        if is_outer:
            reference, y = raw, labels
            original = own['outer']['stage']['candidate']['features']
        else:
            assert block['reference_indices'] == source['reference_indices'] and block['validation_indices'] == source['validation_indices']
            ca = np.array(source['reference_indices']); reference, y = raw[ca], labels[ca]
            original = own['inner'][fi]['stage']['candidate']['features']
        recipes = []
        for method in ('fgs2023', 'sfrip2025'):
            for index, path in enumerate(source['paths'][method]):
                for fraction in (.5, .65, .8):
                    recipes.append({'features': sorted(path['ranking'][:int(np.ceil(fraction*raw.shape[1]))]),
                                    'origin': {'method': method, 'config_index': index, 'fraction': fraction}})
        recipes += [{'features': sorted(original), 'origin': {'method': 'OOF_ordinary_endpoint'}},
                    {'features': list(range(raw.shape[1])), 'origin': {'method': 'full_fallback'}}]
        equivalent(recipes, block['recipes']); assert len(recipes) == 47
        A, _ = preprocess_fit(reference, reference)
        check_pool(A, y, recipes, block['pool'], counts)
        cache = {tuple(r['features']): r for r in source['pool']}
        for record in block['pool']:
            fs = tuple(record['features'])
            assert record['cache_reused'] == (fs in cache)
            if fs in cache:
                equivalent(record, {**cache[fs], 'origins': record['origins'], 'cache_reused': True})
        if is_outer:
            equivalent(block['pool'], outer['pool']); equivalent(block['recipes'], outer['recipes'])
            result['outer'] = {**source, **block}
        else:
            result['inner'].append({**source, **block})
    return result

'''
    auditor = once(auditor, '\ndef audit(name):', validation+'\ndef audit(name):')
    auditor = once(auditor, 'raw, labels = X[ri], y[ri]; counts = Counter(datasets=1)', 'raw, labels = X[ri], y[ri]; counts = Counter(datasets=1)\n    parent = validated_multiscale_parent(name, parent, saved, raw, labels, counts)')
    return {'scripts/run_multiscale_certified_metric_knn_pool.py': source, 'scripts/audit_multiscale_certified_metric_knn_pool.py': auditor}


if __name__ == '__main__':
    for name, source in generated().items():
        compile(source, name, 'exec'); (ROOT/name).write_text(source, encoding='utf-8')
    print('Generated47recipe metric-linked runner and auditor.')
