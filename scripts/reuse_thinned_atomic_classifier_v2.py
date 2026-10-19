"""Explicit, verified raw reuse; never claims old failed cohort was audited."""
import json
import copy
from collections import Counter
import numpy as np
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from run_thinned_atomic_classifier import ROOT, FOLDER as OLD, EXACT, digest
from run_multiscale_certified_metric_knn_pool import policy_metrics
from exact_count_anchor_v2 import exact_count_policies


def validate_legacy(name, old):
    assert old['complete'] and not old['query_labels_used_for_selection']
    for rel, sha in old['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    from run_thinned_atomic_classifier import POOL
    parent = json.loads((POOL/(name+'.json')).read_text())
    X, y = load_dataset(name)
    raw = X[np.array(old['reference_indices'])]
    checked = 0
    for bi, block in enumerate(old['inner']+[old['outer']]):
        is_outer = bi == len(old['inner'])
        train = raw if is_outer else raw[np.array(block['reference_indices'])]
        A, _ = preprocess_fit(train, train)
        count = block['exact_proposed_count']
        assert 2*count >= A.shape[1], 'legacy artifact outside legacy feasible domain'
        raw_block = parent['candidate_reference_blocks'][bi]
        legacy = policy_metrics(A, raw_block, count=count)
        new = exact_count_policies(A, raw_block, count=count)
        for method, key in EXACT.items():
            assert legacy[key] == new[key], 'anchor changed'
            rows = [block['results'][method]] if is_outer else block['options'][method]
            for row in rows:
                base = new[key][row['selector_index']]
                metric = row['metric']
                assert metric['features'] == base['features']
                assert metric['base_origin'] == base['origin']
                if metric['learner_id'] is not None:
                    learner = block['exact_stage']['learners'][metric['learner_id']]
                    assert learner['anchor'] == base['weights']
                else:
                    assert metric['weights'] == base['weights']
                checked += 1
    return checked


def validate_reused_artifact(name, saved):
    provenance = saved['raw_reuse_provenance']
    path = OLD/(name+'.json')
    assert provenance['sha256'] == digest(path)
    assert provenance['source'] == path.relative_to(ROOT).as_posix()
    original = json.loads(path.read_text())
    assert provenance['verified_anchor_options'] == validate_legacy(name, original)
    clone = copy.deepcopy(saved)
    clone.pop('raw_reuse_provenance')
    clone.pop('control_recipe_version')
    clone['fingerprints'] = original['fingerprints']
    clone['scope'] = original['scope']
    assert clone == original, 'raw evidence changed beyond declared provenance'


def main():
    from run_thinned_atomic_classifier_v2 import FOLDER, fingerprints
    FOLDER.mkdir(exist_ok=True)
    manifest = []
    for path in sorted(OLD.glob('*.json')):
        if path.stem.startswith('complete'):
            continue
        old = json.loads(path.read_text())
        checked = validate_legacy(path.stem, old)
        new = copy.deepcopy(old)
        new['fingerprints'] = fingerprints(path.stem)
        new['control_recipe_version'] = 2
        new['scope'] = 'all15_exposed_development_thinned_atomic_v2_minimal_feasible_exact_count_anchors'
        new['raw_reuse_provenance'] = {'source': path.relative_to(ROOT).as_posix(),
            'sha256': digest(path), 'verified_anchor_options': checked,
            'status': 'raw_only_full_v2_independent_audit_required'}
        validate_reused_artifact(path.stem, new)
        target = FOLDER/path.name
        if target.exists():
            assert json.loads(target.read_text()) == new
        else:
            target.write_text(json.dumps(new,separators=(',',':')),encoding='utf-8')
        manifest.append({'dataset':path.stem,**new['raw_reuse_provenance']})
        print(path.stem,'verified equivalent raw reuse',checked,flush=True)
    (FOLDER/'raw-reuse-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
