"""Reference-only ceiling for any ranking under the frozen single-delete policy."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/crossfit-label-loss-recent-pilot'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    audit = json.loads((BASE/'complete-audit.json').read_text())
    assert audit['passed']
    rows = []
    for name, sha in sorted(audit['artifact_hashes'].items()):
        path = BASE/(name+'.json')
        assert digest(path) == sha
        saved = json.loads(path.read_text())
        stage, d = saved['outer']['stage'], saved['d']
        target = int(np.ceil(.8*d))
        assert target == stage['target_count']
        full_blocked = False
        if stage['path']['rankings']:
            first = stage['path']['rankings'][0]
            assert first['features'] == list(range(d))
            attempts = stage['path']['attempts'][:d]
            full_blocked = (len(attempts) == d and {a['removed'] for a in attempts} == set(range(d))
                            and all(not a['accepted'] for a in attempts)
                            and all(len(a['features']) == d-1 for a in attempts))
        # No single deletion from full is feasible: no ranking can start.
        lower_count = d if full_blocked else target
        rows.append({'dataset': name, 'd': d, 'target': target,
                     'all_first_deletions_rejected': full_blocked,
                     'any_single_path_count_lower_bound': lower_count,
                     'any_single_path_removal_upper_bound': 1-lower_count/d,
                     'actual_count': stage['actual_count']})
    bound = float(np.mean([r['any_single_path_removal_upper_bound'] for r in rows]))
    result = {'rows': rows, 'mean_removal_upper_bound': bound,
              'fixed_gate_minimum_removal': .1, 'compression_gate_attainable': bound >= .1,
              'reference_audit_sha256': digest(BASE/'complete-audit.json'),
              'scope': 'ranking_independent_frozen_transfer_single_deletion_target_ceiling_not_all_feasible_metrics'}
    (ROOT/'results/single-deletion-compression-ceiling.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result, flush=True)
