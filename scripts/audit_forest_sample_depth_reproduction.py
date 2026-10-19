"""Run the independent auditor and compare every replayed model/prediction."""
import json
import runpy
from pathlib import Path
from run_nested_joint_path import ROOT, digest
from audit_forest_compact_transfer import equivalent
import diagnose_forest_sample_depth as frozen
import diagnose_forest_sample_depth_reproduction as replay


if __name__ == '__main__':
    original_folder = frozen.FOLDER
    frozen.FOLDER = replay.FOLDER
    runpy.run_path(str(ROOT/'scripts/audit_forest_sample_depth.py'), run_name='__main__')
    old = json.loads((original_folder/'complete.json').read_text())
    new = json.loads((replay.FOLDER/'complete.json').read_text())
    equivalent(old, new)
    rows = []
    for item in old['rows']:
        name = item['dataset']
        a = json.loads((original_folder/(name+'.json')).read_text())
        b = json.loads((replay.FOLDER/(name+'.json')).read_text())
        assert len(a['records']) == len(b['records']) == 9
        for original, fresh in zip(a['records'], b['records']):
            for key in ('outer', 'fraction', 'outer_train_indices', 'reference_indices',
                        'validation_indices', 'tree', 'models', 'evaluations'):
                equivalent(original[key], fresh[key])
        # A second call must now return the verified completed cache safely.
        before = digest(replay.FOLDER/(name+'.json'))
        replay.run(name)
        assert digest(replay.FOLDER/(name+'.json')) == before
        rows.append({'dataset': name, 'models_predictions_equivalent': True, 'completed_cache_replay_passed': True})
    result = {'passed': True, 'rows': rows, 'auditor_sha256': digest(Path(__file__)),
              'generator_sha256': digest(ROOT/'scripts/prepare_forest_sample_depth_reproduction.py'),
              'scope': 'same270_models_and_predictions_checkpoint_fingerprint_repair_not_new_effect',
              'goal_achieved': False}
    (replay.FOLDER/'reproduction-equivalence-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result, flush=True)
