"""Balanced objective replay with aligned repair source consistency."""
import json
import audit_nested_balanced_joint as original

original.grouped.repair.original.FOLDER = original.grouped.repair.original.ROOT/'results/nested-balanced-joint-aligned-repair'

if __name__ == '__main__':
    folder = original.grouped.repair.original.FOLDER
    summary = json.loads((folder/'complete.json').read_text())
    for row in summary['rows']:
        saved = json.loads((folder/(row['dataset']+'.json')).read_text())
        source = json.loads((folder.parent/'nested-balanced-joint'/(row['dataset']+'.json')).read_text())
        for split, prior in zip(saved['splits'], source['splits']):
            assert split['full_config'] == prior['full_config']
            assert split['reference_indices'] == prior['reference_indices']
            assert split['validation_indices'] == prior['validation_indices']
            for inner, previous in zip(split['inner'], prior['inner']):
                assert inner['reference_indices'] == previous['reference_indices']
                assert inner['validation_indices'] == previous['validation_indices']
                for C, record in inner['by_C'].items():
                    stage, old = record['stage'], previous['by_C'][C]['stage']
                    assert stage['repair_target'] == 'balanced_joint_scores'
                    for key in ('teacher_coef', 'teacher_intercept', 'joint_fit'):
                        assert stage[key] == old[key]
                    assert stage['policies']['1.0'] == old['policies']['1.0']
    original.grouped.repair.original.main()
