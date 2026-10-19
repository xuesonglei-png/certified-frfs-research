"""Preserved admission runner with a separate all-class postfilter."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def generated():
    source = (ROOT/'scripts/run_flexible_majority_repair_admission.py').read_text(encoding='utf-8')
    source = source.replace('flexible-majority-repair-admission', 'classwise-repair-admission')
    source = source.replace('from flexible_majority_repair import fit_flexible_majority_repair',
                            'from flexible_majority_repair import fit_flexible_majority_repair\nfrom certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate')
    source = source.replace("ROOT/'research/classwise-repair-admission-protocol.md', artifact,",
                            "ROOT/'research/classwise-repair-admission-protocol.md', ROOT/'scripts/prepare_classwise_repair_admission.py', artifact,")
    marker = "    result = {'dataset': name,"
    source = source.replace(marker, "    _, all_labels = load_dataset(name)\n    original_classes = classwise_dependency_certificate(all_labels[np.array(saved['reference_indices'])], stage['candidate']['certificate'])\n"+marker)
    source = source.replace("'fingerprints': hashes, 'attempts': [], 'complete': False,",
                            "'fingerprints': hashes, 'attempts': [], 'complete': False, 'ordinary_classwise_certificate': original_classes,")
    marker = '            result[\'attempts\'].append('
    insert = '''            if 'weights' in info:
                info['classwise_dependency_certificate'] = classwise_dependency_certificate(y[ri], info['certificate'])
                info['classwise_guard_rejected'] = weights is not None and not info['classwise_dependency_certificate']['feasible_all_classes']
                if info['classwise_guard_rejected']:
                    weights = None
            info['guarded_repair_accepted'] = weights is not None
'''
    assert source.count(marker) == 1
    source = source.replace(marker, insert+marker)
    source = source.replace("'original_certified_repair_selected'", "'original_and_classwise_certified_repair_selected'")
    source = source.replace('all15_reference_only_flexible_weight_admission_not_prediction_superiority',
                            'all15_reference_only_classwise_guarded_repair_admission_not_whole_path_or_prediction_guarantee')
    return source


if __name__ == '__main__':
    target = ROOT/'scripts/run_classwise_repair_admission.py'
    assert not target.exists()
    source = generated()
    compile(source, str(target), 'exec')
    target.write_text(source, encoding='utf-8')
