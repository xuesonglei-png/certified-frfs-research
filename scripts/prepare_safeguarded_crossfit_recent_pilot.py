"""Separate guarded stage/runner/auditor; preserve earlier failed variants."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def generated():
    stage = (ROOT/'src/certified_frfs/repaired_crossfit_stage.py').read_text(encoding='utf-8')
    stage = stage.replace('def fit_repaired_crossfit_stage(', 'def fit_safeguarded_crossfit_stage(')
    stage = stage.replace('from .flexible_majority_repair import fit_flexible_majority_repair',
                          'from .flexible_majority_repair import fit_flexible_majority_repair\nfrom .classwise_dependency_certificate import classwise_dependency_certificate')
    helper = '''
def safeguarded_metric_repair(A, y, fs, *, time_limit):
    weights, info = fit_flexible_majority_repair(A, y, fs, time_limit=time_limit)
    if 'weights' in info:
        guarded = classwise_dependency_certificate(y, info['certificate'])
        info = {**info, 'classwise_dependency_certificate': guarded}
        if weights is not None and not guarded['feasible_all_classes']:
            return None, {**info, 'status': 'classwise_guard_failed'}
    return weights, info

'''
    stage = stage.replace('\n\ndef fit_safeguarded_crossfit_stage(', '\n'+helper+'\ndef fit_safeguarded_crossfit_stage(')
    stage = stage.replace('weights, info = fit_flexible_majority_repair(context.A, y, fs, time_limit=10.)',
                          'weights, info = safeguarded_metric_repair(context.A, y, fs, time_limit=10.)')
    stage = stage.replace('original_FRS_reference_certificate_OOF_flexible_metric_repair_separate_classifier_not_test_risk',
                          'original_FRS_reference_certificate_OOF_classwise_guarded_repairs_only_separate_classifier_not_test_risk')
    source = (ROOT/'scripts/run_repaired_crossfit_recent_pilot.py').read_text(encoding='utf-8')
    source = source.replace('repaired-crossfit-recent-pilot', 'safeguarded-crossfit-recent-pilot')
    source = source.replace('certified_frfs.repaired_crossfit_stage import fit_repaired_crossfit_stage',
                            'certified_frfs.safeguarded_crossfit_stage import fit_safeguarded_crossfit_stage')
    source = source.replace('fit_repaired_crossfit_stage(', 'fit_safeguarded_crossfit_stage(')
    source = source.replace("ROOT/'research/safeguarded-crossfit-recent-pilot-protocol.md',",
                            "ROOT/'research/safeguarded-crossfit-recent-pilot-protocol.md', ROOT/'scripts/prepare_safeguarded_crossfit_recent_pilot.py',")
    source = source.replace('original_FRS_repaired_crossfit_and_tuned_recent', 'original_FRS_safeguarded_repairs_and_tuned_recent')
    audit = (ROOT/'scripts/audit_repaired_crossfit_recent_pilot_v2.py').read_text(encoding='utf-8')
    audit = audit.replace('from run_repaired_crossfit_recent_pilot import', 'from run_safeguarded_crossfit_recent_pilot import')
    audit = audit.replace('from audit_flexible_majority_repair_admission import check_repair',
                          'from audit_flexible_majority_repair_admission import check_repair\nfrom audit_classwise_repair_admission import independent_class_certificate')
    marker = "                    equivalent(repaired['certificate'], info['certificate'])"
    assert audit.count(marker) == 1
    audit = audit.replace(marker, marker+"\n                    guard = independent_class_certificate(y, repaired['certificate'])\n                    equivalent(guard, info['classwise_dependency_certificate'])\n                    counts['independent_repair_class_aggregations'] += 1")
    marker = "                        assert repaired['certificate']['feasible'] and record is repair['attempts'][-1]"
    audit = audit.replace(marker, marker+"\n                        assert guard['feasible_all_classes']")
    audit = audit.replace("                    assert not repaired['certificate']['feasible']",
        "                    if info['status'] == 'classwise_guard_failed':\n                        assert repaired['certificate']['feasible'] and not guard['feasible_all_classes']\n                        counts['classwise_guard_rejections'] += 1\n                    else:\n                        assert not repaired['certificate']['feasible']")
    audit = audit.replace("(name+'-audit-v2.json')", "(name+'-audit.json')")
    audit = audit.replace("FOLDER/'complete-audit-v2.json'", "FOLDER/'complete-audit.json'")
    return {'src/certified_frfs/safeguarded_crossfit_stage.py': stage,
            'scripts/run_safeguarded_crossfit_recent_pilot.py': source,
            'scripts/audit_safeguarded_crossfit_recent_pilot.py': audit}


if __name__ == '__main__':
    for name, source in generated().items():
        target = ROOT/name
        assert not target.exists(), 'preserve existing source'
        compile(source, str(target), 'exec')
        target.write_text(source, encoding='utf-8')
