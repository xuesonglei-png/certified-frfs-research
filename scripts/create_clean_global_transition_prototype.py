"""Generate explicit new-version sources; never alter frozen robust fits.

Only the optimization loss radius may differ. Every original majority/FRS
constraint still uses radius .01, eta .1 and original retention .9.
"""
import json
import time
from run_simplex_soft5_knn_pilot import ROOT,digest


def replace_once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def main():
    source=ROOT/'scripts/fit_sparse_global_transition_frs.py'
    auditor=ROOT/'scripts/audit_sparse_global_transition_frs.py'
    target=ROOT/'scripts/fit_clean_global_transition_frs.py'
    audit_target=ROOT/'scripts/audit_clean_global_transition_frs.py'
    provenance=ROOT/'research/clean-global-transition-prototype-provenance.json'
    assert not target.exists() and not audit_target.exists() and not provenance.exists()
    raw=source.read_text(encoding='utf-8');text=raw
    changes=[
        ('def fit_sparse_global_transition_target(', 'def fit_clean_global_transition_target('),
        ('sparsity_delta=.01, margin=.02,','sparsity_delta=.01, optimization_eps=0., margin=.02,'),
        ('or not np.isfinite(reg) or reg < 0','or not np.isfinite(optimization_eps) or optimization_eps not in (0., .01)\n        or not np.isfinite(reg) or reg < 0'),
        ('CappedThirdEnemyHinge(A[:, fs], y, initial, margin=margin)','CappedThirdEnemyHinge(A[:, fs], y, initial, margin=margin, eps=optimization_eps)'),
        ("'margin': margin, 'reg': reg,","'margin': margin, 'reg': reg, 'optimization_eps':optimization_eps, 'certificate_eps':cell['eps'],"),
        ("'scope': 'capped_loss_plus_concave_support_penalty_global_FRS_region_prototype_not_prediction'",
         "'scope': 'nominal_or_robust_training_loss_unchanged_eps01_FRS_and_majority_certification_not_prediction'")]
    for old,new in changes:
        text=replace_once(text,old,new)
    audit_text=auditor.read_text(encoding='utf-8')
    audit_changes=[
        ('def check_sparse_global_transition_fit(', 'def check_clean_global_transition_fit('),
        ("assert fit['margin'] == info['margin'] == .02 and info['eps'] == .01 and fit['reg'] == .01",
         "assert fit['margin'] == info['margin'] == .02 and fit['reg'] == .01\n    assert info['eps'] == fit['optimization_eps'] and info['eps'] in (0., .01)\n    assert fit['certificate_eps'] == fit['class_cell']['eps'] == .01")]
    for old,new in audit_changes:
        audit_text=replace_once(audit_text,old,new)
    target.write_text(text,encoding='utf-8');audit_target.write_text(audit_text,encoding='utf-8')
    provenance.write_text(json.dumps({'created_unix':time.time(),'generator_sha256':digest(ROOT/'scripts/create_clean_global_transition_prototype.py'),
        'original_optimizer_sha256':digest(source),'original_auditor_sha256':digest(auditor),
        'new_optimizer_sha256':digest(target),'new_auditor_sha256':digest(audit_target),
        'optimizer_exact_replacements':changes,'auditor_exact_replacements':audit_changes,
        'scope':'new_loss_radius_prototype_only_original_certificates_not_relaxed',
        'real_datasets_fitted':0,'classification_scored':False,'goal_achieved':False},indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
