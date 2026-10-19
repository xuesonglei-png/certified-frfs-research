"""Balanced-teacher refit plus scalar margins/objectives/path replay."""
from sklearn.linear_model import LogisticRegression as ActualLogisticRegression
import audit_nested_joint_backward as original

audit = original.original.grouped.repair.original
audit.FOLDER = audit.ROOT/'results/nested-balanced-teacher-backward'


def balanced_teacher(**kwargs):
    return ActualLogisticRegression(**kwargs, class_weight='balanced')


audit.LogisticRegression = balanced_teacher
base_audit = audit.audit_stage


def audit_stage(raw, y, query, truth, stage, scores, counts, *, seed=None):
    assert stage['teacher_class_balanced']
    base_audit(raw, y, query, truth, stage, scores, counts, seed=seed)
    counts['balanced_teachers_rebuilt'] = counts.get('balanced_teachers_rebuilt', 0)+1


audit.audit_stage = audit_stage

if __name__ == '__main__':
    audit.main()
