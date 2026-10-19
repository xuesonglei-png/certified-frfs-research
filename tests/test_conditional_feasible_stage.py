import copy
import numpy as np
import certified_frfs.conditional_feasible_stage as module


def test_failed_group_does_not_block_path_and_restores_until_verified(monkeypatch):
    path = {'trace': [{'features': list(range(k)), 'weights': [1.]*4,
                       'certificate': {'feasible': True}} for k in (4, 3, 2)]}
    full = {'candidate': path['trace'][0], 'fit': {'feasible': True}, 'fallback': False, 'failure': None}
    source = {'d': 4, 'joint_fit': {'feasible': False}, 'teacher_coef': [],
              'teacher_intercept': [], 'teacher_classes': [0, 1], 'teacher_C': 1.,
              'policies': {str(f): copy.deepcopy(full) for f in (.5, .65, .8, 1.)}}
    original = copy.deepcopy(source)
    monkeypatch.setattr(module, 'conditional_backward_path', lambda *a, **k: path)
    calls = []
    def fit(*args, **kwargs):
        fs = args[5]
        calls.append(list(fs))
        return {'feasible': len(fs) == 3}
    monkeypatch.setattr(module, 'fit_constrained_logistic', fit)
    stage = module.refit_feasible_stage(np.arange(24).reshape(6, 4), np.array([0, 1]*3), source)
    assert source == original
    assert calls == [[0, 1], [0, 1, 2]]
    record = stage['policies']['0.5']
    assert record['candidate']['features'] == [0, 1, 2] and not record['fallback']
    assert [r['fit']['feasible'] for r in record['feasibility_repair']] == [False, True]


def test_all_compact_failures_are_retained_with_full_fallback(monkeypatch):
    path = {'trace': [{'features': list(range(k)), 'weights': [1.]*4,
                       'certificate': {'feasible': True}} for k in (4, 3, 2)]}
    full = {'candidate': path['trace'][0], 'fit': {'feasible': True}, 'fallback': False, 'failure': None}
    source = {'d': 4, 'joint_fit': {'feasible': False}, 'teacher_coef': [],
              'teacher_intercept': [], 'teacher_classes': [0, 1], 'teacher_C': 1.,
              'policies': {str(f): copy.deepcopy(full) for f in (.5, .65, .8, 1.)}}
    monkeypatch.setattr(module, 'conditional_backward_path', lambda *a, **k: path)
    monkeypatch.setattr(module, 'fit_constrained_logistic', lambda *a, **k: {'feasible': False})
    stage = module.refit_feasible_stage(np.arange(24).reshape(6, 4), np.array([0, 1]*3), source)
    record = stage['policies']['0.5']
    assert record['fallback'] and len(record['candidate']['features']) == 4
    assert len(record['failure']['attempts']) == 2
