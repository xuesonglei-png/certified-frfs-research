"""Dynamic pool and entropy orientation differ from static singleton ranking."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from mafrfs_conditional_search import select_conditional_mafrfs
from mafrfs_relation_measures import fuzzy_labels_from_full_relation, fuzzy_dependency, fuzzy_conditional_entropy


def test_dynamic_pool_rebuilds_context_and_handles_opposite_measure_orientation():
    values = {(0,): 4, (1,): 3, (2,): 2, (3,): 1,
              (0, 1): 4.1, (0, 2): 8, (0, 3): 4.2, (1, 2): 10, (2, 3): 7,
              (0, 1, 2): 10.5, (1, 2, 3): 11}
    margins = {(0,): 1, (2,): .1, (1, 2): .2, (2, 3): .05}
    A = np.zeros((4, 4)); y = [0, 0, 1, 1]
    traces = []
    for maximize in (True, False):
        score = lambda fs: values[fs] if maximize else 100-values[fs]
        selected, report = select_conditional_mafrfs(A, y, 2, subset_measure=score,
            maximize=maximize, sop=2, margin_type='local', pool_boundary='truncate',
            margin_evaluator=lambda fs: margins[fs])
        assert selected == [2, 3]
        assert report['trace'][0]['pool'] == [0, 2]  # Static top2 would be [0,1].
        assert report['trace'][1]['pool'] == [1, 3]
        assert report['trace'][1]['pool_trace'][0]['base'] == [2]
        assert report['trace'][1]['pool_trace'][1]['base'] == [1, 2]
        traces.append([q['selected_feature'] for q in report['trace']])
    assert traces[0] == traces[1]


def test_relation_equations_feed_dynamic_engine_with_declared_conventions():
    A = np.array([[0., 0.], [.1, .8], [.8, .2], [1., 1.]])
    y = np.array([0, 0, 1, 1])
    # This declared illustrative kernel is NOT an author-specific kernel.
    relation = lambda fs: 1-np.max(abs(A[:, None, fs]-A[None, :, fs]), axis=-1)
    _, labels = fuzzy_labels_from_full_relation(relation((0, 1)), y)
    measures = [(True, lambda fs: fuzzy_dependency(relation(fs), labels, t_conorm=np.maximum, negation=lambda x: 1-x)),
                (False, lambda fs: fuzzy_conditional_entropy(relation(fs), np.eye(4), log_base=2))]
    for maximize, measure in measures:
        selected, report = select_conditional_mafrfs(A, y, 2, subset_measure=measure, maximize=maximize,
            sop=2, margin_type='local', pool_boundary='truncate')
        assert sorted(selected) == [0, 1] and len(report['trace']) == 2
        assert report['trace'][-1]['pool'] == [selected[-1]]
    with pytest.raises(ValueError):
        select_conditional_mafrfs(A, y, 2, subset_measure=measures[0][1], maximize=True,
            sop=2, margin_type='local', pool_boundary='error')
