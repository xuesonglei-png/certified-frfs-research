from check_clipped_redistribution_counterexample import run


def test_dependency_preservation_does_not_imply_classification_preservation():
    result = run()
    assert result['passed']
    assert not result['production_guard']['feasible']
    assert result['query_checks']['reduced']['correct'] == [False, False, True]
    assert result['query_checks']['reduced']['certified'] == [True]*3
