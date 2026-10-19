"""An alternative global region must not be treated as always feasible."""
from collections import Counter
import copy
import numpy as np
import pytest
from frs_coordinate_minimum import solve_diagnostic
from sparse_global_frs_training_ablation import fit_block
from audit_sparse_global_training_block import audit_block


def test_unavailable_global_anchor_is_recorded_without_search_and_replayed_independently():
    rng = np.random.default_rng(9139)
    for _ in range(44):
        A = rng.uniform(0, 1, (12, 3))
    y = np.repeat([0, 1], 6); fs = [0, 2]; anchor = np.ones(2)*np.sqrt(3)/2
    diagnostic = solve_diagnostic(A, y, fs, anchor)
    block = fit_block(A, y, fs, anchor, anchor, diagnostic)
    assert block['fixed_probe']['anchor_strictly_feasible']
    assert not block['global_probe']['anchor_strictly_feasible']
    assert block['arms']['fixed_signed_risk']['available']
    for arm in ('global_signed_risk', 'global_signed_sparse'):
        result = block['arms'][arm]
        assert not result['available'] and result['optimizer_success'] is None
        assert result['fits'] == {} and result['selected_start'] is None
        assert result['target'] == anchor.tolist()
        assert block['evaluation']['optimizer_failures'][arm] == 0
    counts = Counter(); audit_block(A, y, block, diagnostic, counts)
    assert counts['independent_dc_fit_replays'] == 3
    assert counts['global_signed_sparse_unavailable'] == 1
    tampered = copy.deepcopy(block)
    tampered['arms']['global_signed_sparse']['optimizer_success'] = True
    with pytest.raises(AssertionError):
        audit_block(A, y, tampered, diagnostic, Counter())
