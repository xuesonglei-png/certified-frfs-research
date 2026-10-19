import json
import numpy as np
from pathlib import Path
from verified_global_cell_v2 import evaluate_class_cell,repair_cell_metric
from independent_global_frs_transition import scalar_cell_evaluation


def test_real_rejected_boundary_is_repaired_without_relaxing_eta():
    root=Path(__file__).resolve().parents[1]
    saved=json.loads((root/'results/sparse-global-frs-training/ecoli.json').read_text())
    record=saved['block']['arms']['global_signed_sparse']['fits']['anchor'];fit=record['fit']
    cell=fit['class_cell'];candidate=np.asarray(fit['steps'][4]['candidate_weights'])
    assert not scalar_cell_evaluation(cell,candidate)['feasible']
    assert not evaluate_class_cell(cell,candidate)['feasible']
    repaired,trace,fraction=repair_cell_metric(np.empty((0,len(candidate))),np.empty(0),cell,candidate,np.asarray(record['anchor']))
    assert scalar_cell_evaluation(cell,repaired)['feasible']
    assert evaluate_class_cell(cell,repaired)['feasible']
    assert max(scalar_cell_evaluation(cell,repaired)['raw_upper_bounds_with_inflation'])<=.1
    assert fraction>0 and any(r['kind']=='dual_path_convex_class' for r in trace)
    np.testing.assert_allclose(repaired.sum(),candidate.sum(),atol=1e-10)
