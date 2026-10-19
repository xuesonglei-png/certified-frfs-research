import json
import numpy as np
import pytest
from run_frs_coordinate_diagnostic_v2 import builtin


def test_solver_metadata_numpy_scalars_convert_without_numeric_or_schema_changes():
    source = {'index':np.int64(7), 'value':np.float64(.123456789123),
              'accepted':np.bool_(False), 'weights':[0., np.float64(1.4142135623730951)]}
    actual = json.loads(json.dumps(source, default=builtin))
    assert actual == {'index':7, 'value':.123456789123, 'accepted':False,
                      'weights':[0., 1.4142135623730951]}
    with pytest.raises(TypeError):
        json.dumps({'unsupported':object()}, default=builtin)
