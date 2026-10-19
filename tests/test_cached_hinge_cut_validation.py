import json
from pathlib import Path
import numpy as np
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from independent_hinge_cut_validation import validate_cut
from cached_independent_hinge_cut_validation import CachedValidator


def test_geometry_cache_preserves_independent_nonunique_face_checks():
    root=Path(__file__).resolve().parents[1];s=json.loads((root/'results/majority-hinge-inner-pilot/iris.json').read_text())
    X,_=load_dataset('iris');raw=X[s['reference_indices']];b=s['inner'][1]
    A,_=preprocess_fit(raw[b['reference_indices']],raw[b['validation_indices']]);r=b['stage']['learners'][3]
    info=r['fit']['training_witnesses'];table=A[:,r['features']];validator=CachedValidator(table,info)
    for ci in (0,1,84):
        cut=r['fit']['cuts'][ci];point=np.array(r['fit']['cut_points'][cut['point_id']])
        assert validator.validate(point,cut)==validate_cut(table,point,info,cut)
    assert not validator.A.flags.writeable
