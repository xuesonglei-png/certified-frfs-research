"""Independent training metric export and complete native recipe provenance."""
import numpy as np
from audit_forest_compact_transfer import equivalent

NATIVE_ARMS=('fgs_native120','fgs_weighted_native120','sfrip_native6','sfrip_weighted_native6')


def reconstruct_metrics(training,native):
    first=native['inner'][0];block=training['block'];d=training['d']
    assert training['fold']==0 and training['reference_indices']==native['reference_indices']
    assert training['training_indices']==first['reference_indices']
    assert not set(first['reference_indices'])&set(first['validation_indices'])
    base=first['bases']['proposed'][0];fs=base['features'];anchor=base['weights']
    if block is not None:
        assert block['features']==fs;equivalent(block['anchor'],anchor)
    metrics={'anchor':{'features':fs,'weights':anchor},
        'full_uniform':{'features':list(range(d)),'weights':[1/np.sqrt(d)]*d}}
    points={'robust_training':anchor if block is None else block['control']['target'],
            'nominal_training':anchor if block is None else block['nominal']['target']}
    for arm,point in points.items():
        assert len(point)==len(fs) and all(np.isfinite(point)) and all(w>=0 for w in point)
        pairs=[(a,w) for a,w in zip(fs,point,strict=True) if w>0.]
        metrics[arm]={'features':[a for a,w in pairs],'weights':[w for a,w in pairs]}
    choices={}
    for method,width in zip(NATIVE_ARMS,(120,1680,6,84),strict=True):
        choice=native['selection'][method];index=choice['selected_index']
        assert len(choice['options'])==len(first['options'][method])==width
        metric=first['options'][method][index]['metric'];metrics[method]={'features':metric['features'],'weights':metric['weights']}
        choices[method]={'selected_index':index,'candidate_count':width,'recipe':choice['options'][index]}
    return metrics,choices
