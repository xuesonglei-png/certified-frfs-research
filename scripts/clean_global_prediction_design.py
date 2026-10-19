"""Fixed training endpoints and unchanged full native control selection."""
import numpy as np

NATIVE_ARMS=('fgs_native120','fgs_weighted_native120','sfrip_native6','sfrip_weighted_native6')
METHODS=('anchor','full_uniform','robust_training','nominal_training')+NATIVE_ARMS


def metrics_from_training(training,native):
    assert training['complete'] and native['complete'] and training['fold']==0
    assert training['reference_indices']==native['reference_indices']
    first=native['inner'][0]
    assert training['training_indices']==first['reference_indices']
    assert not set(first['reference_indices'])&set(first['validation_indices'])
    base=first['bases']['proposed'][0];block=training['block'];d=training['d']
    if block is not None:
        assert block['features']==base['features'] and block['anchor']==base['weights']
    features=base['features'];anchor=base['weights']
    metrics={'anchor':{'features':features,'weights':anchor},
        'full_uniform':{'features':list(range(d)),'weights':(np.ones(d)/np.sqrt(d)).tolist()}}
    for arm,key in (('robust_training','control'),('nominal_training','nominal')):
        weights=anchor if block is None else block[key]['target']
        assert len(weights)==len(features) and np.isfinite(weights).all() and min(weights)>=0
        active=[i for i,w in enumerate(weights) if w!=0.]
        metrics[arm]={'features':[features[i] for i in active],'weights':[weights[i] for i in active]}
    choices={};widths={'fgs_native120':120,'fgs_weighted_native120':1680,'sfrip_native6':6,'sfrip_weighted_native6':84}
    for method in NATIVE_ARMS:
        selection=native['selection'][method];index=selection['selected_index'];options=first['options'][method]
        assert len(options)==len(selection['options'])==widths[method]
        metrics[method]={k:options[index]['metric'][k] for k in ('features','weights')}
        choices[method]={'selected_index':index,'candidate_count':widths[method],'recipe':selection['options'][index]}
    return metrics,choices
