from copy import deepcopy
import pytest
from certified_path_prefixes import prefixes,progress_options


def fixture():
    guard={'accepted':True,'certificate':{},'class_certificate':{}}
    checks=[];fs=[0,1,2,3,4];w=[1.]*5
    for i in range(4):
        nf=fs[:-1];nw=w[:-1];nw[0]+=w[-1]
        checks.append({'id':i,'features_before':fs,'weights_before':w,'features':nf,'weights':nw,'eligible':True,**guard})
        fs,w=nf,nw
    return {'guard_reference':'original_full','initial_guard':guard,'initial_features':[0,1,2,3,4],
        'initial_weights':[1.]*5,'chosen_ids':[0,1,2,3],'checks':checks,'features':fs,'weights':w}


def test_all_intermediate_models_exposed_without_dropping_admission():
    fit=fixture();rows=progress_options(fit,require_original=True)
    assert [r['path_step'] for r in rows]==[0,1,2,3,4]
    assert [len(r['features']) for r in rows]==[5,4,3,2,1]
    assert all(sum(r['weights'])==5 and r['guard_reference']=='original_full' for r in rows)
    rows[1]['features'].clear();assert fit['checks'][0]['features']==[0,1,2,3]


def test_nonaccepted_skipped_or_reordered_points_cannot_be_prefixes():
    bad=fixture();bad['checks'][1]['eligible']=False
    with pytest.raises(ValueError):prefixes(bad)
    bad=fixture();bad['chosen_ids']=[0,2,3]
    with pytest.raises(ValueError):prefixes(bad)
    bad=fixture();bad['chosen_ids']=[0,1,1,3]
    with pytest.raises(ValueError):prefixes(bad)


def test_own_reference_cannot_be_promoted_and_empty_path_recipes_remain_fixed():
    own=fixture();own['guard_reference']='own_anchor'
    with pytest.raises(ValueError):progress_options(own,require_original=True)
    assert len(progress_options(own))==5
    fit=deepcopy(own);fit['chosen_ids']=[];fit['checks']=[]
    fit['features']=fit['initial_features'];fit['weights']=fit['initial_weights']
    assert [r['path_step'] for r in progress_options(fit)]==[0]*5
