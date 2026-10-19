import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from graph_certified_restoration_v1 import GuardCache,restored,summarize,FAMILIES,FRACTIONS


def candidate(fs=(0,1),*,legacy=False,majority=True):
    cert={'majority_feasible':majority,'dependency_certificate':{}}
    return {'features':list(fs),'weights':[1.]*len(fs),'certificate':cert,
            'family_certificate':cert,'joint_feasible':legacy}


def fixture(bound=.11,*,available=True,upper=.09):
    A=np.arange(32,dtype=float).reshape(8,4)/32;y=np.repeat([0,1],4);calls=[]
    def signed(*args,**kwargs):
        calls.append('signed')
        return {'class_certificate':{'upper_bounds':[bound,bound]},
                'signed_object_report':{'signed_object_upper_bounds':[bound]*8,'objects':[]}}
    def graph(*args,**kwargs):
        calls.append(('graph',kwargs['seconds']))
        return {'upper_bound':upper,'solver_bound_available':available,'solver_status':1,
                'relaxation_primal_objective':0.,'solver_upper_bound':upper}
    return A,y,GuardCache(A,y,signed_fn=signed,graph_fn=graph),calls


def test_majority_fail_never_runs_signed_or_graph():
    A,y,cache,calls=fixture()
    report=cache.assess(candidate(majority=False),np.ones(4)/2,mode='graph')
    assert not report['accepted'] and calls==[]


def test_stronger_legacy_contract_implies_signed_without_graph():
    A,y,cache,calls=fixture()
    report=cache.assess(candidate(legacy=True),np.ones(4)/2,mode='graph')
    assert report['accepted'] and report['reason']=='legacy_positive_implies_signed' and calls==[]


def test_identical_reference_is_solved_once_and_all_classes_needed():
    A,y,cache,calls=fixture()
    report=cache.assess(candidate(),np.ones(4)/2,mode='graph')
    assert report['accepted'] and not report['legacy_positive_joint_feasible']
    assert len(calls)==3 and calls.count('signed')==1
    assert cache.counters['graph_class_calls']==2
    assert report['guards'][1]['reused_graph_reference']
    assert report['certificate_scope']=='signed_same_table_class_mean_decrease_not_mean_positive_object_losses'


def test_distinct_references_are_both_required():
    A,y,cache,calls=fixture();ref=np.array([.4,.45,.55,.6])
    report=cache.assess(candidate(),ref,mode='graph')
    assert report['accepted'] and calls.count('signed')==2 and cache.counters['graph_class_calls']==4


def test_screen_rejects_without_changing_scientific_threshold():
    A,y,cache,calls=fixture(.1300001)
    report=cache.assess(candidate(),np.ones(4)/2,mode='graph')
    assert not report['accepted'] and report['reason']=='prospective_graph_resource_screen'
    assert cache.counters['graph_class_calls']==0


def test_missing_dual_timeout_cannot_use_good_incumbent():
    A,y,cache,calls=fixture(available=False,upper=0.)
    report=cache.assess(candidate(),np.ones(4)/2,mode='graph')
    assert not report['accepted']
    assert report['guards'][0]['new_class_bounds']==[.11,.11]


def test_one_failed_class_prevents_admission():
    A,y,cache,calls=fixture();state=[.09,.101]
    cache.graph_fn=lambda *args,**kwargs:{'upper_bound':state.pop(0),'solver_bound_available':True,'solver_status':0}
    report=cache.assess(candidate(),np.ones(4)/2,mode='graph')
    assert not report['accepted'] and report['guards'][0]['new_class_bounds']==[.09,.101]


def test_restoration_first_pass_and_explicit_checked_fallback():
    A=np.zeros((8,4));y=np.repeat([0,1],4);ranking=[2,0,3,1];ref=np.ones(4)/2
    class MockGuard:
        def assess(self,c,reference,mode):return {'accepted':bool(c['joint_feasible'])}
    def provide(count):return candidate(sorted(ranking[:count]),legacy=count==3)
    out=restored(A,y,ref,ranking,2,'graph',provide,None,MockGuard())
    assert out['model']['features']==[0,2,3] and out['restored_count']==1 and len(out['trace'])==2
    def rejected(count):return candidate(sorted(ranking[:count]),legacy=False)
    full=candidate(range(4),legacy=True);full['weights']=ref.tolist()
    out=restored(A,y,ref,ranking,2,'graph',rejected,lambda:full,MockGuard())
    assert out['uniform_fallback'] and out['trace'][-1]['explicit_uniform_fallback']


def test_policy_comparison_is_same_eight_recipes_and_fixed_ties():
    families={}
    for family in FAMILIES:
        rows=[]
        for index,_ in enumerate(FRACTIONS):
            rows.append({'model':{'features':list(range(index+1))},
                         'score':{'clean_BA':.896 if index==0 else .9,'query_certified_BA':.8}})
        families[family]={'modes':{'graph':rows}}
    result=summarize([{'families':families}],'graph')
    assert len(result['options'])==8
    assert result['clean_first']['recipe_index']==1 and result['clean_first']['family']=='uniform_sqrt'
    assert result['compression_policy']['recipe_index']==0


def test_invalid_mode_is_explicit_error():
    A,y,cache,calls=fixture()
    with pytest.raises(ValueError):cache.assess(candidate(),np.ones(4)/2,mode='not_a_mode')


def test_resume_reuses_time_limited_class_dual_instead_of_rerunning():
    A,y,cache,calls=fixture(available=False,upper=0.)
    first=cache.assess(candidate(),np.ones(4)/2,mode='graph');saved=cache.snapshot()
    def forbidden(*args,**kwargs):raise AssertionError('resume must not rerun saved class solve')
    resumed=GuardCache(A,y,signed_fn=forbidden,graph_fn=forbidden,state=saved)
    second=resumed.assess(candidate(),np.ones(4)/2,mode='graph')
    assert not first['accepted'] and not second['accepted']
    assert resumed.counters['graph_class_calls']==2
