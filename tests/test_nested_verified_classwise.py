import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_nested_verified_classwise import hierarchical_select

def test_fraction_does_not_change_teacher_regularization_selection():
    row={'by_C':{str(C):{'policies':{f:{'clean':.7,'retained':1.} for f in ('0.5','0.65','0.8','1.0')}} for C in (.1,1.,10.)}}
    row['by_C']['1.0']['policies']['1.0']['clean']=.8
    row['by_C']['1.0']['policies']['0.65']={'clean':.797,'retained':.65}
    row['by_C']['10.0']['policies']['0.5']={'clean':.95,'retained':.5}
    selected,full,trace=hierarchical_select([row])
    assert selected['C']==full['C']==1 and selected['fraction']=='0.65'
    assert trace['hierarchical'] and all(r['C']==1 for r in trace['eligible'])
