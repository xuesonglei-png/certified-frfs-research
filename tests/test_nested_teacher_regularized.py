import importlib.util
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_nested_teacher_regularized import select,CS,FRACTIONS

def test_full_and_reduced_choices_use_inner_scores_and_different_C_when_needed():
    row={'by_C':{str(C):{'policies':{str(f):{'clean':.7,'retained':1.} for f in FRACTIONS}} for C in CS}}
    row['by_C']['1.0']['policies']['1.0']['clean']=.8
    row['by_C']['10.0']['policies']['0.65']={'clean':.797,'retained':.65}
    proposed,full,trace=select([row])
    assert proposed['C']==10 and proposed['fraction']=='0.65'
    assert full['C']==1 and full['fraction']=='1.0'
    assert len(trace['options'])==12 and trace['tolerance']==.005
