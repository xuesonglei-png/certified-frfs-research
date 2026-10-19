import importlib.util
from pathlib import Path
import sys
import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPT))
spec=importlib.util.spec_from_file_location('expansion',SCRIPT/'analyze_crossfit_majority_expansion.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_signflip_uses_dataset_units_and_two_sided_tail():
    assert module.signflip_p(np.ones(5))==2/32
    assert module.signflip_p([0,0])==1
    assert module.signflip_p([1,-1])==1
    assert module.signflip_p([1,1,0])==.5


def test_dataset_macro_is_not_fold_weighted_when_sizes_differ():
    grouped={'protected':{'a':[{'clean':1.}], 'b':[{'clean':0.}]*3},
             'all':{'a':[{'clean':0.}], 'b':[{'clean':0.}]*3}}
    result=module.analyze(grouped,[])
    assert result['methods']['protected']['means']['clean']==.5
    assert result['pairs'][0]['mean_delta']==.5
