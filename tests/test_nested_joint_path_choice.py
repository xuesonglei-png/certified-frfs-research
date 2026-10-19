import importlib.util
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from run_nested_joint_path import choose, choices


def row(family,mi,fi,clean,features):
    return {'family':family,'family_index':int(family == 'joint'),'model_index':mi,'fraction_index':fi,
        'folds':[{'clean':value,'features':fs} for value,fs in zip(clean,features)]}


def test_validation_accuracy_precedes_compression_and_family_preference():
    rows = [row('original',0,0,[.75,.75],[[0],[0]]),row('joint',7,3,[.76,.76],[[0,1,2],[0,1,2]])]
    assert choose(rows) == 1


def test_tie_uses_actual_counts_then_config_fraction_and_original():
    rows = [row('original',0,0,[.8,.8],[[0,1,2],[0,1,2]]),
        row('joint',1,0,[.8,.8],[[0,1],[0,1]]),
        row('joint',0,2,[.8,.8],[[0,1],[0,1]]),
        row('original',0,2,[.8,.8],[[0,1],[0,1]])]
    assert choose(rows) == 3


def test_all_original_choices_retained_and_no_query_noise_in_selection():
    rows = choices(); assert len(rows) == 96
    assert len({(r['family'],r['model_index'],r['fraction']) for r in rows}) == 96
    for r in rows:
        r['folds'] = [{'clean':.5,'features':[0],'noise':int(r['family'] == 'joint')}]
    assert choose(rows) == 0 and rows[0]['family'] == 'original'
