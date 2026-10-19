from copy import deepcopy
import pytest
from diagnose_postmetric_pruning_transfer import METHODS, inner_rows, aggregate


def fixture():
    fit = {'initial_features': [0,1], 'features': [0], 'initial_hard_LOO_BA': .8,
           'final_hard_LOO_BA': .9, 'chosen_ids': [0], 'guard_reference': 'original_full'}
    options = [{'variant': v, 'metric': {'features': fs, 'pruning_id': 0},
                'evaluation': {'metrics': {'clean_BA': ba}}}
               for v, fs, ba in [(0,[0,1],.8), (1,[0],.7)]]
    return {'options': {m: deepcopy(options) for m in METHODS},
            'pruning_records': [{'fit': fit}]}


def test_training_gain_can_fail_on_validation_and_datasets_have_equal_weight():
    first = inner_rows('first', 2, [fixture()]*3)
    second = inner_rows('second', 2, [fixture()])
    for row in second:
        row['inner_validation_delta_pp'] = 30
        row['training_gain_validation_loss'] = False
    result = aggregate(first+second)['proposed']
    assert result['training_gain_validation_loss'] == 3
    assert result['dataset_equal_means']['inner_validation_delta_pp'] == pytest.approx(10)


def test_analysis_does_not_accept_outer_blocks_or_consult_outer_predictions():
    dataset = {'inner': [fixture()], 'outer': object()}
    rows = inner_rows('qa', 2, dataset['inner'])
    assert len(rows) == 6
    assert all(r['training_gain_validation_loss'] for r in rows)
