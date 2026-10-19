import numpy as np
from certified_frfs.functional_pipeline_cv import CONFIGS,choose


def test_full_grid_choice_uses_training_scores_and_prefers_simple_strongly_regularized_ties():
    rows = [{'config_index':i,'fold_clean':[.75,.75],'fold_model_ids':[i,i]} for i in range(len(CONFIGS))]
    assert choose(rows)['config_index'] == 0
    rows[19]['fold_clean'] = [.80,.82]
    chosen = choose(rows)
    assert chosen['config_index'] == 19 and np.isclose(chosen['mean_clean'],.81)
    assert len(chosen['all_mean_clean']) == 30
