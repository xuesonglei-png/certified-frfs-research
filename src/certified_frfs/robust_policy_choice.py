"""Training-validation clean-accuracy band followed by exact box coverage."""
import numpy as np


def choose_robust_policy(rows,*,clean_tolerance=.005):
    if not rows or not np.isfinite(clean_tolerance) or not 0 <= clean_tolerance <= 1:
        raise ValueError('nonempty rows and tolerance in [0,1] required')
    for row in rows:
        if not row['folds']: raise ValueError('training validation folds required')
        row['mean_clean'] = float(np.mean([v['clean'] for v in row['folds']]))
        row['mean_robust'] = float(np.mean([v['robust'] for v in row['folds']]))
        row['mean_size'] = float(np.mean([len(v['features']) for v in row['folds']]))
        if not all(np.isfinite(row[k]) for k in ('mean_clean','mean_robust','mean_size')):
            raise ValueError('nonfinite selection score')
    best_clean = max(round(row['mean_clean'],12) for row in rows)
    eligible = [i for i,row in enumerate(rows) if round(row['mean_clean'],12) >= best_clean-clean_tolerance-1e-12]
    chosen = max(eligible,key=lambda i:(round(rows[i]['mean_robust'],12),round(rows[i]['mean_clean'],12),
        -rows[i]['mean_size'],-rows[i]['model_index'],-rows[i]['selector_index'],-rows[i]['fraction_index']))
    return {'chosen_index':chosen,'eligible_indices':eligible,'best_clean':best_clean,'clean_tolerance':float(clean_tolerance),
        'scope':'adaptive_inner_training_validation_rule_not_test_noninferiority_or_population_bound'}
