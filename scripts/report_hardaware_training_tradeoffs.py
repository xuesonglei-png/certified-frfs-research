"""Audited training-only gains, guard/task conflicts and dataset aggregation."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from run_hardaware_guarded_soft5_admission import ROOT, FOLDER, digest
from continue_soft_topk_knn_pilot import verify


def main():
    audit = verify(FOLDER)
    review_path = ROOT/'research/hardaware-guarded-soft5-admission-review.json'
    reviewed = json.loads(review_path.read_text())
    assert reviewed['audit_sha256'] == digest(FOLDER/'complete-audit.json')
    previous = {r['dataset']: r for r in reviewed['rows']}
    counts, rows = Counter(), []
    for name in sorted(audit['artifact_hashes']):
        saved = json.loads((FOLDER/(name+'.json')).read_text()); dataset_counts = Counter(); gains, parent_gains = [], []
        for r, comparison in zip(saved['records'], previous[name]['targets'], strict=True):
            assert r['features'] == comparison['features']; fit = r['fit']
            ba, loss = fit['initial_hard_LOO_BA'], fit['training_objective_initial']
            for q in fit['seed_checks']:
                dataset_counts['seed_checks'] += 1
                dataset_counts['FRS_accepted_but_hard_BA_worsening_seeds'] += bool(q['accepted'] and q['hard_LOO_BA'] < ba-1e-12)
            ba, loss = fit['search_initial_hard_LOO_BA'], fit['search_initial_objective']
            for iteration in sorted(set(q['iteration'] for q in fit['checks'])):
                for q in fit['checks']:
                    if q['iteration'] != iteration: continue
                    dataset_counts['local_checks'] += 1
                    dataset_counts['FRS_accepted_but_hard_BA_worsening_local_checks'] += bool(q['accepted'] and q['hard_LOO_BA'] < ba-1e-12)
                    dataset_counts['FRS_rejected_but_hard_BA_improving_local_checks'] += bool(not q['accepted'] and q['hard_LOO_BA'] > ba+1e-12)
                chosen = [fit['checks'][rid] for rid in fit['accepted_step_ids'] if fit['checks'][rid]['iteration'] == iteration]
                if chosen:
                    q = chosen[0]
                    dataset_counts['chosen_steps_increasing_soft_loss'] += q['training_objective'] > loss+1e-8
                    ba, loss = q['hard_LOO_BA'], q['training_objective']
            gain = 100*(comparison['target_hard_LOO_BA']-comparison['anchor_hard_LOO_BA'])
            parent_gain = 100*(comparison['target_hard_LOO_BA']-comparison['parent_hard_LOO_BA'])
            gains.append(gain); parent_gains.append(parent_gain)
            dataset_counts['supports'] += 1
            dataset_counts['hard_LOO_improved_vs_anchor'] += gain > 1e-8
            dataset_counts['hard_LOO_worsened_vs_anchor'] += gain < -1e-8
            dataset_counts['lost_mixture_admissions_vs_seeded_parent'] += r['previous_admitted_mixtures']-sum(m['accepted'] for m in r['mixtures'])
        counts.update(dataset_counts)
        rows.append({'dataset': name, 'counts': dict(dataset_counts),
                     'mean_support_training_LOO_gain_vs_anchor_pp': float(np.mean(gains)),
                     'mean_support_training_LOO_gain_vs_seeded_parent_pp': float(np.mean(parent_gains))})
    report = {'rows': rows, 'counts': dict(counts),
              'dataset_balanced_mean_training_LOO_gain_vs_anchor_pp': float(np.mean([r['mean_support_training_LOO_gain_vs_anchor_pp'] for r in rows])),
              'dataset_balanced_mean_training_LOO_gain_vs_seeded_parent_pp': float(np.mean([r['mean_support_training_LOO_gain_vs_seeded_parent_pp'] for r in rows])),
              'scope': 'training_only_descriptive_no_independent_supports_or_query_superiority',
              'goal_achieved': False, 'audit_sha256': digest(FOLDER/'complete-audit.json'),
              'endpoint_review_sha256': digest(review_path), 'reviewer_sha256': digest(Path(__file__))}
    (ROOT/'research/hardaware-training-tradeoffs.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__': main()
