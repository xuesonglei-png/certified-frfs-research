"""Selection-transfer failure review; never chooses a new winning setting."""
import json
from collections import Counter
import numpy as np
from review_box_iterations import ROOT
from certified_frfs.datasets import load_dataset


def main():
    folder = ROOT/'results/forest-penalty-selection'
    summary = json.loads((folder/'complete.json').read_text())
    assert (folder/'complete-audit.json').exists()
    records = []; rows = []
    for row in summary['rows']:
        name = row['dataset']; saved = json.loads((folder/(name+'.json')).read_text())
        _, y = load_dataset(name); group = []
        for i, r in enumerate(saved['records']):
            indices = np.array(r['outer_reference_indices']); query = y[indices[r['validation_indices']]]
            reference = y[indices[r['reference_indices']]]; key = r['selection']['selected']
            _, counts = np.unique(query, return_counts=True)
            item = {'dataset': name, 'record_index': i, 'selected': key,
                'reference_count': len(reference), 'query_count': len(query), 'minimum_present_query_class_count': int(counts.min()),
                'query_classes': int(len(counts)), 'reference_classes': int(len(np.unique(reference))),
                'internal_selection_gain_pp': 100*(r['selection']['mean_scores'][key]-r['selection']['mean_scores']['additive']),
                'heldout_clean_gain_pp': 100*(r['metrics']['selected']['clean']-r['metrics']['additive']['clean']),
                'heldout_noise_gain_pp': 100*(r['metrics']['selected']['noise']-r['metrics']['additive']['noise']),
                'forest_failure': r['fallback']}
            records.append(item); group.append(item)
        rows.append({'dataset': name, 'selection_counts': dict(Counter(r['selected'] for r in group)),
            **{k: float(np.mean([r[k] for r in group])) for k in
               ('internal_selection_gain_pp', 'heldout_clean_gain_pp', 'heldout_noise_gain_pp')}})
    result = {'rows': rows, 'records': records,
        'dataset_equal_internal_selection_gain_pp': float(np.mean([r['internal_selection_gain_pp'] for r in rows])),
        'dataset_equal_heldout_gain_pp': summary['clean_gain']['mean_pp'],
        'worst_reference_cases': sorted(records, key=lambda r: (r['heldout_clean_gain_pp'], r['dataset'], r['record_index']))[:10],
        'scope': 'descriptive_selection_transfer_and_failure_review_no_posthoc_hyperparameter_choice'}
    (folder/'failure-review.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# Interaction penalty selection: development review', '',
        'All15/all128 references; full raw support. Independent audit passed. No feature-selection or independent-confirmation claim.', '',
        f"Clean gain: {summary['clean_gain']['mean_pp']:.4f}pp; descriptive 95% interval {summary['clean_gain']['bootstrap95_pp']}; descriptive sign-flip p={summary['clean_gain']['sign_flip_two_sided_p']:.6f}.",
        f"Noise gain: {summary['noise_gain']['mean_pp']:.4f}pp. Frozen resource gate passed: {summary['gate_passed']}.",
        f"Choices: {summary['selection_counts']}; final forest failures: {summary['counts']['final_forest_failures']}.",
        f"Mean selected-minus-additive internal-CV gain: {result['dataset_equal_internal_selection_gain_pp']:.4f}pp. This is selection evidence, not an independent performance estimate.", '',
        '| Dataset | Internal CV gain (pp) | Held-out gain (pp) | Noise gain (pp) |',
        '|---|---:|---:|---:|']
    lines += [f"| {r['dataset']} | {r['internal_selection_gain_pp']:.4f} | {r['heldout_clean_gain_pp']:.4f} | {r['heldout_noise_gain_pp']:.4f} |" for r in rows]
    lines += ['', 'Individual worst cases and rare-class counts are retained in results/forest-penalty-selection/failure-review.json. No cases were excluded.', '',
        'This compares against the identical constrained C1 additive model. It does not show superiority to FGS, SFRIP, FD+, FCE+, or optimally tuned full-feature classifiers. Those require separate matched comparisons. Repeatedly exposed datasets cannot support confirmatory significance.']
    (ROOT/'research/forest-penalty-selection-review.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print({k: v for k, v in result.items() if k not in ('rows', 'records', 'worst_reference_cases')}, flush=True)


if __name__ == '__main__': main()
