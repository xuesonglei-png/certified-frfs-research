"""Balanced matched controls and predeclared failure/compression diagnostics."""
import json
from collections import Counter
import numpy as np
from review_box_iterations import ROOT, paired


def main(source='nested-balanced-joint'):
    folder = ROOT/'results'/source
    summary = json.loads((folder/'complete.json').read_text())
    rows, failures = [], Counter()
    full_count = 0
    for row in summary['rows']:
        saved = json.loads((folder/(row['dataset']+'.json')).read_text())
        for split in saved['splits']:
            record = split['stage']['policies'][split['chosen']]
            full_count += len(record['candidate']['features']) == saved['d']
            for inner in split['inner']:
                for stage_record in inner['by_C'].values():
                    stage = stage_record['stage']
                    if not stage['joint_fit']['feasible']:
                        failures['inner_joint'] += 1
            failures['outer_selected'] += record['fallback']
        rows.append({k: row[k] for k in ('dataset', 'clean', 'full', 'noise', 'full_noise', 'removed')})
    report = {'rows': rows, 'mean_clean': summary['mean_clean'],
              'mean_full': summary['mean_full'], 'mean_removed': summary['mean_removed'],
              'vs_matched_balanced_full': paired([r['clean']-r['full'] for r in rows]),
              'noise_vs_matched_balanced_full': paired([r['noise']-r['full_noise'] for r in rows]),
              'full_support_outer_splits': int(full_count), 'failures': dict(failures),
              'scope': 'all15_all44_exposed_development_balanced_objective_not_independent_confirmation'}
    budget = folder.parent/(folder.name+'-common-budget')/'complete.json'
    if budget.exists():
        controls = json.loads(budget.read_text())
        report['common_balanced_classifier_budget_means'] = controls['means']
        report['common_balanced_classifier_budget_comparisons'] = controls['comparisons']
    (folder/'complete-review.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({k: v for k, v in report.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='nested-balanced-joint')
    main(parser.parse_args().source)
