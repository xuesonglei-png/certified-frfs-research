"""Report every predeclared budget and descriptive recent native controls."""
import json
from review_box_iterations import ROOT, paired


def main():
    folder = ROOT/'results/joint-group-fuzzy-pilot'
    source = json.loads((folder/'complete.json').read_text())
    controls = json.loads((ROOT/'results/common-fuzzy-classifier/complete-fuzzy-review.json').read_text())
    byname = {r['dataset']: r for r in controls['rows']}
    report = {'budgets': {key: {'pilot': values, 'vs_native': {
        m: paired([r['methods'][key]['clean']-byname[r['dataset']]['methods'][m]['clean']
                   for r in source['rows']])
        for m in ('fgs2023_native', 'sfrip2025_native')}}
        for key, values in source['budgets'].items()},
        'scope': 'all_predeclared_budget_curves_exposed_development_descriptive',
        'limitations': 'Classifier training and support budgets differ from native controls. '
                      'Three budgets are evaluated; no test-selected winner or multiplicity-adjusted '
                      'confirmatory claim. No independently nested joint C/penalty/support selection.'}
    (folder/'complete-recent-review.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
