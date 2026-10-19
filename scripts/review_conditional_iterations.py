"""Complete-only iteration history, matched comparisons and repair diagnostics."""
import hashlib
import json
import numpy as np
from review_box_iterations import ROOT, paired

SOURCES = ['nested-balanced-joint', 'nested-joint-backward', 'nested-conditional-backward',
           'nested-conditional-feasible', 'nested-conditional-label', 'nested-crossfit-classification-rounded']


def main():
    reports, fingerprints = [], {}
    baseline = json.loads((ROOT/'results/nested-joint-backward/complete.json').read_text())
    static = {r['dataset']: r for r in baseline['rows']}
    for source in SOURCES:
        folder = ROOT/'results'/source
        path = folder/'complete.json'
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        rows = summary['rows']
        assert set(r['dataset'] for r in rows) == set(static) and len(rows) == 15
        counts = {'outer_splits': 0, 'inner_stages': 0, 'outer_full_support': 0,
                  'outer_selected_fallback': 0, 'outer_selected_restored': 0,
                  'outer_selected_restored_attributes': 0, 'failed_group_bypassed_stages': 0}
        for row in rows:
            saved = json.loads((folder/(row['dataset']+'.json')).read_text())
            for split in saved['splits']:
                record = split['stage']['policies'][split['chosen']]
                counts['outer_splits'] += 1
                counts['outer_full_support'] += len(record['candidate']['features']) == saved['d']
                counts['outer_selected_fallback'] += bool(record['fallback'])
                attempts = record.get('feasibility_repair', [])
                if attempts:
                    growth = len(record['candidate']['features'])-len(attempts[0]['features'])
                    counts['outer_selected_restored'] += growth > 0
                    counts['outer_selected_restored_attributes'] += growth
                stages = [split['stage']]+[r['stage'] for inner in split['inner'] for r in inner['by_C'].values()]
                counts['inner_stages'] += len(stages)-1
                counts['failed_group_bypassed_stages'] += sum(
                    not st['joint_fit']['feasible'] and 'conditional_path' in st for st in stages)
        assert counts['outer_splits'] == 44 and counts['inner_stages'] == 384
        report = {'source': source, 'BA': summary['mean_clean'], 'removed': summary['mean_removed'],
                  'matched_full': paired([r['clean']-r['full'] for r in rows]),
                  'vs_static_backward_BA': paired([r['clean']-static[r['dataset']]['clean'] for r in rows]),
                  'vs_static_backward_removal': paired([r['removed']-static[r['dataset']]['removed'] for r in rows]),
                  'counts': counts}
        budget = folder.parent/(source+'-common-budget')/'complete.json'
        if budget.exists():
            controls = json.loads(budget.read_text())
            report['common_budget'] = {'means': controls['means'], 'comparisons': controls['comparisons']}
            fingerprints[str(budget.relative_to(ROOT))] = hashlib.sha256(budget.read_bytes()).hexdigest()
        reports.append(report)
        fingerprints[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {'iterations': reports, 'fingerprints': fingerprints,
               'scope': 'all15_exposed_development_history_uncorrected_descriptive_comparisons',
               'limitations': 'No test-selected winner; historical mixed-classifier rows excluded. FGS/SFRIP same-budget '
                             'adaptations are not native SOTA; FD+/FCE+ fidelity remains unresolved. Training surrogate '
                             'improvement, absence of a significant loss and certification do not prove accuracy superiority.'}
    destination = ROOT/'results/conditional-iteration-review.json'
    destination.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    lines = ['# Conditional support iteration review', '', payload['scope'], '',
             '| Version | Balanced accuracy | Attribute removal | Gap vs matched full (pp) |',
             '| --- | ---: | ---: | ---: |']
    for r in reports:
        lines.append(f"| {r['source']} | {100*r['BA']:.4f}% | {100*r['removed']:.4f}% | {r['matched_full']['mean_pp']:.4f} |")
    lines += ['', payload['limitations'], '',
              'Detailed paired intervals, common-budget controls, restorations and fallback counts:',
              '`results/conditional-iteration-review.json`. No LaTeX modifications or compilation.']
    (ROOT/'research/conditional-iteration-review.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print([{'source': r['source'], 'BA': r['BA'], 'removed': r['removed'], 'counts': r['counts']} for r in reports], flush=True)


if __name__ == '__main__':
    main()
