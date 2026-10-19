"""Compare only complete audited cohorts; retain formula/author limitations."""
import json
from pathlib import Path
import numpy as np
from run_dsnfrs_formula_native_pilot import ROOT, digest
from continue_soft_topk_knn_pilot import verify
from review_box_iterations import paired


def verified_rows(folder):
    audit = verify(folder)
    summary = json.loads((folder/'complete.json').read_text())
    rows = {r['dataset']: r for r in summary['rows']}
    assert len(rows) == 15 and set(rows) == set(audit['artifact_hashes'])
    return rows


def main():
    baseline = ROOT/'results/dsnfrs-formula-native-pilot'
    native = verified_rows(baseline)
    baselines = {'native12': native}
    expanded = ROOT/'results/dsnfrs-formula-expanded-pilot'
    if (expanded/'complete-audit.json').exists(): baselines['expanded24'] = verified_rows(expanded)
    comparisons, snapshots = {}, {}
    candidates = ['simplex-soft5-knn-pilot', 'joint-guarded-soft5-knn-pilot', 'hardaware-soft5-knn-pilot']
    for cohort in candidates:
        folder = ROOT/'results'/cohort
        if not (folder/'complete-audit.json').exists(): continue
        rows = verified_rows(folder); assert set(rows) == set(native)
        snapshots[cohort] = digest(folder/'complete-audit.json')
        for baseline_name, controls in baselines.items():
            bfolder = baseline if baseline_name == 'native12' else expanded
            details = []
            for name in sorted(rows):
                ours = json.loads((folder/(name+'.json')).read_text())
                theirs = json.loads((bfolder/(name+'.json')).read_text())
                assert ours['reference_indices'] == theirs['reference_indices']
                assert ours['validation_indices'] == theirs['validation_indices']
                assert not ours['query_labels_used_for_selection'] and not theirs['query_labels_used_for_selection']
                detail = {'dataset': name, 'd': theirs['d'], 'DSN_count': controls[name]['actual_count'],
                          'DSN_recipe': controls[name]['recipe'], 'DSN_metrics': controls[name]['metrics'],
                          'arm_metrics': rows[name]['metrics'],
                          'proposed_count': len(ours['outer']['results']['proposed']['metric']['features'])}
                details.append(detail)
            contrasts = {}
            for arm in rows[sorted(rows)[0]]['metrics']:
                contrasts[arm] = {}
                for metric in ('clean_BA', 'mean_perturbed_BA', 'query_certified_BA'):
                    differences = np.array([r['arm_metrics'][arm][metric]-r['DSN_metrics'][metric] for r in details])
                    contrasts[arm][metric] = paired(differences)
            comparisons[cohort+'/'+baseline_name] = {'rows': details, 'contrasts': contrasts,
                'mean_DSN_removed': float(np.mean([1-r['DSN_count']/r['d'] for r in details])),
                'mean_proposed_removed': float(np.mean([1-r['proposed_count']/r['d'] for r in details]))}
    for name in baselines:
        folder = baseline if name == 'native12' else expanded
        snapshots[name] = digest(folder/'complete-audit.json')
    result = {'comparisons': comparisons, 'audit_hashes': snapshots,
              'goal_achieved': False, 'script_sha256': digest(Path(__file__)),
              'limitations': ['All15 and reused folds are exposed development; statistics are descriptive.',
                  'DSNFRS uses declared formula conventions; author example/table reproduction remains unresolved.',
                  'Uniform cityblock5NN is shared but original DSNFRS classifier distance conventions are unverified.',
                  'Native DSN counts can differ; this is not a cardinality-matched selection comparison.',
                  'Winning against this implementation alone does not establish superiority over recent author methods.',
                  'No selective exclusion, query-based recipe choice or main-cohort overwrite.']}
    (ROOT/'research/dsnfrs-formula-comparison-review.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: {a: r['clean_BA'] for a, r in v['contrasts'].items()} for k, v in comparisons.items()}, indent=2))


if __name__ == '__main__': main()
