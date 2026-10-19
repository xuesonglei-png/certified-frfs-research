"""Training-only reason decomposition of the complete audited pool."""
import json
from collections import Counter
import numpy as np
from run_recent_pool_certificate_admission import ROOT, FOLDER, digest


def main():
    audit = json.loads((FOLDER/'complete-audit.json').read_text())
    assert audit['passed']
    rows, counts = [], Counter()
    for name, sha in sorted(audit['artifact_hashes'].items()):
        artifact = FOLDER/(name+'.json'); assert digest(artifact) == sha
        saved = json.loads(artifact.read_text())
        records = []
        for r in saved['pool']:
            cert = r['ordinary']['certificate']; classes = r['ordinary_class_certificate']
            causes = []
            if not cert['majority_feasible']: causes.append('majority')
            if cert['dependency_certificate']['upper_bound'] > .1: causes.append('global_dependency')
            if not classes['feasible_all_classes']: causes.append('class_dependency')
            for cause in causes: counts['ordinary_failed_'+cause] += 1
            info = r['repair']
            if info is not None: counts['repair_'+info['status']] += 1
            counts['nonfull_supports'] += len(r['features']) < saved['d']
            counts['accepted_nonfull'] += r['accepted'] and len(r['features']) < saved['d']
            records.append({'features': r['features'], 'origins': r['origins'], 'accepted': r['accepted'],
                            'ordinary_failure_reasons': causes,
                            'ordinary_majority_shortfalls': np.maximum(0, np.array(cert['required'])-cert['retained_counts']).tolist(),
                            'global_dependency_upper': cert['dependency_certificate']['upper_bound'],
                            'max_class_dependency_upper': classes['max_upper_bound'],
                            'repair_status': None if info is None else info['status'],
                            'repair_solver_status': None if info is None else info['solver_status']})
        rows.append({'dataset': name, 'd': saved['d'], 'records': records,
                     'all_recent_supports_rejected': all(not r['accepted'] for r in saved['pool'] if any(o['method'] in ('fgs2023', 'sfrip2025') for o in r['origins'])),
                     'only_full_admitted': all(len(r['features']) == saved['d'] for r in saved['pool'] if r['accepted'])})
    report = {'rows': rows, 'counts': dict(counts), 'query_labels_or_accuracy_used': False,
              'only_full_datasets': [r['dataset'] for r in rows if r['only_full_admitted']],
              'recent_pool_rejected_datasets': [r['dataset'] for r in rows if r['all_recent_supports_rejected']],
              'goal_achieved': False, 'admission_audit_sha256': digest(FOLDER/'complete-audit.json'),
              'scope': 'training_contract_bottlenecks_not_prediction_selection_or_proof_of_infeasibility'}
    (ROOT/'results/recent-pool-admission-bottlenecks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({k: v for k, v in report.items() if k != 'rows'}, flush=True)


if __name__ == '__main__': main()
