"""All failed source retries, all fixed mixtures, no validation scoring."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT, FOLDER, digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from audit_flexible_majority_repair_admission import independent_protected
from audit_classwise_repair_admission import independent_class_certificate
from audit_simplex_source_budget_diagnostic import independent_loo


def main():
    threadpool_limits(1)
    path = ROOT/'research/simplex-source-budget-diagnostic.json'
    audit_path = ROOT/'research/simplex-source-budget-diagnostic-audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['passed'] and audit['diagnostic_sha256'] == digest(path)
    saved = json.loads(path.read_text())
    for rel, sha in saved['fingerprints'].items(): assert digest(ROOT/rel) == sha
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in
              [path, audit_path, Path(__file__), ROOT/'research/source-budget-interpolation-protocol.md']}
    rows = []
    for row in saved['rows']:
        name = row['dataset']; source = json.loads((FOLDER/(name+'.json')).read_text())
        block = source['inner'][row['block']]; learner = block['learners'][row['learner']]
        X, y = load_dataset(name); ri = np.array(source['reference_indices']); ca = np.array(block['reference_indices'])
        raw, labels = X[ri][ca], y[ri][ca]; A, _ = preprocess_fit(raw, raw[:1])
        fs = learner['features']; anchor = np.array(learner['anchor']); target = np.array(row['retry_fit']['weights'])
        full = np.ones(A.shape[1])/np.sqrt(A.shape[1]); protected = independent_protected(A, labels, full)
        points = []
        for t in (0., .05, .1, .25, .5, .75, 1.):
            w = (1-t)*anchor+t*target
            cert = weight_certificate(A, labels, fs, full, w, protected=protected)
            classes = independent_class_certificate(labels, cert)
            ba = independent_loo(A, labels, fs, w)
            points.append({'t': t, 'weights': w.tolist(), 'certificate': cert, 'class_certificate': classes,
                'accepted': bool(cert['feasible'] and classes['feasible_all_classes']),
                'hard_training_LOO_BA': ba, 'hard_gain_pp': 100*(ba-row['hard_training_LOO_BA_anchor'])})
        rows.append({'dataset': name, 'block': row['block'], 'learner': row['learner'], 'points': points})
        print(name, row['block'], row['learner'], 'accepted', sum(p['accepted'] for p in points),
              'accepted_improving', sum(p['accepted'] and p['hard_gain_pp'] > 1e-9 for p in points), flush=True)
    points = [p for r in rows for p in r['points']]
    result = {'rows': rows, 'fingerprints': hashes, 'cases': 7, 'points': len(points),
              'accepted_points': sum(p['accepted'] for p in points),
              'accepted_hard_improving_points': sum(p['accepted'] and p['hard_gain_pp'] > 1e-9 for p in points),
              'accepted_hard_harmful_points': sum(p['accepted'] and p['hard_gain_pp'] < -1e-9 for p in points),
              'scope': 'seven_source_failures_training_only_fixed_mixtures_not_predictive_evidence',
              'query_labels_used': False, 'goal_achieved': False, 'complete': True}
    (ROOT/'research/source-budget-interpolation-diagnostic.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k:v for k,v in result.items() if k not in ('rows','fingerprints')}, flush=True)


if __name__ == '__main__': main()
