"""Inner-only comparison of direct majority surrogate and held-out5NN utility."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_metric import MajorityTerms


def main():
    threadpool_limits(1); rows = []
    for name in ['glass', 'haberman', 'heart_statlog', 'sonar', 'vehicle']:
        for fold in range(3):
            p = json.loads((ROOT/'results/predictive-iteration41'/f'{name}-s101-f{fold}.json').read_text())
            X, y = load_dataset(name); tr = np.array(p['train_indices'])
            for index, s in enumerate(p['inner_splits']):
                ids = tr[np.array(s['train'])]; A, _ = preprocess_fit(X[ids], X[ids]); labels = y[ids]
                ws = MajorityTerms(A, labels, eps=.01, weights=np.ones(A.shape[1])/A.shape[1])
                applicable = ws.applicable & (ws.cross.sum(axis=1) >= 3); records = []
                for r in s['fits']:
                    w = np.asarray(r['weights']); T = np.einsum('d,dijf->ijf', w, ws.terms, optimize=True)
                    lower = np.partition(np.where(ws.cross, T.min(axis=2), np.inf), 2, axis=1)[:, 2]
                    coverage = float(np.mean([np.mean((lower > 1e-10)[(labels == q)&applicable]) for q in np.unique(labels) if np.any((labels == q)&applicable)]))
                    D = abs(A[:, None, :]-A[None, :, :])@w; np.fill_diagonal(D, np.inf)
                    nn = np.argsort(D, axis=1, kind='stable')[:, :min(5, len(labels)-1)]
                    classes = np.unique(labels); pred = classes[np.stack([(labels[nn] == q).sum(axis=1) for q in classes], axis=1).argmax(axis=1)]
                    loo = float(np.mean([(pred[labels == q] == q).mean() for q in classes]))
                    validation = (r['validation']['clean']+r['validation']['noise01'])/2
                    records.append({'config': r['config'], 'size': len(r['features']), 'reference_loo_ba': loo,
                        'reference_majority_balanced_coverage_applicable_only': coverage, 'validation_utility': validation,
                        'loo_validation_gap': loo-validation, 'applicable_count': int(applicable.sum()), 'n': len(labels)})
                rows.append({'dataset': name, 'outer_fold': fold, 'inner': index, 'candidates': records})
        print(name, 'inner-only witness diagnostic complete', flush=True)
    out = ROOT/'results/predictive-iteration41/witness-bias-diagnostic.json'
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps({'rows': rows, 'scope': 'training_inner_dependent_candidates_not_independent_tests'}, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
