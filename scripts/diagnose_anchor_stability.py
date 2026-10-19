"""Held-out inner teacher margins diagnose reference-anchor stability."""
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_classifier import triangular_box_margins
from audit_triangular_classifier_diagnostic import manual_margins

SOURCE = ROOT/'results/nested-fuzzy-logistic'
FOLDER = ROOT/'results/anchor-stability-diagnostic'


def main():
    threadpool_limits(1)
    FOLDER.mkdir(exist_ok=True)
    summary = json.loads((SOURCE/'complete.json').read_text())
    rows = []
    for row in summary['rows']:
        name = row['dataset']
        source = SOURCE/(name+'.json')
        out = FOLDER/(name+'.json')
        if out.exists():
            raise FileExistsError(out)
        saved = json.loads(source.read_text())
        X, y = load_dataset(name)
        records = []
        for split in saved['splits']:
            ri = np.array(split['reference_indices'])
            raw, labels = X[ri], y[ri]
            A, _ = preprocess_fit(raw, raw)
            stage = split['stage']
            classes = np.array(stage['teacher_classes'])
            original = triangular_box_margins(A, labels, stage['teacher_coef'],
                                               stage['teacher_intercept'], classes, eps=.01)
            oof = np.full(len(ri), np.nan)
            clean = np.full(len(ri), np.nan)
            visits = np.zeros(len(ri), int)
            C = str(split['selected_config']['C'])
            for inner in split['inner']:
                a, b = np.array(inner['reference_indices']), np.array(inner['validation_indices'])
                assert set(a).isdisjoint(b)
                _, B = preprocess_fit(raw[a], raw[b])
                teacher = inner['by_C'][C]['stage']
                W, bias = np.array(teacher['teacher_coef']), np.array(teacher['teacher_intercept'])
                inner_classes = np.array(teacher['teacher_classes'])
                exact = triangular_box_margins(B, labels[b], W, bias, inner_classes, eps=.01)
                scalar, _ = manual_margins(B, labels[b], W, bias, inner_classes)
                np.testing.assert_allclose(exact, scalar, atol=1e-10)
                oof[b] = exact
                clean[b] = triangular_box_margins(B, labels[b], W, bias, inner_classes, eps=0)
                visits[b] += 1
            known = visits == 1
            assert np.all(visits <= 1)
            unique, counts = np.unique(labels, return_counts=True)
            assert np.array_equal(~known, np.isin(labels, unique[counts == 1]))
            protected = original > 1e-10
            verified = protected & known
            stable = verified & (oof > 1e-10)
            wrong = verified & (clean <= 1e-10)
            records.append({'reference_indices': ri.tolist(), 'teacher_C': float(C),
                'original_margins': original.tolist(),
                'oof_margins': [float(v) if np.isfinite(v) else None for v in oof],
                'oof_clean_margins': [float(v) if np.isfinite(v) else None for v in clean],
                'protected_count': int(protected.sum()), 'verified_protected_count': int(verified.sum()),
                'stable_protected_count': int(stable.sum()), 'oof_wrong_protected_count': int(wrong.sum()),
                'unverified_singleton_count': int((~known).sum()),
                'class_counts': [{'class': int(c), 'protected': int((verified & (labels == c)).sum()),
                                  'stable': int((stable & (labels == c)).sum())} for c in unique]})
        fingerprints = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (source, Path(__file__), ROOT/'src/certified_frfs/triangular_classifier.py')}
        out.write_text(json.dumps({'dataset': name, 'splits': records, 'fingerprints': fingerprints}, indent=2), encoding='utf-8')
        verified = sum(s['verified_protected_count'] for s in records)
        stable = sum(s['stable_protected_count'] for s in records)
        wrong = sum(s['oof_wrong_protected_count'] for s in records)
        rows.append({'dataset': name, 'verified_protected': verified,
                     'stable_fraction': stable/verified if verified else None,
                     'oof_wrong_fraction': wrong/verified if verified else None,
                     'unverified_singletons': sum(s['unverified_singleton_count'] for s in records)})
        print(name, rows[-1], flush=True)
    result = {'rows': rows,
              'mean_stable_fraction': float(np.mean([r['stable_fraction'] for r in rows if r['stable_fraction'] is not None])),
              'mean_wrong_fraction': float(np.mean([r['oof_wrong_fraction'] for r in rows if r['oof_wrong_fraction'] is not None])),
              'scope': 'reference_only_heldout_inner_margin_diagnostic_same_selected_C_exposed_development',
              'limitations': 'C is selected with these inner folds; this diagnoses instability, not unbiased risk. '
                             'An unstable anchor is not proof of label noise. Stable-mask constraints would change '
                             'the all-positive-anchor guarantee and require separate theory and experiments.'}
    (FOLDER/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k:v for k,v in result.items() if k != 'rows'})


if __name__ == '__main__':
    main()
