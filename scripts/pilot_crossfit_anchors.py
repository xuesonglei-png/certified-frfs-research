"""Frozen supports with a weaker explicitly crossfitted margin scope."""
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.crossfit_anchor_mask import crossfit_anchor_mask
from certified_frfs.triangular_classifier import triangular_box_margins
from run_common_fuzzy_classifier import train, score
from review_box_iterations import paired

SOURCE = ROOT/'results/nested-triangular-repair'
FOLDER = ROOT/'results/crossfit-anchor-pilot'


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
        splits = []
        for index, split in enumerate(saved['splits']):
            ri, vi = np.array(split['reference_indices']), np.array(split['validation_indices'])
            raw, labels = X[ri], y[ri]
            A, B = preprocess_fit(raw, X[vi])
            stage = split['stage']
            C = stage['teacher_C']
            stability = crossfit_anchor_mask(raw, labels, C=C)
            original = triangular_box_margins(A, labels, stage['teacher_coef'],
                stage['teacher_intercept'], stage['teacher_classes'], eps=.01)
            protected = (original > 1e-10) & np.array(stability['mask'])
            policy = stage['policies'][split['chosen']]
            candidate = policy['candidate']
            target = train(A, labels, candidate['features'], C)
            margins = triangular_box_margins(A, labels, target['coef'], target['intercept'],
                                             target['classes'], eps=.01)
            accepted = bool(np.all(margins[protected] >= .5*original[protected]-1e-8)
                            and np.all(margins[protected] > 1e-10))
            full = train(A, labels, list(range(X.shape[1])), C)
            actual, fs = (target, candidate['features']) if accepted else (full, list(range(X.shape[1])))
            splits.append({'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(),
                'teacher_C': C, 'stability': stability, 'original_margins': original.tolist(),
                'protected_mask': protected.tolist(), 'candidate': candidate, 'accepted': accepted,
                'fit': actual, 'full_fit': full, 'features': fs,
                'target_margins': margins.tolist(), 'clean': score(B, y[vi], actual),
                'full_clean': score(B, y[vi], full),
                'excluded_positive_count': int(((original > 1e-10) & ~protected).sum())})
            print(name, index, 'crossfit scope pilot', accepted, len(fs), flush=True)
        files = [source, Path(__file__), ROOT/'src/certified_frfs/crossfit_anchor_mask.py',
                 ROOT/'research/crossfit-anchor-pilot-protocol.md']
        out.write_text(json.dumps({'dataset': name, 'd': X.shape[1], 'splits': splits,
            'fingerprints': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in files}}, indent=2), encoding='utf-8')
        rows.append({'dataset': name, 'clean': float(np.mean([s['clean'] for s in splits])),
            'full': float(np.mean([s['full_clean'] for s in splits])),
            'removed': float(np.mean([1-len(s['features'])/X.shape[1] for s in splits])),
            'excluded_positive': sum(s['excluded_positive_count'] for s in splits)})
    result = {'rows': rows, 'mean_clean': float(np.mean([r['clean'] for r in rows])),
        'mean_removed': float(np.mean([r['removed'] for r in rows])),
        'vs_full': paired([r['clean']-r['full'] for r in rows]),
        'scope': 'frozen_LP_selected_support_C_crossfit_anchor_screening_diagnostic_weaker_guarantee'}
    (FOLDER/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k:v for k,v in result.items() if k!='rows'})


if __name__ == '__main__':
    main()
