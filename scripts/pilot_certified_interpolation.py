"""Cached certified anchors, fresh reference-only logistic targets and selection."""
import copy
import hashlib
import json
from pathlib import Path
from run_decision_iteration import ROOT
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.certified_score_interpolation import certified_step
from run_common_fuzzy_classifier import train
from run_nested_triangular import score
from review_box_iterations import paired

SOURCE = ROOT / 'results/nested-triangular-repair'
FOLDER = ROOT / 'results/certified-interpolation-pilot'


def update_stage(raw, y, saved):
    stage = copy.deepcopy(saved)
    A, _ = preprocess_fit(raw, raw)
    seen = {}
    for policy in stage['policies'].values():
        key = tuple(policy['candidate']['features'])
        if key not in seen:
            target = train(A, y, list(key), stage['teacher_C'])
            anchor = policy['fit']
            original = np.array(anchor.get('original_margins', []))
            if not len(original):
                from certified_frfs.triangular_classifier import triangular_box_margins
                original = triangular_box_margins(A, y, stage['teacher_coef'],
                    stage['teacher_intercept'], stage['teacher_classes'], eps=.01)
            fitted = certified_step(A, y, anchor['coef'], anchor['intercept'],
                target['coef'], target['intercept'], stage['teacher_classes'], original)
            seen[key] = {**anchor, **fitted, 'feasible': True,
                         'anchor_coef': anchor['coef'], 'anchor_intercept': anchor['intercept'],
                         'target_coef': target['coef'], 'target_intercept': target['intercept'],
                         'original_margins': original.tolist()}
        policy['fit'] = seen[key]
    return stage


def main():
    threadpool_limits(1)
    FOLDER.mkdir(exist_ok=True)
    summary = json.loads((SOURCE / 'complete.json').read_text())
    rows = []
    for row in summary['rows']:
        name = row['dataset']
        source = SOURCE / (name+'.json')
        out = FOLDER / (name+'.json')
        if out.exists():
            raise FileExistsError(out)
        saved = json.loads(source.read_text())
        X, y = load_dataset(name)
        splits = []
        for i, prior in enumerate(saved['splits']):
            ri, vi = np.array(prior['reference_indices']), np.array(prior['validation_indices'])
            raw, labels = X[ri], y[ri]
            C = str(prior['selected_config']['C'])
            inner_records = []
            for inner in prior['inner']:
                a, b = np.array(inner['reference_indices']), np.array(inner['validation_indices'])
                stage = update_stage(raw[a], labels[a], inner['by_C'][C]['stage'])
                values = score(raw[a], labels[a], raw[b], labels[b], stage)
                inner_records.append({'reference_indices': a.tolist(), 'validation_indices': b.tolist(),
                                      'stage': stage, 'scores': values})
            options = []
            for key in ('0.5', '0.65', '0.8', '1.0'):
                options.append({'fraction': key,
                    'clean': float(np.mean([s['scores'][key]['clean'] for s in inner_records])),
                    'retained': float(np.mean([s['scores'][key]['retained'] for s in inner_records]))})
            best = max(s['clean'] for s in options)
            selected = min((s for s in options if s['clean'] >= best-.005-1e-12),
                           key=lambda s: (s['retained'], -s['clean'], float(s['fraction'])))
            stage = update_stage(raw, labels, prior['outer_stages'][C])
            values = score(raw, labels, X[vi], y[vi], stage, noise=True, seed=144+i)
            splits.append({'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(),
                'teacher_C': float(C), 'inner': inner_records, 'options': options,
                'chosen': selected['fraction'], 'stage': stage, 'metrics': values[selected['fraction']],
                'full_metrics': values['1.0']})
            print(name, i, 'certified direction pilot completed', flush=True)
        files = [source, Path(__file__), ROOT / 'src/certified_frfs/certified_score_interpolation.py',
                 ROOT / 'research/certified-interpolation-protocol.md']
        out.write_text(json.dumps({'dataset': name, 'splits': splits,
            'fingerprints': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in files}}, indent=2), encoding='utf-8')
        rows.append({'dataset': name, 'clean': float(np.mean([s['metrics']['clean'] for s in splits])),
                     'full': float(np.mean([s['full_metrics']['clean'] for s in splits])),
                     'removed': float(np.mean([1-s['metrics']['retained'] for s in splits]))})
    report = {'rows': rows, 'mean_clean': float(np.mean([r['clean'] for r in rows])),
              'mean_removed': float(np.mean([r['removed'] for r in rows])),
              'vs_full': paired([r['clean']-r['full'] for r in rows]),
              'scope': 'cached_audited_LP_anchors_frozen_inner_LP_C_reselected_fraction_direction_diagnostic'}
    (FOLDER / 'complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({k:v for k,v in report.items() if k!='rows'})


if __name__ == '__main__':
    main()
