"""Freeze nested-LP supports; independently tune a common fuzzy classifier."""
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT

from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from run_common_fuzzy_classifier import train, score, final_metrics
from review_box_iterations import paired

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/nested-triangular'
FOLDER = ROOT / 'results/nested-triangular-common-classifier-ablation'
CS = (.1, 1., 10.)


def run():
    threadpool_limits(1)
    FOLDER.mkdir(exist_ok=True)
    summary = json.loads((SOURCE / 'complete.json').read_text())
    controls = json.loads((ROOT / 'results/common-fuzzy-classifier/complete-fuzzy-review.json').read_text())
    byname = {r['dataset']: r for r in controls['rows']}
    rows = []
    for row in summary['rows']:
        name = row['dataset']
        out = FOLDER / (name + '.json')
        if out.exists():
            raise FileExistsError(out)
        source = SOURCE / (name + '.json')
        saved = json.loads(source.read_text())
        X, y = load_dataset(name)
        records = []
        for index, split in enumerate(saved['splits']):
            ri, vi = np.array(split['reference_indices']), np.array(split['validation_indices'])
            raw, labels = X[ri], y[ri]
            chosen = split['chosen']
            teacher_C = str(split['selected_config']['C'])
            trace = []
            for C in CS:
                folds = []
                for inner in split['inner']:
                    a, b = np.array(inner['reference_indices']), np.array(inner['validation_indices'])
                    A, B = preprocess_fit(raw[a], raw[b])
                    fs = inner['by_C'][teacher_C]['stage']['policies'][chosen]['candidate']['features']
                    fit = train(A, labels[a], fs, C)
                    folds.append({'features': fs, 'clean': score(B, labels[b], fit)})
                trace.append({'C': C, 'folds': folds,
                              'mean_clean': float(np.mean([f['clean'] for f in folds]))})
            selected = min(trace, key=lambda r: (-r['mean_clean'], r['C']))
            A, B = preprocess_fit(raw, X[vi])
            fs = split['stage']['policies'][chosen]['candidate']['features']
            fit, metrics = final_metrics(A, labels, B, y[vi], fs, selected['C'], 144 + index)
            records.append({'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(),
                            'features': fs, 'trace': trace, 'selected_C': selected['C'],
                            'fit': fit, 'metrics': metrics})
        fingerprints = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (source, Path(__file__), ROOT / 'scripts/run_common_fuzzy_classifier.py',
                                  ROOT / 'data/processed' / (name + '.npz'))}
        out.write_text(json.dumps({'dataset': name, 'splits': records,
                                  'fingerprints': fingerprints}, indent=2), encoding='utf-8')
        control = byname[name]['methods']
        rows.append({'dataset': name, 'clean': float(np.mean([r['metrics']['clean'] for r in records])),
                     'proposed_LP_clean': row['clean'], 'removed': row['removed'],
                     'controls': {m: control[m]['clean'] for m in ('full', 'fgs2023_native', 'sfrip2025_native')}})
        print(name, 'frozen-support classifier ablation complete', flush=True)
    result = {'rows': rows, 'mean_clean': float(np.mean([r['clean'] for r in rows])),
              'vs_LP': paired([r['clean'] - r['proposed_LP_clean'] for r in rows]),
              'vs_common_controls': {m: paired([r['clean'] - r['controls'][m] for r in rows])
                                     for m in rows[0]['controls']},
              'scope': 'frozen_nested_LP_supports_common_fuzzy_classifier_postselection_ablation_development',
              'limitation': 'Supports and policy were selected using LP inner scores. This ablation '
                            'does not replace an independently selected common-classifier pipeline.'}
    (FOLDER / 'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k: v for k, v in result.items() if k != 'rows'})


if __name__ == '__main__':
    run()
