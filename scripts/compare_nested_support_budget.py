"""Same raw counts and common fuzzy classifier; explicit budget adaptations."""
import argparse
import hashlib
import json
from pathlib import Path

from run_decision_iteration import ROOT
import numpy as np
from threadpoolctl import threadpool_limits

from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from run_common_fuzzy_classifier import train, score, final_metrics, choose
from run_realized_budget_diagnostic import GRIDS, PATHS, CS
from review_box_iterations import paired


def run(source_name):
    threadpool_limits(1)
    source_folder = ROOT / 'results' / source_name
    folder = ROOT / 'results' / (source_name + '-common-budget')
    folder.mkdir(exist_ok=True)
    summary = json.loads((source_folder / 'complete.json').read_text())
    rows = []
    for row in summary['rows']:
        name = row['dataset']
        out = folder / (name + '.json')
        if out.exists():
            raise FileExistsError(out)
        source = source_folder / (name + '.json')
        paths_source = ROOT / 'results/common-fuzzy-classifier' / (name + '.json')
        saved, paths_saved = [json.loads(p.read_text()) for p in (source, paths_source)]
        X, y = load_dataset(name)
        records = []
        for index, (split, control) in enumerate(zip(saved['splits'], paths_saved['splits'])):
            assert split['reference_indices'] == control['reference_indices']
            assert split['validation_indices'] == control['validation_indices']
            ri, vi = np.array(split['reference_indices']), np.array(split['validation_indices'])
            raw, labels = X[ri], y[ri]
            chosen, teacher_C = split['chosen'], str(split['selected_config']['C'])
            trace = {m: [{'config': cfg, 'C': C, 'folds': []} for cfg in GRIDS[m] for C in CS]
                     for m in GRIDS}
            trace['proposed'] = [{'C': C, 'folds': []} for C in CS]
            for inner, prior in zip(split['inner'], control['inner']):
                assert inner['reference_indices'] == prior['reference_indices']
                assert inner['validation_indices'] == prior['validation_indices']
                a, b = np.array(inner['reference_indices']), np.array(inner['validation_indices'])
                A, B = preprocess_fit(raw[a], raw[b])
                fs = inner['by_C'][teacher_C]['stage']['policies'][chosen]['candidate']['features']
                for method, options in trace.items():
                    for j, option in enumerate(options):
                        support = fs if method == 'proposed' else prior['paths'][method][j//len(CS)]['ranking'][:len(fs)]
                        fit = train(A, labels[a], support, option['C'])
                        option['folds'].append({'features': support, 'clean': score(B, labels[b], fit)})
            A, B = preprocess_fit(raw, X[vi])
            fs = split['stage']['policies'][chosen]['candidate']['features']
            results = {}
            for method, options in trace.items():
                selected = choose(options)
                option = options[selected]
                path = None if method == 'proposed' else PATHS[method](A, labels, **option['config'])
                support = fs if method == 'proposed' else path['ranking'][:len(fs)]
                assert len(support) == len(fs)
                fit, metrics = final_metrics(A, labels, B, y[vi], support, option['C'], 144 + index)
                results[method] = {'chosen_index': selected, 'features': support, 'fit': fit,
                                   'metrics': metrics, 'path': path}
            records.append({'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(),
                            'target_size': len(fs), 'trace': trace, 'results': results})
        files = [source, paths_source, Path(__file__), ROOT / 'src/certified_frfs/recent_budget.py',
                 ROOT / 'scripts/run_common_fuzzy_classifier.py', ROOT / 'data/processed' / (name + '.npz')]
        out.write_text(json.dumps({'dataset': name, 'splits': records,
            'fingerprints': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in files}}, indent=2), encoding='utf-8')
        methods = {m: float(np.mean([s['results'][m]['metrics']['clean'] for s in records]))
                   for m in trace}
        rows.append({'dataset': name, 'methods': methods, 'removed': row['removed']})
        print(name, 'exact common-classifier attribute budgets completed', flush=True)
    result = {'rows': rows,
              'means': {m: float(np.mean([r['methods'][m] for r in rows])) for m in rows[0]['methods']},
              'comparisons': {m: paired([r['methods']['proposed'] - r['methods'][m] for r in rows])
                              for m in GRIDS},
              'scope': 'same_raw_attribute_counts_common_classifier_frozen_proposed_policy_development_ablation',
              'limitations': 'FGS ranking completion and SFRIP continuation are explicit budget adaptations, '
                             'not original native algorithms. Proposed policy remains LP-selected.'}
    (folder / 'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k: v for k, v in result.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='nested-triangular-repair')
    run(parser.parse_args().source)
