"""Crossfit predictive supports plus original dual certification diagnostic."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from run_decision_iteration import ROOT
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.predictive_deletion_support import predictive_supports
from certified_frfs.certificate_repair import repair_support
from certified_frfs.triangular_classifier import triangular_box_margins
from run_common_fuzzy_classifier import train, score
from review_box_iterations import paired

SOURCE = ROOT/'results/nested-fuzzy-logistic'
FOLDER = ROOT/'results/predictive-deletion-pilot'


def run(name):
    threadpool_limits(1)
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
        C = split['selected_config']['C']
        teacher = train(A, labels, list(range(X.shape[1])), C)
        original = triangular_box_margins(A, labels, teacher['coef'], teacher['intercept'], teacher['classes'])
        protected = original > 1e-10
        generated, trace = predictive_supports(raw, labels, C=C)
        full_candidate = repair_support(A, labels, teacher['coef'], list(range(X.shape[1])))
        options = [{'candidate': full_candidate, 'fit': teacher,
                    'inner_balanced_accuracy': trace['full']['mean_balanced_accuracy'],
                    'requested_features': list(range(X.shape[1])), 'margin_passed': True}]
        for candidate in generated:
            repaired = repair_support(A, labels, teacher['coef'], candidate['features'])
            target = train(A, labels, repaired['features'], C)
            margin = triangular_box_margins(A, labels, target['coef'], target['intercept'], teacher['classes'])
            accepted = bool(np.all(margin[protected] >= .5*original[protected]-1e-8)
                            and np.all(margin[protected] > 1e-10))
            options.append({'candidate': repaired if accepted else full_candidate,
                'fit': target if accepted else teacher,
                'inner_balanced_accuracy': candidate['crossfit_metrics']['mean_balanced_accuracy'],
                'requested_features': candidate['features'], 'margin_passed': accepted,
                'requested_repaired_candidate': repaired, 'target_fit': target,
                'target_margins': margin.tolist()})
        best = max(r['inner_balanced_accuracy'] for r in options)
        selected = min((i for i,r in enumerate(options) if r['inner_balanced_accuracy'] >= best-.005-1e-12),
                       key=lambda i:(len(options[i]['candidate']['features']), -options[i]['inner_balanced_accuracy'], i))
        deployed = options[selected]
        splits.append({'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(),
            'teacher_C': C, 'teacher': teacher, 'original_margins': original.tolist(),
            'trace': trace, 'generated_candidates': generated, 'options': options, 'chosen_index': selected,
            'clean': score(B, y[vi], deployed['fit']), 'full_clean': score(B, y[vi], teacher)})
        print(name, index, 'predictive deletion scored', len(deployed['candidate']['features']), flush=True)
    files = [source, Path(__file__), ROOT/'src/certified_frfs/predictive_deletion_support.py',
             ROOT/'research/predictive-deletion-pilot-protocol.md']
    out.write_text(json.dumps({'dataset': name, 'd': X.shape[1], 'splits': splits,
        'fingerprints': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in files}}, indent=2), encoding='utf-8')


def summarize(names):
    rows = []
    for name in names:
        p = json.loads((FOLDER/(name+'.json')).read_text())
        splits = p['splits']
        rows.append({'dataset': name, 'clean': float(np.mean([s['clean'] for s in splits])),
            'full': float(np.mean([s['full_clean'] for s in splits])),
            'removed': float(np.mean([1-len(s['options'][s['chosen_index']]['candidate']['features'])/p['d'] for s in splits]))})
    report = {'rows': rows, 'mean_clean': float(np.mean([r['clean'] for r in rows])),
        'mean_removed': float(np.mean([r['removed'] for r in rows])),
        'vs_full': paired([r['clean']-r['full'] for r in rows]),
        'scope': 'reference_crossfit_single_deletion_rank_screened_logistic_pilot_inherited_full_C',
        'limitations': 'Inner candidate scores precede certificate repair and fallback; selection is diagnostic '
                      'and optimistically reuses ranking folds. Fresh nested pipeline must score actual repaired '
                      'deployments in independent inner validation.'}
    (FOLDER/'complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({k:v for k,v in report.items() if k!='rows'})


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    summarize(names)
