"""Training-only diagnosis of an equivalent nominal rival epigraph."""
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT, digest
from screen_robust_logloss_classifier import FOLDER
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.forest_robust_logloss_rival_epigraph import fit_robust_forest_logloss


if __name__ == '__main__':
    threadpool_limits(1)
    summary = json.loads((FOLDER/'complete.json').read_text())
    rows = []
    paths = [Path(__file__), ROOT/'scripts/prepare_rival_epigraph_robust_logloss.py',
             ROOT/'src/certified_frfs/forest_robust_logloss.py',
             ROOT/'src/certified_frfs/forest_robust_logloss_rival_epigraph.py',
             ROOT/'research/robust-logloss-rival-epigraph-diagnosis-protocol.md', FOLDER/'complete.json']
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    target = ROOT/'results/robust-logloss-rival-epigraph-diagnosis.json'
    for item in summary['rows']:
        name = item['dataset']
        artifact = FOLDER/(name+'.json')
        saved = json.loads(artifact.read_text())
        if not any(f['ordinary_fallback'] for f in saved['folds']):
            continue
        hashes[artifact.relative_to(ROOT).as_posix()] = digest(artifact)
        X, y = load_dataset(name)
        ri, a = np.array(saved['outer_reference_indices']), np.array(saved['reference_indices'])
        raw, labels = X[ri[a]], y[ri[a]]
        for index, fold in enumerate(saved['folds']):
            if not fold['ordinary_fallback']:
                continue
            train = np.array(fold['reference_indices'])
            A, _ = preprocess_fit(raw[train], raw[train])
            start = time.perf_counter()
            fit = fit_robust_forest_logloss(A, labels[train], list(range(A.shape[1])),
                                          fold['tree']['edges'], C=1., eps=.01)
            rows.append({'dataset': name, 'fold': index, 'original_status': fold['fit']['status'],
                         'rival_fit': fit, 'seconds': time.perf_counter()-start})
            result = {'fingerprints': hashes, 'rows': rows, 'complete': False,
                      'validation_scores_used': False, 'original_results_overwritten': False,
                      'scope': 'equivalent_rival_epigraph_numerical_diagnosis_not_accuracy_gain'}
            target.write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(name, index, 'old', fold['fit']['status'], 'rival', fit['status'],
                  'usable', fit['usable'], 'seconds', round(rows[-1]['seconds'], 2), flush=True)
    assert len(rows) == summary['counts']['new_failures']
    result.update({'complete': True, 'recovered': sum(r['rival_fit']['usable'] for r in rows),
                   'remaining_failed': sum(not r['rival_fit']['usable'] for r in rows)})
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k: v for k, v in result.items() if k not in ('fingerprints', 'rows')}, flush=True)
