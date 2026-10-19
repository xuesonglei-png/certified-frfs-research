"""Training-only edge-complexity diagnosis; no query accuracy or tuning."""
import json
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from sklearn.metrics import mutual_info_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_classifier import triangular_memberships


def edge_statistics(A, y):
    bins = triangular_memberships(A).argmax(axis=-1)
    n, d = A.shape
    q = len(np.unique(y))
    singles = [mutual_info_score(bins[:, a], y) for a in range(d)]
    categories = [len(np.unique(bins[:, a])) for a in range(d)]
    records = []
    for u in range(d):
        for v in range(u+1, d):
            joint = 3*bins[:, u]+bins[:, v]
            gain = mutual_info_score(joint, y)-max(singles[u], singles[v])
            # Conventional conditional-multinomial BIC count. A heuristic for
            # observed empty cells, not calibrated multiple-comparison testing.
            df = (len(np.unique(joint))-max(categories[u], categories[v]))*(q-1)
            penalty = df*np.log(n)/(2*n)
            records.append({'edge': [u, v], 'gain': float(gain), 'df': int(df),
                            'penalty': float(penalty), 'adjusted_gain': float(gain-penalty)})
    return records


def kruskal(d, edges, *, penalized):
    ordered = sorted(edges, key=lambda r: (-(r['adjusted_gain'] if penalized else max(0., r['gain'])), r['edge']))
    parent, selected = list(range(d)), []
    def root(a):
        while parent[a] != a:
            a = parent[a]
        return a
    for edge in ordered:
        if penalized and edge['adjusted_gain'] <= 0:
            break
        u, v = edge['edge']
        a, b = root(u), root(v)
        if a != b:
            parent[a] = b
            selected.append(edge)
    return selected


def real_reference(name):
    threadpool_limits(1)
    saved = json.loads((ROOT/'results/adaptive-multiknot-recent-pilot'/(name+'.json')).read_text())
    X, y = load_dataset(name)
    ri = np.array(saved['reference_indices'])
    rows = []
    for index, block in enumerate(saved['inner']+[saved['outer']]):
        a = np.array(block['reference_indices']) if index < len(saved['inner']) else np.arange(len(ri))
        ref, labels = X[ri[a]], y[ri[a]]
        A, _ = preprocess_fit(ref, ref)
        records = edge_statistics(A, labels)
        old, new = kruskal(A.shape[1], records, penalized=False), kruskal(A.shape[1], records, penalized=True)
        assert [r['edge'] for r in old] == block['stage']['tree']['edges']
        rows.append({'dataset': name, 'stage_index': index, 'n': len(ref), 'd': A.shape[1],
                     'old_edges': len(old), 'adjusted_edges': len(new),
                     'old_edges_nonpositive_adjusted': sum(r['adjusted_gain'] <= 0 for r in old)})
    print(name, 'training edge diagnosis complete', flush=True)
    return rows


def null_case(n, q):
    threadpool_limits(1)
    rows = []
    for rep in range(16):
        rng = np.random.default_rng(20261040+1000*n+100*q+rep)
        A = rng.uniform(size=(n, 12))
        y = np.arange(n) % q
        rng.shuffle(y)
        records = edge_statistics(A, y)
        rows.append({'n': n, 'q': q, 'rep': rep, 'old_edges': 11,
                     'adjusted_edges': len(kruskal(12, records, penalized=True))})
    return rows


if __name__ == '__main__':
    names = [r['dataset'] for r in json.loads((ROOT/'results/adaptive-multiknot-recent-pilot/complete.json').read_text())['rows']]
    real, null = [], []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(real_reference, name) for name in names]):
            real.extend(future.result())
        for future in as_completed([pool.submit(null_case, n, q) for n in (64, 128, 256) for q in (2, 3)]):
            null.extend(future.result())
    result = {'real_training_references': sorted(real, key=lambda r: (r['dataset'], r['stage_index'])),
              'independent_null_simulations': sorted(null, key=lambda r: (r['n'], r['q'], r['rep'])),
              'counts': {'real_stages': len(real), 'old_real_edges': sum(r['old_edges'] for r in real),
                         'adjusted_real_edges': sum(r['adjusted_edges'] for r in real),
                         'nonpositive_old_real_edges': sum(r['old_edges_nonpositive_adjusted'] for r in real),
                         'null_replicates': len(null), 'old_null_edges': sum(r['old_edges'] for r in null),
                         'adjusted_null_edges': sum(r['adjusted_edges'] for r in null)},
              'query_labels_or_accuracy_used': False,
              'scope': 'training_structure_diagnosis_not_prediction_or_novelty_or_FDR_guarantee'}
    path = ROOT/'results/fuzzy-forest-edge-complexity-diagnosis.json'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result['counts'], flush=True)
