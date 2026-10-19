"""Posthoc test-query stability audit, never used for fitting or tuning."""
import json
import argparse
import sys
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from certified_frfs.datasets import load_dataset
from certified_frfs.query_certificate import certify_queries
from run_predictive_certified_iteration import scaled_split
threadpool_limits(1)
ap = argparse.ArgumentParser()
ap.add_argument('--iteration', type=int, default=7)
args = ap.parse_args()
directory = ROOT/f'results/predictive-iteration{args.iteration}'
records = []
for path in sorted(directory.glob('*-s*-f*.json')):
    obj = json.loads(path.read_text())
    X, y = load_dataset(obj['dataset'])
    tr, te = obj['train_indices'], obj['test_indices']
    A, B = scaled_split(X[tr], X[te])
    for row in obj['rows']:
        fs = row['features']
        if not fs:
            continue
        model = KNeighborsClassifier(n_neighbors=1, metric='manhattan', algorithm='brute').fit(A[:, fs], y[tr])
        pred = model.predict(B[:, fs])
        certificate = certify_queries(A, y[tr], B, pred, fs)
        mask = np.array(certificate['certified'])
        correct = pred == y[te]
        record = {'dataset':obj['dataset'], 'fold':obj['fold'], 'method':row['method'],
                  'coverage':certificate['coverage'],
                  'certified_correct_fraction':float(np.mean(mask & correct)),
                  'certified_wrong_fraction':float(np.mean(mask & ~correct)),
                  'uncertified_correct_fraction':float(np.mean(~mask & correct)),
                  'certificate':certificate}
        records.append(record)
    print(json.dumps({'dataset':obj['dataset'], 'fold':obj['fold'], 'status':'audited'}),flush=True)
summary = {}
for method in sorted(set(r['method'] for r in records)):
    per_dataset = {}
    for name in sorted(set(r['dataset'] for r in records)):
        rows = [r for r in records if r['method']==method and r['dataset']==name]
        per_dataset[name] = {key:float(np.mean([r[key] for r in rows])) for key in
                            ['coverage','certified_correct_fraction','certified_wrong_fraction']}
    summary[method] = {key:float(np.mean([r[key] for r in per_dataset.values()])) for key in
                       ['coverage','certified_correct_fraction','certified_wrong_fraction']}
out = directory/'query-audit.json'
out.write_text(json.dumps({'status':'posthoc_development_no_selection_or_tuning',
    'scope':'query_prediction_stability_not_population_accuracy',
    'records':records, 'dataset_balanced_means':summary},indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
