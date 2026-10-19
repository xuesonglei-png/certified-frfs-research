"""All44 reference-only mechanism diagnostic; deliberately no test labels."""
import hashlib
import json
from pathlib import Path

from run_decision_iteration import ROOT
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.certificate_repair import repair_support

FOLDER = ROOT / 'results/certificate-repair-mechanism'


def main():
    threadpool_limits(1)
    FOLDER.mkdir(exist_ok=True)
    summary = json.loads((ROOT / 'results/nested-triangular/complete.json').read_text())
    rows = []
    for row in summary['rows']:
        name = row['dataset']
        out = FOLDER / (name + '.json')
        if out.exists():
            raise FileExistsError(out)
        source = ROOT / 'results/nested-triangular' / (name + '.json')
        saved = json.loads(source.read_text())
        X, y = load_dataset(name)
        records = []
        for i, split in enumerate(saved['splits']):
            ri = np.array(split['reference_indices'])
            A, _ = preprocess_fit(X[ri], X[ri])
            stage = split['stage']
            trials = []
            for fraction in ('0.5', '0.65', '0.8'):
                record = stage['policies'][fraction]
                failure = record['failure']
                candidate = failure['requested_candidate'] if failure and 'requested_candidate' in failure else record['candidate']
                repaired = repair_support(A, y[ri], stage['teacher_coef'], candidate['features'])
                trials.append({'fraction': fraction, 'initial_features': candidate['features'],
                               'old_actual_features': record['candidate']['features'], 'repaired': repaired})
            records.append({'reference_indices': ri.tolist(), 'trials': trials})
            print(name, i, 'reference-only certificate repairs completed', flush=True)
        out.write_text(json.dumps({'dataset': name, 'splits': records,
            'fingerprints': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (source, Path(__file__), ROOT / 'src/certified_frfs/certificate_repair.py')}}, indent=2), encoding='utf-8')
        rows.append({'dataset': name,
                     'old_actual_removed': float(np.mean([1-len(t['old_actual_features'])/X.shape[1] for s in records for t in s['trials']])),
                     'repaired_removed': float(np.mean([1-len(t['repaired']['features'])/X.shape[1] for s in records for t in s['trials']]))})
    report = {'rows': rows, 'scope': 'reference_only_mechanism_no_prediction_evaluation_no_selection',
              'warning': 'Repairs pass metric certificates only; deployed classifier constraints still require independent refitting.'}
    (FOLDER / 'complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(report, flush=True)


if __name__ == '__main__':
    main()
