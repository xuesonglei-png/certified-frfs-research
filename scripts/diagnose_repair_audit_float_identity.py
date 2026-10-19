"""Measure independent raw-imputation versus cached nodal-sum roundoff."""
import json
from pathlib import Path
import numpy as np
from audit_repaired_crossfit_recent_pilot import direct_projection
from run_repaired_crossfit_recent_pilot import FOLDER
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

ROOT = Path(__file__).resolve().parents[1]


if __name__ == '__main__':
    summary = json.loads((FOLDER/'complete.json').read_text())
    rows = []
    for row in summary['rows']:
        saved = json.loads((FOLDER/(row['dataset']+'.json')).read_text())
        X, y = load_dataset(row['dataset'])
        ri = np.array(saved['reference_indices'])
        for stage_index, block in enumerate(saved['inner']+[saved['outer']]):
            if stage_index < len(saved['inner']):
                a = np.array(block['reference_indices'])
                raw, train = X[ri[a]], y[ri[a]]
            else:
                raw, train = X[ri], y[ri]
            stage = block['stage']
            selected = stage['teacher_selection']['chosen_index']
            fold_tables = []
            for fold in stage['teacher_selection']['folds']:
                a, b = np.array(fold['reference_indices']), np.array(fold['validation_indices'])
                A, B = preprocess_fit(raw[a], raw[b])
                alpha = np.array([1/(len(np.unique(train[a]))*sum(train[a] == c)) for c in train[a]])
                fold_tables.append((B, train[b], fold['models'][selected], alpha@A))
            for repair in stage['path']['repair_blocks']:
                ranking = stage['path']['rankings'][repair['ranking_index']]
                for index, record in enumerate(repair['attempts']):
                    assert record['removed'] == ranking['choices'][index]['removed']
                    fs = [a for a in ranking['features'] if a != record['removed']]
                    direct = float(np.mean([direct_projection(B, labels, fit, center, fs) for B, labels, fit, center in fold_tables]))
                    rows.append({'dataset': row['dataset'], 'stage_index': stage_index,
                                 'removed': record['removed'], 'saved_loss': record['loss'],
                                 'independent_loss': direct, 'absolute_error': abs(record['loss']-direct)})
    result = {'rows': rows, 'max_absolute_error': max(r['absolute_error'] for r in rows),
              'all_deletion_identifiers_identical': True,
              'scope': 'independent_scalar_roundoff_diagnosis_no_algorithm_score_or_scientific_threshold_change'}
    (ROOT/'results/repair-audit-float-identity-diagnosis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k: v for k, v in result.items() if k != 'rows'})
