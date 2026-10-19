"""Fixed selected-support ablation, not a classifier or method selector."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from compare_quantile_granulation import ROOT, FOLDER
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit, validation_scores


def main():
    threadpool_limits(1); complete = json.loads((FOLDER/'complete.json').read_text()); rows = []
    for row in complete['rows']:
        name = row['dataset']; p = json.loads((FOLDER/f'{name}.json').read_text()); X, y = load_dataset(name); records = []
        for index, s in enumerate(p['splits']):
            ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices'])
            A, B = preprocess_fit(X[ri], X[vi], geometry='quantile')
            fs = s['fit']['compact_selected']['features']
            metrics = validation_scores(A, y[ri], B, y[vi], fs, np.ones(X.shape[1])/np.sqrt(X.shape[1]), 144+index, joint=True)
            records.append({'split': index, 'uniform_selected_support': metrics,
                            'learned_metric': s['metrics']['proposed'], 'uniform_full': s['metrics']['full'],
                            'removed': 1-len(fs)/X.shape[1]})
        means = {m: float(np.mean([(r[m]['clean']+r[m]['noise01'])/2 for r in records]))
                 for m in ('uniform_selected_support', 'learned_metric', 'uniform_full')}
        rows.append({'dataset': name, 'splits': records, 'utility': means,
                     'full_fallbacks': sum(r['removed'] == 0 for r in records)})
    result = {'rows': rows,
              'means': {m: float(np.mean([r['utility'][m] for r in rows])) for m in rows[0]['utility']},
              'full_fallbacks': sum(r['full_fallbacks'] for r in rows),
              'scope': 'posthoc_fixed_support_metric_attribution_exposed_development_not_selection_or_causal_proof'}
    (FOLDER/'complete-attribution.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
