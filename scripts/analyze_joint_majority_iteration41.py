"""Complete-only dataset-level paired analysis; no fold-as-dataset tests."""
import json, hashlib
import numpy as np
from scipy.stats import wilcoxon
from run_decision_iteration import ROOT


DATASETS = ['banknote', 'ecoli', 'glass', 'haberman', 'heart_statlog', 'ionosphere',
    'iris', 'seeds', 'sonar', 'transfusion', 'vehicle', 'vertebral', 'wdbc', 'wine', 'yeast']


def paired_test(delta, alternative):
    delta = np.round(delta, 12)
    if np.all(abs(delta) < 1e-12): return 1.
    return float(wilcoxon(delta, alternative=alternative, zero_method='wilcox', method='auto').pvalue)


def main():
    folder = ROOT/'results/predictive-iteration41'; paths = [folder/f'{n}-s101-f{f}.json' for n in DATASETS for f in range(3)]
    missing = [p.name for p in paths if not p.exists()]
    if missing: raise RuntimeError(f'incomplete: {len(missing)} tasks missing; no complete analysis produced')
    audits = []
    for path in paths:
        audit_path = folder/'audits'/path.name
        if not audit_path.exists(): raise RuntimeError(f'missing score/certificate audit: {path.name}')
        checked = json.loads(audit_path.read_text())
        assert checked['result_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        audits.append(checked)
    rows = []; sources = {}
    for name in DATASETS:
        folds = []
        for fold in range(3):
            path = folder/f'{name}-s101-f{fold}.json'; p = json.loads(path.read_text()); sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            values = {r['method']: r['metrics'] for r in p['rows']+p['baseline_rows']}
            sizes = {r['method']: len(r['fit']['features']) for r in p['rows']}
            fallback = next(r for r in p['rows'] if r['method'] == 'certified_sparse')['full_fallback']
            assert next(r for r in p['rows'] if r['method'] == 'certified_sparse')['fit']['certificate']['feasible']
            folds.append({'metrics': values, 'sizes': sizes, 'd': len(next(r for r in p['rows'] if r['method'] == 'sparse')['fit']['weights']),
                          'certified_full_fallback': fallback, 'seconds': p['seconds']})
        methods = list(folds[0]['metrics'])
        rows.append({'dataset': name, 'folds': folds,
            'metrics': {m: {k: float(np.mean([f['metrics'][m][k] for f in folds])) for k in ['clean', 'noise01']} for m in methods},
            'removed_fraction': {m: float(np.mean([1-f['sizes'][m]/f['d'] for f in folds])) for m in ['sparse', 'certified_sparse', 'unlimited_joint_metric']}})
    pairs = [('sparse', 'fgs'), ('sparse', 'all'), ('certified_sparse', 'sparse')]; tests = []
    rng = np.random.default_rng(1005); samples = rng.integers(0, len(rows), size=(10000, len(rows)))
    for method, baseline in pairs:
        delta = np.array([r['metrics'][method]['clean']-r['metrics'][baseline]['clean'] for r in rows])
        interval = np.quantile(delta[samples].mean(axis=1), [.025, .975])
        tests.append({'method': method, 'baseline': baseline, 'mean_gain': float(delta.mean()), 'dataset_deltas': delta.tolist(),
            'wins': int(sum(delta > 1e-12)), 'ties': int(sum(abs(delta) <= 1e-12)), 'losses': int(sum(delta < -1e-12)),
            'raw_one_sided_p': paired_test(delta, 'greater'), 'raw_two_sided_p': paired_test(delta, 'two-sided'),
            'bootstrap_mean_gain_interval95': interval.tolist()})
    order = np.argsort([r['raw_one_sided_p'] for r in tests], kind='stable'); last = 0.
    for rank, index in enumerate(order):
        last = max(last, min(1., (len(tests)-rank)*tests[index]['raw_one_sided_p']))
        tests[index]['holm_one_sided_p'] = last
    prediction_gate = all(r['mean_gain'] >= .005 and r['holm_one_sided_p'] < .05 for r in tests[:2])
    methods = list(rows[0]['metrics']); means = {m: {k: float(np.mean([r['metrics'][m][k] for r in rows])) for k in ['clean', 'noise01']} for m in methods}
    payload = {'scope': 'complete45task15dataset_repeated_development_not_untouched_confirmation',
        'rows': rows, 'means': means, 'primary_tests': tests, 'predictive_development_gate_passed': prediction_gate,
        'source_sha256': sources, 'uncertified_success_does_not_complete_research_goal': True,
        'audited_inner_scores': sum(a['inner_scores_checked'] for a in audits),
        'audited_outer_scores': sum(a['outer_scores_checked'] for a in audits),
        'sampled_loss_checks': sum(a['sampled_loss_checks'] for a in audits),
        'mean_removed_fraction': {m: float(np.mean([r['removed_fraction'][m] for r in rows])) for m in ['sparse', 'certified_sparse', 'unlimited_joint_metric']},
        'limitations': 'Development reuse;cached independent baselines have fidelity/budget gaps;selected certificates/scoring audited but independent optimizer and all-inner certification audits still needed'}
    out = folder/'complete-development-analysis.json'
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8'); print(json.dumps(means), json.dumps(tests), flush=True)


if __name__ == '__main__': main()
