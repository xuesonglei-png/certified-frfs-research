"""Separate equal-budget predictive pilot: change loss, preserve all guards."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def save(name, text):
    path = ROOT/'scripts'/name
    if path.exists(): raise FileExistsError(path)
    path.write_text(text, encoding='utf-8')


def main():
    cap = (ROOT/'scripts/capped_predictive_metric.py').read_text(encoding='utf-8')
    cap = once(cap, 'from certified_frfs.softvote_metric import softvote_terms, softvote_objective',
               'from certified_frfs.softvote_metric import softvote_terms\nfrom soft_topk_metric_objective import soft_topk_objective')
    cap = once(cap, 'def objective(w): return softvote_objective(w, C, same, mask, positive_labels, temperature, reg)',
               'def objective(w): return soft_topk_objective(w, (C, same, mask, positive_labels), temperature, reg)')
    save('capped_soft_topk_metric.py', cap)
    stage = (ROOT/'scripts/majority_predictive_knn_stage.py').read_text(encoding='utf-8')
    stage = stage.replace('from capped_predictive_metric import', 'from capped_soft_topk_metric import')
    stage = stage.replace('from adaptive_majority_predictive_metric import', 'from soft_topk_predictive_metric import')
    stage = stage.replace("'unconstrained'", "'soft_topk_unconstrained'").replace("'majority_constrained'", "'soft_topk_constrained'")
    stage = once(stage, '            self.lookup[key] = len(self.learners)', "            fit = {**fit, 'neighborhood_k': 5}\n            self.lookup[key] = len(self.learners)")
    save('soft_topk_knn_stage.py', stage)
    runner = (ROOT/'scripts/run_majority_predictive_knn_pilot.py').read_text(encoding='utf-8')
    runner = runner.replace('from majority_predictive_knn_stage import', 'from soft_topk_knn_stage import')
    runner = runner.replace('majority-predictive-knn-pilot', 'soft-topk-knn-pilot')
    runner = runner.replace('scripts/majority_predictive_knn_stage.py', 'scripts/soft_topk_knn_stage.py')
    runner = runner.replace('scripts/adaptive_majority_predictive_metric.py', 'scripts/soft_topk_predictive_metric.py')
    runner = once(runner, "ROOT/'scripts/capped_predictive_metric.py',", "ROOT/'scripts/capped_predictive_metric.py', ROOT/'scripts/capped_soft_topk_metric.py', ROOT/'scripts/soft_topk_metric_objective.py', ROOT/'scripts/predictive_target_knn_stage.py',")
    runner = once(runner, "ROOT/'results/majority-constrained-target-admission-v3/complete-audit.json',", "ROOT/'results/majority-constrained-target-admission-v3/complete-audit.json', ROOT/'results/soft-topk-target-admission/complete-audit.json',")
    save('run_soft_topk_knn_pilot.py', runner)
    audit = (ROOT/'scripts/audit_majority_predictive_knn_pilot.py').read_text(encoding='utf-8')
    audit = audit.replace('from run_majority_predictive_knn_pilot import', 'from run_soft_topk_knn_pilot import')
    audit = audit.replace('from majority_predictive_knn_stage import GRID', 'from soft_topk_knn_stage import GRID')
    audit = audit.replace('from audit_majority_constrained_target_admission import independent_objective',
                          'from independent_soft_topk_objective import independent_objective')
    audit = once(audit, "    ordinary = [r for r in block['learners'] if r['family'] == 'unconstrained']", '    ordinary = []')
    audit = once(audit, "        if r['family'] == 'unconstrained': continue\n        assert r['family'] == 'majority_constrained'", "        assert r['family'] in ('soft_topk_unconstrained', 'soft_topk_constrained')")
    audit = once(audit, "        if fit['polytope'] is None:", '''        assert fit['neighborhood_k'] == 5
        if r['family'] == 'soft_topk_unconstrained':
            if fit['status'] == 'unique_capped_simplex':
                assert abs(len(fs)*2/d-1) <= 1e-12 and fit['optimizer_iterations'] == 0
                np.testing.assert_allclose(target, np.full(len(fs), mass/len(fs)), atol=1e-10)
            elif fit['status'] == 'no_LOO_same_class_supervision':
                _, numbers = np.unique(y, return_counts=True); assert np.all(numbers == 1)
                equivalent(anchor.tolist(), target.tolist())
            else:
                start = (1-1e-5)*anchor/mass+1e-5*np.ones(len(fs))/len(fs)
                initial = independent_objective(A[:, fs], y, start)
                equivalent(initial, fit['training_objective_initial'])
                if fit['status'] == 'local_predictive_target':
                    assert fit['optimizer_success'] and fit['optimizer_status'] == 0
                    final = independent_objective(A[:, fs], y, target/mass)
                    equivalent(final, fit['training_objective_final']); assert final <= initial+1e-8
                else:
                    assert 'fallback' in fit['status']; equivalent(anchor.tolist(), target.tolist())
            counts['independent_target_objective_replays'] += 1
            counts['target_optimizer_failures'] += not fit['optimizer_success']
            continue
        if fit['polytope'] is None:''')
    save('audit_soft_topk_knn_pilot.py', audit)


if __name__ == '__main__': main()
