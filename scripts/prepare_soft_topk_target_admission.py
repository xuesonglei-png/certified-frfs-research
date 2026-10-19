"""Create separate training-only topk fits without touching live ancestors."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save(name, text):
    path = ROOT/'scripts'/name
    if path.exists(): raise FileExistsError(path)
    path.write_text(text, encoding='utf-8')


def main():
    core = (ROOT/'scripts/adaptive_majority_predictive_metric.py').read_text(encoding='utf-8')
    start, end = core.index('def smooth_objective('), core.index('\n\ndef anchor_witness_polytope')
    core = core[:start]+'from soft_topk_metric_objective import soft_topk_objective as smooth_objective\n'+core[end:]
    core = core.replace('Training-only local predictive optimization', 'Training-only local soft-topk predictive optimization')
    save('soft_topk_predictive_metric.py', core)
    runner = (ROOT/'scripts/run_majority_constrained_target_admission.py').read_text(encoding='utf-8')
    runner = runner.replace('from run_predictive_target_metric_admission import ROOT, FOLDER as PARENT, digest, GRID',
                            'from run_majority_constrained_target_admission import ROOT, FOLDER as PARENT, digest, GRID')
    runner = runner.replace('from majority_constrained_predictive_metric import', 'from soft_topk_predictive_metric import')
    runner = runner.replace('majority-constrained-target-admission-v3', 'soft-topk-target-admission')
    runner = runner.replace('scripts/majority_constrained_predictive_metric.py', 'scripts/soft_topk_predictive_metric.py')
    runner = runner.replace("ROOT/'scripts/capped_predictive_metric.py',", "ROOT/'scripts/capped_predictive_metric.py', ROOT/'scripts/soft_topk_metric_objective.py',")
    runner = runner.replace('research/majority-constrained-target-admission-protocol.md', 'research/soft-topk-target-admission-protocol.md')
    runner = runner.replace('training_constrained_target_admission_not_prediction', 'training_soft5neighbor_target_admission_not_prediction')
    save('run_soft_topk_target_admission.py', runner)
    audit = (ROOT/'scripts/audit_majority_constrained_target_admission.py').read_text(encoding='utf-8')
    audit = audit.replace('from run_majority_constrained_target_admission import', 'from run_soft_topk_target_admission import')
    start, end = audit.index('def independent_objective('), audit.index('\n\ndef independent_polytope')
    audit = audit[:start]+'from independent_soft_topk_objective import independent_objective\n'+audit[end:]
    audit = audit.replace("        if fit['polytope'] is not None:", "        if fit['polytope'] is not None:\n            assert fit['polytope']['protection_reference'] == 'original_full'")
    audit = audit.replace('independent_fixed_witness_geometry_objective_and_fresh_original_classwise_guards_not_optimizer_refit_or_prediction',
                          'independent_fixed_witness_and_scalarBrent_soft5_objective_fresh_guards_not_optimizer_refit_or_prediction')
    save('audit_soft_topk_target_admission.py', audit)


if __name__ == '__main__': main()
