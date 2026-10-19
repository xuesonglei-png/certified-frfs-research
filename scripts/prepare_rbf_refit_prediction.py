"""Create separately fingerprinted common-predictor runner/auditor artifacts.

Copies preserve frozen experiment21 sources. No scientific result files change.
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old,new)


def main():
    runner = (ROOT/'scripts/run_crossfit_kernel_prediction.py').read_text(encoding='utf-8')
    runner = replace_once(runner,"INPUT = ROOT/'results/crossfit-kernel-screen'; FOLDER = ROOT/'results/crossfit-kernel-prediction'",
        "INPUT = ROOT/'results/crossfit-rbf-refit-screen'; FOLDER = ROOT/'results/crossfit-rbf-refit-prediction'\nIMPUTATION = ROOT/'results/crossfit-kernel-screen'")
    runner = replace_once(runner,"METHODS = ('new','old','full','fgs_matched','sfrip_matched','fgs_native','sfrip_native')",
        "METHODS = ('new','old','full','fgs_matched','sfrip_matched','fgs_native','sfrip_native','imputed')")
    runner = runner.replace('research/crossfit-kernel-prediction-protocol.md','research/crossfit-rbf-refit-prediction-protocol.md')
    runner = replace_once(runner,"fullaudit = json.loads((INPUT/'complete-audit.json').read_text()); assert fullaudit['passed'] and fullaudit['resource_gate']",
        "fullaudit = json.loads((INPUT/'complete-audit.json').read_text()); assert fullaudit['passed'] and fullaudit['resource_gate']\n    imputed = json.loads((IMPUTATION/(name+'.json')).read_text())\n    imputed_audit = json.loads((IMPUTATION/(name+'-audit.json')).read_text())\n    assert imputed_audit['fingerprints']['artifact_sha256'] == digest(IMPUTATION/(name+'.json'))")
    runner = replace_once(runner,"INPUT/(name+'.json'),INPUT/(name+'-audit.json')):",
        "INPUT/(name+'.json'),INPUT/(name+'-audit.json'),IMPUTATION/(name+'.json'),IMPUTATION/(name+'-audit.json')):")
    runner = replace_once(runner,"supports = {'new':path['features'],'old':path['initial_features'],'full':list(range(d))}; selectors = {}",
        "ir = imputed['rows'][index]; assert ir['outer'] == i and ir['inner'] == j and ir['reference_indices'] == a.tolist()\n        assert ir['path']['initial_features'] == path['initial_features'] and ir['path']['candidate']['certificate']['feasible']\n        supports = {'new':path['features'],'old':path['initial_features'],'full':list(range(d)),'imputed':ir['path']['features']}; selectors = {}")
    runner = replace_once(runner,"exact = all(len(supports[m]) == target for m in ('old','fgs_matched','sfrip_matched'))",
        "exact = all(len(supports[m]) == target for m in ('old','fgs_matched','sfrip_matched','imputed'))")
    audit = (ROOT/'scripts/audit_crossfit_kernel_prediction.py').read_text(encoding='utf-8')
    audit = audit.replace('from run_crossfit_kernel_prediction import INPUT,FOLDER,METHODS,CLASSIFIERS',
        'from run_crossfit_rbf_refit_prediction import INPUT,FOLDER,METHODS,CLASSIFIERS,IMPUTATION')
    audit = replace_once(audit,"upstream = json.loads((INPUT/(name+'-audit.json')).read_text()); assert upstream['fingerprints']['artifact_sha256'] == digest(INPUT/(name+'.json'))",
        "upstream = json.loads((INPUT/(name+'-audit.json')).read_text()); assert upstream['fingerprints']['artifact_sha256'] == digest(INPUT/(name+'.json'))\n    imputed = json.loads((IMPUTATION/(name+'.json')).read_text())\n    ia = json.loads((IMPUTATION/(name+'-audit.json')).read_text()); assert ia['fingerprints']['artifact_sha256'] == digest(IMPUTATION/(name+'.json'))")
    audit = replace_once(audit,"assert r['candidate'] == sr['path']['candidate']; fs = supports['new']; target = len(fs); assert r['target_count'] == target == len(supports['old'])",
        "ir = next(row for row in imputed['rows'] if row['outer'] == i and row['inner'] == j)\n        assert ir['reference_indices'] == r['reference_indices'] and supports['imputed'] == ir['path']['features']\n        assert ir['path']['initial_features'] == supports['old'] and ir['path']['candidate']['certificate']['feasible']\n        assert r['candidate'] == sr['path']['candidate']; fs = supports['new']; target = len(fs); assert r['target_count'] == target == len(supports['old'])")
    audit = audit.replace("('old','fgs_matched','sfrip_matched')","('old','fgs_matched','sfrip_matched','imputed')")
    for path,text in [('scripts/run_crossfit_rbf_refit_prediction.py',runner),('scripts/audit_crossfit_rbf_refit_prediction.py',audit)]:
        (ROOT/path).write_text(text,encoding='utf-8')


if __name__ == '__main__': main()
