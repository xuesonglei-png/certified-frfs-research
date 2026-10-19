"""Check run completeness and independently rerun one retained outer split."""
import json
from pathlib import Path
import sys
import math
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from run_experiments import run_fold,code_fingerprint,atomic_json

def main():
    config=json.loads((ROOT/"results/primary/config.json").read_text())
    runs=[json.loads(p.read_text()) for p in sorted((ROOT/"results/primary").glob("*-s*-f*.json"))]
    assert len(runs)==132
    assert len({r['dataset'] for r in runs})==15
    assert sum(len(r['rows']) for r in runs)==8316
    assert json.loads((ROOT/"results/primary/errors.json").read_text())==[]
    expected_hash=code_fingerprint()
    assert config['code_fingerprint']==expected_hash
    for run in runs:
        assert run['code_fingerprint']==expected_hash
        assert run['eps']==.01 and run['eta']==.1
        assert set(run['train_indices']).isdisjoint(run['test_indices'])
        assert len(run['subsets'])==7 and len(run['rows'])==63
        cert=run['subsets']['certified']
        assert cert['certificate']['upper_bound']<=.1+1e-9
        assert cert['attack_lower_bound']<=cert['certificate']['upper_bound']+1e-8
        k=len(cert['features'])
        for name in ['mi_matched','mrmr_matched','relieff_matched','maf_framework']:
            assert len(run['subsets'][name]['features'])==k
    output=ROOT/"results/reproduction-check"
    task=('wine',17,0,.01,.1,str(output),'independent_reproduction',expected_hash)
    result=run_fold(task)
    original=json.loads((ROOT/"results/primary/wine-s17-f0.json").read_text())
    reproduced=json.loads((output/"wine-s17-f0.json").read_text())
    for name in original['subsets']:
        a,b=original['subsets'][name],reproduced['subsets'][name]
        assert a['features']==b['features']
        assert math.isclose(a['certificate']['upper_bound'],b['certificate']['upper_bound'],abs_tol=1e-12)
    for a,b in zip(original['rows'],reproduced['rows']):
        for key in ['balanced_accuracy','macro_f1']:
            assert math.isclose(a[key],b[key],abs_tol=1e-12)
    report={'all_checks_passed':True,'completed_splits':132,'datasets':15,'evaluation_rows':8316,
            'reproduced_dataset':'wine','reproduced_seed':17,'reproduced_fold':0,
            'matched_feature_subsets':7,'matched_evaluation_rows':63,
            'code_fingerprint':expected_hash}
    atomic_json(ROOT/"results/reproduction-audit.json",report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
