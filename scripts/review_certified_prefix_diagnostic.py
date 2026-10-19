"""Complete audited opportunities, without promoting an inner search maximum."""
import json
from pathlib import Path
from run_certified_prefix_diagnostic import ROOT,FOLDER,digest
from continue_soft_topk_knn_pilot import verify


def main():
    verify(FOLDER);r=json.loads((FOLDER/'complete.json').read_text())
    assert r['datasets']==15 and r['inner_folds']==44
    details={m:{k:v for k,v in report.items() if k!='rows'} for m,report in r['methods'].items()}
    output={'methods':details,'datasets':15,'inner_folds':44,'independent_evaluations':r['unique_metrics'],
        'candidate_evidence_only':True,'goal_achieved':False,
        'audit_sha256':digest(FOLDER/'complete-audit.json'),'review_script_sha256':digest(Path(__file__)),
        'limits':['Five recipes contain binary endpoints; selected inner BA cannot worsen by construction.',
            'Strictly better intermediate models identify missed choices, not unbiased performance.',
            'No outer prediction, modern-baseline superiority or significance established.',
            'All source recipes and folds are exposed dependent development.',
            'Future full classifier must give all weighted arms equal options and rebuild selected-count controls.']}
    path=ROOT/'research/certified-prefix-diagnostic-review.json'
    path.write_text(json.dumps(output,indent=2),encoding='utf-8');print(json.dumps(output,indent=2))


if __name__=='__main__':main()
