"""Read only complete verified broad/nine/source cohorts."""
import json
from pathlib import Path
from run_expanded_fgs_comparison import ROOT,FOLDER,SOURCE,digest
from continue_soft_topk_knn_pilot import verify


def main():
    verify(FOLDER);verify(SOURCE);r=json.loads((FOLDER/'complete.json').read_text())
    assert r['datasets']==15 and r['inner_folds']==44 and r['configs_per_fit']==120
    output={k:v for k,v in r.items() if k!='rows'}
    output.update({'audit_sha256':digest(FOLDER/'complete-audit.json'),'source_audit_sha256':digest(SOURCE/'complete-audit.json'),
        'review_script_sha256':digest(Path(__file__)),'goal_achieved':False})
    (ROOT/'research/expanded-fgs-comparison-review.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
