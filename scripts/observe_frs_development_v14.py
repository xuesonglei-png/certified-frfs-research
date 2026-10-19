"""Live density-aware experiments plus retained warm failure and guarded recovery."""
import json
from observe_frs_development_v13 import ROOT,STUDIES,current_authority

NAME='warm-majority-frs-development-v3'
STUDIES[NAME]=(NAME+'-chain-status.json','warm-majority-FRS-development-v3-review.json',NAME+'-source-freeze.json')


if __name__=='__main__':
    out=current_authority()
    (ROOT/'research/latest-live-experiment-observation-v14.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    names=(NAME,'warm-majority-frs-development-v2','density-corrected-vote-development-v1',
        'ensemble-vote-certificate-development-v1','plurality-certificate-development-v1',
        'dynamic-certificate-development-v3','joint-certificate-refresh-development-v1',
        'direct-majority-frs-development','sparse-global-frs-training','pair-transition-frs-inner')
    keys=('phase','child_pid','changed_frozen_python_sources','complete_audit_passed',
        'completed_per_dataset_audits','recovery_v2')
    print(json.dumps({n:{k:v for k,v in out['states'][n].items() if k in keys} for n in names},indent=2),flush=True)
