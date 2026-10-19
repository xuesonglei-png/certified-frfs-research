"""Lightweight authority: retained failures, recovery and certificate union."""
import json
from observe_frs_development_v7 import ROOT,STUDIES,current

for version in ('v1','v2','v3'):
    name='dynamic-certificate-development-'+version
    STUDIES[name]=(name+'-chain-status.json',name+'-review.json',name+'-source-freeze.json')

STUDIES['joint-certificate-refresh-development-v1']=('joint-certificate-refresh-development-v1-chain-status.json','joint-certificate-refresh-development-v1-review.json','joint-certificate-refresh-development-v1-source-freeze.json')

if __name__=='__main__':
    out=current()
    (ROOT/'research/latest-live-experiment-observation-v10.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    names=('joint-certificate-refresh-development-v1','dynamic-certificate-development-v1','dynamic-certificate-development-v2','dynamic-certificate-development-v3','certified-path-guard-development',
        'direct-majority-frs-development','warm-majority-frs-development-v2','fixed-face-frs-development-v2',
        'pair-transition-frs-inner','sparse-global-frs-training','clean-global-frs-training','clean-global-frs-prediction')
    keys=('phase','child_pid','changed_frozen_python_sources','complete_audit_passed','summary_matches_full_audit',
        'review_matches_full_audit','completed_per_dataset_audits','completed_per_dataset_v2_audits','recovery_v2')
    print(json.dumps({n:{k:v for k,v in out['states'][n].items() if k in keys} for n in names},indent=2),flush=True)
