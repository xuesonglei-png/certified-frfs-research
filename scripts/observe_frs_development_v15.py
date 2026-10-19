"""Live authority including failed robust projection and mixture/recovery audits."""
import json
from pathlib import Path
from observe_frs_development_v14 import current_authority,ROOT


def main():
    out=current_authority()
    additions={}
    for name in ('robustaware-projection-v1','mixture-enemy-diagnostic-v1','mixture-dependency-development-v1'):
        folder=ROOT/'results'/name;summary=json.loads((folder/'complete.json').read_text())
        audit=json.loads((folder/'complete-audit.json').read_text())
        from run_mixture_enemy_diagnostic_v1 import digest
        assert audit['passed'] and audit['summary_sha256']==digest(folder/'complete.json')
        additions[name]={k:v for k,v in summary.items() if k not in ('rows',)}
        additions[name]['audit_sha256']=digest(folder/'complete-audit.json')
        additions[name]['independent_audit_passed']=True
    recovery=ROOT/'results/sparse-global-frs-training-recovery-v2-chain-status.json'
    out['new_audited_studies']=additions
    out['sparse_global_recovery_v2']=json.loads(recovery.read_text()) if recovery.exists() else None
    out['latest_complete_review']='research/projection-and-mixture-v1-review.json'
    (ROOT/'research/latest-live-experiment-observation-v15.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'new_audited_studies':additions,'sparse_global_recovery_v2':out['sparse_global_recovery_v2']},indent=2))


if __name__=='__main__':main()
