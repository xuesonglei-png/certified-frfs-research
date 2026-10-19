"""Track retained v1 failures alongside verified recovery and guard experiments."""
import json
from observe_frs_development_v6 import ROOT,STUDIES,observe
from run_simplex_soft5_knn_pilot import digest

STUDIES.update({
    'certified-path-guard-development':('certified-path-guard-development-chain-status.json','certified-path-guard-development-review.json','certified-path-guard-development-source-freeze.json'),
    'warm-majority-frs-development-v2':('warm-majority-frs-development-v2-chain-status.json','warm-majority-FRS-development-v2-review.json','warm-majority-frs-development-v2-source-freeze.json'),
})


def current():
    out=observe()
    for name,state in out['states'].items():
        folder=ROOT/'results'/name
        if folder.exists():
            state['completed_per_dataset_audits']=[p.name.removesuffix('.audit.json') for p in folder.glob('*.audit.json')]
            state['completed_per_dataset_v2_audits']=[p.name.removesuffix('.audit-v2.json') for p in folder.glob('*.audit-v2.json')]
            state['complete_rows_before_study_summary']=[p.stem for p in folder.glob('*.json') if p.stem not in ('complete','complete-audit') and '.' not in p.stem]
        summary=folder/'complete.json'
        if summary.exists():
            s=json.loads(summary.read_text());state['raw_contrasts']=s.get('contrasts')
    status=ROOT/'results/direct-majority-rank-recovery-v2-chain-status.json'
    if status.exists():
        recovery=json.loads(status.read_text());freeze=ROOT/'research/direct-majority-rank-recovery-v2-source-freeze.json'
        if freeze.exists():
            hashes=json.loads(freeze.read_text())['fingerprints']
            recovery['changed_recovery_python_sources']=[rel for rel,sha in hashes.items() if rel.endswith('.py') and digest(ROOT/rel)!=sha]
        out['states']['direct-majority-frs-development']['recovery_v2']=recovery
    return out


if __name__=='__main__':
    out=current();(ROOT/'research/latest-live-experiment-observation-v7.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    names=('certified-path-guard-development','direct-majority-frs-development','warm-majority-frs-development-v2',
        'fixed-face-frs-development-v2','pair-transition-frs-inner','sparse-global-frs-training','clean-global-frs-training')
    keys=('phase','child_pid','changed_frozen_python_sources','complete_audit_passed','summary_matches_full_audit','review_matches_full_audit',
        'completed_per_dataset_audits','completed_per_dataset_v2_audits','complete_rows_before_study_summary','recovery_v2','raw_contrasts')
    print(json.dumps({n:{k:v for k,v in out['states'][n].items() if k in keys} for n in names},indent=2),flush=True)
