"""Append new studies to the existing lightweight, small-file observer."""
import json
from observe_frs_development import ROOT,STUDIES,observe

STUDIES.update({
    'clean-global-frs-training':('clean-global-frs-training-chain-status.json','clean-global-FRS-training-review.json','clean-global-frs-training-source-freeze.json'),
    'clean-global-frs-prediction':('clean-global-frs-prediction-chain-status.json','clean-global-FRS-prediction-review.json','clean-global-frs-prediction-source-freeze.json'),
    'fixed-face-frs-development':('fixed-face-frs-development-chain-status.json','fixed-face-FRS-development-review.json','fixed-face-frs-development-source-freeze.json'),
})


if __name__=='__main__':
    out=observe()
    for name,state in out['states'].items():
        folder=ROOT/'results'/name
        if folder.exists():
            state['completed_per_dataset_audits']=[p.name.removesuffix('.audit.json') for p in folder.glob('*.audit.json') if p.name!='complete-audit.json']
            state['complete_rows_before_study_summary']=[p.stem for p in folder.glob('*.json') if p.stem not in ('complete','complete-audit') and '.' not in p.stem]
    (ROOT/'research/latest-live-experiment-observation-v3.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:{f:v for f,v in s.items() if f in ('phase','child_pid','changed_frozen_python_sources',
        'complete_audit_passed','summary_matches_full_audit','review_matches_full_audit','completed_per_dataset_audits',
        'complete_rows_before_study_summary')} for k,s in out['states'].items()},indent=2),flush=True)
