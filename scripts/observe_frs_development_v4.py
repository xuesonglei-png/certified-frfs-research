"""Observe retained failed v1 and the matched, freshly fitted v2 experiment."""
import json
from observe_frs_development_v3 import ROOT,STUDIES,observe

STUDIES['fixed-face-frs-development-v2']=(
    'fixed-face-frs-development-v2-chain-status.json',
    'fixed-face-FRS-development-v2-review.json',
    'fixed-face-frs-development-v2-source-freeze.json')


if __name__=='__main__':
    out=observe()
    for name,state in out['states'].items():
        folder=ROOT/'results'/name
        if folder.exists():
            state['completed_per_dataset_audits']=[p.name.removesuffix('.audit.json') for p in folder.glob('*.audit.json')]
            state['complete_rows_before_study_summary']=[p.stem for p in folder.glob('*.json') if p.stem not in ('complete','complete-audit') and '.' not in p.stem]
    (ROOT/'research/latest-live-experiment-observation-v4.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:{f:v for f,v in s.items() if f in ('phase','child_pid','changed_frozen_python_sources',
        'complete_audit_passed','summary_matches_full_audit','review_matches_full_audit','completed_per_dataset_audits',
        'complete_rows_before_study_summary')} for k,s in out['states'].items() if k in
        ('fixed-face-frs-development','fixed-face-frs-development-v2','clean-global-frs-training','clean-global-frs-prediction',
         'sparse-global-frs-training','pair-transition-frs-inner')},indent=2),flush=True)
