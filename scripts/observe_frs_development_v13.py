"""Actual singleton, ensemble and inverse-coverage development authority."""
import json
import ijson
from observe_frs_development_v12 import ROOT,STUDIES,observe_latest

NAME='density-corrected-vote-development-v1'
STUDIES[NAME]=(NAME+'-chain-status.json',NAME+'-review.json',NAME+'-source-freeze.json')


def current_authority():
    out=observe_latest();folder=ROOT/'results'/NAME;rows=[]
    for path in sorted(folder.glob('*.json')):
        if '.' in path.stem or path.stem in ('complete','complete-audit'):continue
        with path.open('rb') as stream:results=next(ijson.items(stream,'results',use_float=True))
        scores={m:r['evaluation']['metrics']['clean_BA'] for m,r in results.items()}
        methods=[m for m in scores if m.startswith('density_joint_')]
        rows.append({'dataset':path.stem,'independently_audited':(folder/(path.stem+'.audit.json')).exists(),
            'clean_BA':scores,
            'delta_vs_common_coverage_uniform_pp':{m:100*(scores[m]-scores['uniform_'+m.removeprefix('density_')]) for m in methods},
            'delta_vs_weighted_FGS_pp':{m:100*(scores[m]-scores['fgs_weighted_native120']) for m in methods}})
    out['density_corrected_completed_rows']=rows;return out


if __name__=='__main__':
    out=current_authority()
    (ROOT/'research/latest-live-experiment-observation-v13.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    names=(NAME,'ensemble-vote-certificate-development-v1','plurality-certificate-development-v1',
        'joint-certificate-refresh-development-v1','dynamic-certificate-development-v3',
        'direct-majority-frs-development','warm-majority-frs-development-v2',
        'sparse-global-frs-training','clean-global-frs-training','pair-transition-frs-inner')
    keys=('phase','child_pid','changed_frozen_python_sources','complete_audit_passed',
        'completed_per_dataset_audits','recovery_v2')
    print(json.dumps({'states':{n:{k:v for k,v in out['states'][n].items() if k in keys} for n in names},
        'density_corrected_completed_rows':out['density_corrected_completed_rows']},indent=2),flush=True)
