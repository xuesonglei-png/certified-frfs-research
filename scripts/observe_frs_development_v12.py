"""Read-only authority for actual singleton and three-partition experiments."""
import json
import ijson
from observe_frs_development_v11 import ROOT,STUDIES,latest

NAME='ensemble-vote-certificate-development-v1'
STUDIES[NAME]=(NAME+'-chain-status.json',NAME+'-review.json',NAME+'-source-freeze.json')


def observe_latest():
    out=latest();folder=ROOT/'results'/NAME;rows=[]
    for path in sorted(folder.glob('*.json')):
        if '.' in path.stem or path.stem in ('complete','complete-audit'):continue
        with path.open('rb') as stream:results=next(ijson.items(stream,'results',use_float=True))
        scores={m:r['evaluation']['metrics']['clean_BA'] for m,r in results.items()}
        ensemble=[m for m in scores if m.startswith('ensemble_')]
        rows.append({'dataset':path.stem,'independently_audited':(folder/(path.stem+'.audit.json')).exists(),
            'clean_BA':scores,'delta_vs_singleton_pp':{m:100*(scores[m]-scores[m.removeprefix('ensemble_')]) for m in ensemble},
            'delta_vs_weighted_FGS_pp':{m:100*(scores[m]-scores['fgs_weighted_native120']) for m in ensemble}})
    out['ensemble_completed_rows']=rows;return out


if __name__=='__main__':
    out=observe_latest()
    (ROOT/'research/latest-live-experiment-observation-v12.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    names=(NAME,'plurality-certificate-development-v1','joint-certificate-refresh-development-v1',
        'dynamic-certificate-development-v3','direct-majority-frs-development',
        'warm-majority-frs-development-v2','sparse-global-frs-training','pair-transition-frs-inner')
    keys=('phase','child_pid','changed_frozen_python_sources','complete_audit_passed',
        'completed_per_dataset_audits','recovery_v2')
    print(json.dumps({'states':{n:{k:v for k,v in out['states'][n].items() if k in keys} for n in names},
        'ensemble_completed_rows':out['ensemble_completed_rows']},indent=2),flush=True)
