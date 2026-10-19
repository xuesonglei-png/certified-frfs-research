"""Read-only live authority including calibrated plurality development."""
import json
import ijson
from observe_frs_development_v10 import ROOT,STUDIES,current

NAME='plurality-certificate-development-v1'
STUDIES[NAME]=(NAME+'-chain-status.json',NAME+'-review.json',NAME+'-source-freeze.json')


def latest():
    out=current();folder=ROOT/'results'/NAME;rows=[]
    for path in sorted(folder.glob('*.json')):
        if '.' in path.stem or path.stem in ('complete','complete-audit'):continue
        with path.open('rb') as stream:
            results=next(ijson.items(stream,'results',use_float=True))
        row={'dataset':path.stem,'independently_audited':(folder/(path.stem+'.audit.json')).exists(),
            'clean_BA':{m:r['evaluation']['metrics']['clean_BA'] for m,r in results.items()}}
        row['delta_vs_weighted_FGS_pp']={m:100*(v-row['clean_BA']['fgs_weighted_native120'])
            for m,v in row['clean_BA'].items() if m.endswith(('_plurality','_ownmass'))}
        rows.append(row)
    out['plurality_completed_rows']=rows
    return out


if __name__=='__main__':
    out=latest()
    (ROOT/'research/latest-live-experiment-observation-v11.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    names=(NAME,'joint-certificate-refresh-development-v1','dynamic-certificate-development-v3',
        'direct-majority-frs-development','warm-majority-frs-development-v2','sparse-global-frs-training')
    keys=('phase','child_pid','changed_frozen_python_sources','complete_audit_passed',
        'completed_per_dataset_audits','recovery_v2')
    print(json.dumps({'states':{n:{k:v for k,v in out['states'][n].items() if k in keys} for n in names},
        'plurality_completed_rows':out['plurality_completed_rows']},indent=2),flush=True)
