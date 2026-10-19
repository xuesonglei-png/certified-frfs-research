"""Complete audited pruning diagnosis, retaining every blocked deletion."""
import json
from pathlib import Path
from collections import Counter
from run_postmetric_pruning_admission import ROOT,FOLDER,digest
from continue_soft_topk_knn_pilot import verify


def reasons(point):
    return {'global_dependency':point['certificate']['dependency_certificate']['upper_bound']>.1,
            'classwise_dependency':not point['class_certificate']['feasible_all_classes'],
            'majority_retention':not point['certificate']['majority_feasible'],
            'hard_training_BA':not point['nonworsening_training_BA']}


def main():
    audit=verify(FOLDER); summary=json.loads((FOLDER/'complete.json').read_text()); rows=[]; totals=Counter()
    for row in summary['rows']:
        saved=json.loads((FOLDER/(row['dataset']+'.json')).read_text()); fit=saved['fit']; counts=Counter()
        selected=set(fit['chosen_ids']); terminal=[]
        final_features=fit['features']
        for p in fit['checks']:
            flags=reasons(p); counts['candidate_checks']+=1
            for name,failed in flags.items(): counts['rejected_'+name]+=failed
            counts['FRS_accepted_hard_harmful']+=p['accepted'] and not p['nonworsening_training_BA']
            counts['hard_nonworsening_FRS_rejected']+=not p['accepted'] and p['nonworsening_training_BA']
            counts['eligible']+=p['eligible']; counts['chosen']+=p['id'] in selected
            if p['features_before']==final_features:
                terminal.append({'removed_feature':p['removed_feature'],'reasons':flags,
                    'global_upper':p['certificate']['dependency_certificate']['upper_bound'],
                    'classwise_max_upper':p['class_certificate']['max_upper_bound'],
                    'hard_delta_pp':100*(p['hard_training_LOO_BA']-p['BA_before'])})
        totals.update(counts)
        rows.append({**row,'counts':dict(counts),'terminal_pool':terminal})
    assert totals['candidate_checks']==audit['counts']['scalar_candidate_votes']==117
    output={'rows':rows,'counts':dict(totals),'source_summary':summary,
            'source_audit_sha256':digest(FOLDER/'complete-audit.json'),'script_sha256':digest(Path(__file__)),
            'goal_achieved':False,'limitations':['ALL15 training feasibility, no query prediction evaluated.',
                'Half-support target not reached in any dataset; pool4 stops do not prove irreducibility.',
                'FRS admission and hard training nonworsening are separate tests.',
                'Only proposed deployed metrics studied; no matched native or weighted prediction comparison.',
                'No physical measurement costs, online speed or population accuracy guarantee.']}
    (ROOT/'research/postmetric-pruning-admission-review.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k not in ('rows','source_summary')},indent=2),flush=True)


if __name__=='__main__': main()
