"""Preserve actual-code audit; separately apply the prose's rescale-first tie.

The registered code used lexical kind order (atomic before rescale), while
the protocol stated rescale first. This recheck does not overwrite the run.
It reuses all independently verified endpoint scores, without any query data.
"""
import json
from pathlib import Path
from run_reference_thinning_diagnostic import ROOT,FOLDER,digest
from continue_soft_topk_knn_pilot import verify


def main():
    audit=verify(FOLDER); records=[]
    for name in sorted(audit['artifact_hashes']):
        saved=json.loads((FOLDER/(name+'.json')).read_text())
        for r in saved['records']:
            def key(i,stress):
                p=r['points'][i]; hard=-round(p['hard_BA'],12)
                prefix=(-round(r['scores'][i]['mean_BA'],12),hard) if stress else (hard,)
                return prefix+(0 if p['kind']=='rescale' else 1,p['receiver'] if p['receiver'] is not None else -1)
            eligible=[i for i,p in enumerate(r['points']) if p['eligible']]
            stable=[i for i in eligible if r['scores'][i]['mean_BA']>=r['parent_score']['mean_BA']-1e-12]
            hard=min(eligible,key=lambda i:key(i,False)) if eligible else None
            stress=min(stable,key=lambda i:key(i,True)) if stable else None
            # Independently ensure every eligible competitor is lexically no better.
            if hard is not None: assert all(key(hard,False)<=key(i,False) for i in eligible)
            if stress is not None: assert all(key(stress,True)<=key(i,True) for i in stable)
            records.append({'dataset':name,'source_candidate_id':r['source_candidate_id'],
                'hard_before':r['hard_selected_id'],'hard_rescale_first':hard,
                'stress_before':r['stress_selected_id'],'stress_rescale_first':stress,
                'hard_score_worse':bool(hard is not None and r['scores'][hard]['mean_BA']<r['parent_score']['mean_BA']-1e-12)})
    counts={'candidates':len(records),'original_joint_eligible':sum(r['hard_rescale_first'] is not None for r in records),
        'stress_nonworse_eligible':sum(r['stress_rescale_first'] is not None for r in records),
        'hard_selection_stress_worse':sum(r['hard_score_worse'] for r in records),
        'ranking_changes':sum(r['stress_rescale_first'] is not None and r['stress_rescale_first']!=r['hard_rescale_first'] for r in records),
        'hard_tie_choices_changed':sum(r['hard_before']!=r['hard_rescale_first'] for r in records),
        'stress_tie_choices_changed':sum(r['stress_before']!=r['stress_rescale_first'] for r in records)}
    result={'counts':counts,'records':records,'scope':'prose_tie_order_recheck_of_audited_training_scores_not_new_prediction',
        'original_code_prose_mismatch':'Code lexical atomic-first, protocol prose rescale-first; original evidence preserved.',
        'source_audit_sha256':digest(FOLDER/'complete-audit.json'),'script_sha256':digest(Path(__file__)),
        'goal_achieved':False}
    path=ROOT/'research/reference-thinning-tie-order-recheck.json';assert not path.exists()
    path.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(counts,indent=2))


if __name__=='__main__':main()
