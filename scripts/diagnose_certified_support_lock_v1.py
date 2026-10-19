"""Audit-inherited ALL15 feature reachability, not a new method or result claim.

No labels/learning/new candidate selection: compare known support sets. Any
advantage of a native control using omitted features cannot be reproduced by
weight adjustment confined to the original support.
"""
import json
from pathlib import Path
from run_simplex_soft5_knn_pilot import ROOT,digest
from continue_soft_topk_knn_pilot import verify

BASE=ROOT/'results/frs-joint-zero-face-prediction'
CONTROLS=('fgs_native120','fgs_weighted_native120','sfrip_native6','sfrip_weighted_native6')


def main():
    audit=verify(BASE);rows=[]
    for name in sorted(audit['artifact_hashes']):
        p=BASE/(name+'.json');saved=json.loads(p.read_text());metrics=saved['metrics']
        anchor=set(metrics['anchor']['features']);joint=set(metrics['joint_pair_transition']['features'])
        row={'dataset':name,'original_dimension':saved['d'], 'anchor_support':sorted(anchor),
            'joint_support':sorted(joint),'controls':{},'source_sha256':digest(p)}
        for control in CONTROLS:
            ids=set(metrics[control]['features']);a=saved['results'][control]['evaluation']['metrics']['clean_BA']
            base=saved['results']['anchor']['evaluation']['metrics']['clean_BA']
            pair=saved['results']['joint_pair_transition']['evaluation']['metrics']['clean_BA']
            row['controls'][control]={'features':sorted(ids),'outside_anchor':sorted(ids-anchor),
                'outside_joint':sorted(ids-joint),'support_representable_in_anchor':ids<=anchor,
                'observed_first_inner_BA_delta_vs_anchor_pp':100*(a-base),
                'observed_first_inner_BA_delta_vs_joint_pp':100*(a-pair)}
        rows.append(row)
    summary={control:{'unrepresentable_supports':sum(not r['controls'][control]['support_representable_in_anchor'] for r in rows),
        'positive_BA_gaps_with_unrepresentable_support':sum(
            not r['controls'][control]['support_representable_in_anchor'] and r['controls'][control]['observed_first_inner_BA_delta_vs_anchor_pp']>1e-10 for r in rows),
        'positive_BA_gaps_despite_representable_support':sum(
            r['controls'][control]['support_representable_in_anchor'] and r['controls'][control]['observed_first_inner_BA_delta_vs_anchor_pp']>1e-10 for r in rows)}
        for control in CONTROLS}
    out={'datasets':len(rows),'rows':rows,'summary':summary,'source_complete_audit_sha256':digest(BASE/'complete-audit.json'),
        'script_sha256':digest(Path(__file__)),'classification_superiority_established':False,'goal_achieved':False,
        'limits':['Set reachability is an exact structural observation; it does not prove omitted features cause an accuracy gap.',
            'Representable supports can still have incompatible certificate or weight requirements.',
            'All15 first-inner accuracy observations are exposed development context, never independent confirmation.',
            'No candidate is selected or retrained from these observations; all15 and all four controls are included.']}
    (ROOT/'research/certified-support-lock-diagnosis-v1.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
