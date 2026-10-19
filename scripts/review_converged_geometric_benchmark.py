"""Review audited common-budget results and separate convention from learning."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_converged_geometric_benchmark import ROOT,FOLDER,SOURCE,REFINED,PRIMARY,METHODS,digest
from independent_geometric_evaluation import independent_evaluate
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from review_box_iterations import paired


def main():
    threadpool_limits(1);summary=json.loads((FOLDER/'complete.json').read_text())
    audit=json.loads((FOLDER/'complete-audit.json').read_text());assert audit['passed']
    assert audit['summary_sha256']==digest(FOLDER/'complete.json')
    phases=[];selected_recipes=[]
    for row in summary['rows']:
        name=row['dataset'];assert audit['artifact_hashes'][name]==digest(FOLDER/(name+'.json'))
        saved=json.loads((FOLDER/(name+'.json')).read_text());source=json.loads((SOURCE/(name+'.json')).read_text())
        X,y=load_dataset(name);ri,vi=np.array(saved['reference_indices']),np.array(saved['validation_indices'])
        A,B=preprocess_fit(X[ri],X[vi]);old=source['outer']['results'];fresh=saved['outer']['results']
        split={};recipes={}
        for m in REFINED:
            anchor=old[m]['metric'];geometric=independent_evaluate(A,y[ri],B,y[vi],anchor)['metrics']['clean_BA']
            before=old[m]['evaluation']['metrics']['clean_BA'];after=fresh[m]['evaluation']['metrics']['clean_BA']
            split[m]={'old_lexical_BA':before,'frozen_anchor_geometric_BA':geometric,'refined_geometric_BA':after,
                      'pure_geometric_gain_pp':100*(geometric-before),'crossgroup_refinement_gain_pp':100*(after-geometric)}
        for m in METHODS:
            metric=fresh[m]['metric'];fit=None if metric['learner_id'] is None else saved['outer']['stage']['learners'][metric['learner_id']]['fit']
            recipes[m]={'config_index':fresh[m]['config_index'],'target_index':fresh[m]['target_index'],
                'target_family':metric.get('target_family'),'interpolation':metric['interpolation'],
                'guard_fallback':metric['guard_fallback'],'selected_target_optimizer_success':None if fit is None else fit['optimizer_success']}
        phases.append({'dataset':name,'phases':split});selected_recipes.append({'dataset':name,'recipes':recipes})
    gains={m:{'geometry':paired([r['phases'][m]['pure_geometric_gain_pp']/100 for r in phases]),
              'crossgroup':paired([r['phases'][m]['crossgroup_refinement_gain_pp']/100 for r in phases])} for m in REFINED}
    robustness={m:paired([r['metrics']['proposed']['query_certified_BA']-r['metrics'][m]['query_certified_BA'] for r in summary['rows']]) for m in PRIMARY}
    perturbation={m:paired([r['metrics']['proposed']['mean_perturbed_BA']-r['metrics'][m]['mean_perturbed_BA'] for r in summary['rows']]) for m in PRIMARY}
    reviewed={'audit_sha256':digest(FOLDER/'complete-audit.json'),'summary_sha256':digest(FOLDER/'complete.json'),
        'pilot_gate_passed':summary['pilot_gate_passed'],'goal_achieved':False,'phase_decomposition':gains,
        'certified_BA_contrasts':robustness,'perturbed_BA_contrasts':perturbation,'phase_rows':phases,'selected_recipes':selected_recipes,
        'classification_superiority_established':False,
        'decision':'Promising development gate only; freeze complete method and run independent confirmation.' if summary['pilot_gate_passed'] else
                   'Predictive development gate failed. Preserve negative result and locate constraints/selection/optimization bottleneck before a new method.',
        'limitations':['All15 outer labels repeatedly exposed: descriptive uncertainty cannot confirm superiority.',
                       'Classifier convention effects are common to controls and not novel feature-selection theory.',
                       'Frozen upstream selections and inherited failures remain; this is not an end-to-end new nested benchmark.']}
    (ROOT/'research/converged-geometric-benchmark-review.json').write_text(json.dumps(reviewed,indent=2),encoding='utf-8')
    lines=['# Audited converged geometric development benchmark','',reviewed['decision'],'',
           f"Proposed mean clean BA: {100*summary['means']['proposed']['clean_BA']:.6f}%; mean removed: {100*summary['mean_removed']:.6f}%.",
           f"New optimizer failures: {summary['counts']['new_target_optimizer_failures']}; upstream failures preserved: {summary['counts']['upstream_optimizer_failures_preserved']}.",
           '', '| Control | Proposed minus control (pp) | Wins/losses/ties | Bootstrap95 (pp) | Holm p |',
           '|---|---:|---:|---|---:|']
    for m in PRIMARY:
        c=summary['contrasts_clean_BA'][m]
        lines.append(f"| {m} | {c['mean_pp']:.6f} | {c['wins']}/{c['losses']}/{c['ties']} | {c['bootstrap95_pp']} | {summary['primary_descriptive_holm'][m]:.6f} |")
    lines+=['','Pure geometry and subsequent crossgroup effects are independently replayed in the JSON review.',
            'All contrasts are exposed-development evidence. No confirmation, universal predictive advantage, or submission-readiness claim is established.']
    (ROOT/'research/converged-geometric-benchmark-review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=('proposed','fgs_weighted_native120','fgs_weighted_count120','sfrip_weighted_native6','sfrip_weighted_count6','full_uniform','full_refined')
    labels=('Proposed','FGS native+metric','FGS matched+metric','SFRIP native+metric','SFRIP matched+metric','Full uniform','Full learned')
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),constrained_layout=True)
    x=np.arange(len(chosen));width=.25
    for j,(key,label) in enumerate([('clean_BA','Clean BA'),('mean_perturbed_BA','Perturbed BA'),('query_certified_BA','Certified correct BA')]):
        axes[0].bar(x+(j-1)*width,[100*summary['means'][m][key] for m in chosen],width,label=label)
    axes[0].set_xticks(x,labels,rotation=30,ha='right');axes[0].set_ylabel('Dataset mean (%)');axes[0].legend(fontsize=8)
    ordered=summary['rows'];delta=[100*(r['metrics']['proposed']['clean_BA']-r['metrics']['fgs_weighted_native120']['clean_BA']) for r in ordered]
    axes[1].barh(np.arange(len(ordered)),delta,color=['#1976d2' if v>=0 else '#c44e52' for v in delta])
    axes[1].set_yticks(np.arange(len(ordered)),[r['dataset'] for r in ordered]);axes[1].axvline(0,color='black',lw=.7)
    axes[1].set_xlabel('Proposed minus FGS native+metric (pp)')
    fig.suptitle('Exposed development results; no confirmatory superiority claim')
    fig.savefig(FOLDER/'comparison.png',dpi=180);plt.close(fig)
    print(json.dumps({k:v for k,v in reviewed.items() if k not in ('phase_rows','selected_recipes')},indent=2),flush=True)


if __name__=='__main__':main()
