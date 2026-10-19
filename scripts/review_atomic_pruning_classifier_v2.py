"""Complete audited cohort, including exact controls and recent DSN."""
import json
from pathlib import Path
import numpy as np
from run_atomic_pruning_classifier_v2 import ROOT,FOLDER,SOURCE,METHODS,digest
from continue_soft_topk_knn_pilot import verify
from review_box_iterations import paired


def main():
    audit=verify(FOLDER); original_audit=verify(SOURCE)
    summary=json.loads((FOLDER/'complete.json').read_text()); original=json.loads((SOURCE/'complete.json').read_text())
    assert len(summary['rows'])==15
    old={r['dataset']:r for r in original['rows']}; details=[]
    for row in summary['rows']:
        saved=json.loads((FOLDER/(row['dataset']+'.json')).read_text()); count=row['actual_counts']['proposed']
        assert all(row['actual_counts'][m]==count for m in ('fgs_exact','sfrip_exact','fgs_uniform_exact','sfrip_uniform_exact'))
        assert row['actual_counts']['full_source']==row['d']
        metric=saved['outer']['results']['proposed']['metric']
        details.append({'dataset':row['dataset'],'d':row['d'],'new_count':count,
            'original_count':old[row['dataset']]['actual_counts']['proposed'],
            'selected_variant':metric['variant'],'new_BA':row['metrics']['proposed']['clean_BA'],
            'original_BA':old[row['dataset']]['metrics']['proposed']['clean_BA'],
            'new_minus_original_pp':100*(row['metrics']['proposed']['clean_BA']-old[row['dataset']]['metrics']['proposed']['clean_BA']),
            'all_counts':row['actual_counts'],'positive_weight_counts':row['positive_weight_counts']})
    rescale_folder=ROOT/'results/postmetric-pruning-classifier'
    verify(rescale_folder)
    rescale={r['dataset']:r for r in json.loads((rescale_folder/'complete.json').read_text())['rows']}
    assert set(rescale)==set(old)
    for row in details:
        other=json.loads((rescale_folder/(row['dataset']+'.json')).read_text())
        ours=json.loads((FOLDER/(row['dataset']+'.json')).read_text())
        assert ours['reference_indices']==other['reference_indices'] and ours['validation_indices']==other['validation_indices']
        row['rescale_count']=rescale[row['dataset']]['actual_counts']['proposed']
        row['rescale_BA']=rescale[row['dataset']]['metrics']['proposed']['clean_BA']
        row['new_minus_rescale_pp']=100*(row['new_BA']-row['rescale_BA'])
    dsfolder=ROOT/'results/dsnfrs-formula-expanded-pilot'; ds_audit=verify(dsfolder)
    ds={r['dataset']:r for r in json.loads((dsfolder/'complete.json').read_text())['rows']}
    assert set(ds)==set(old)
    for name in ds:
        ours=json.loads((FOLDER/(name+'.json')).read_text()); theirs=json.loads((dsfolder/(name+'.json')).read_text())
        assert ours['reference_indices']==theirs['reference_indices'] and ours['validation_indices']==theirs['validation_indices']
    ds_contrasts={m:paired([r['metrics'][m]['clean_BA']-ds[r['dataset']]['metrics']['clean_BA'] for r in summary['rows']]) for m in METHODS}
    fgsfolder=ROOT/'results/expanded-fgs-comparison'
    fgs_audit=verify(fgsfolder)
    fgs_summary=json.loads((fgsfolder/'complete.json').read_text())
    fgs={r['dataset']:r for r in fgs_summary['rows']}
    assert set(fgs)==set(old)
    for name in fgs:
        ours=json.loads((FOLDER/(name+'.json')).read_text())
        theirs=json.loads((fgsfolder/(name+'.json')).read_text())
        assert ours['reference_indices']==theirs['reference_indices'] and ours['validation_indices']==theirs['validation_indices']
    fgs_methods=list(next(iter(fgs.values()))['metrics'])
    fgs_contrasts={m:paired([r['metrics']['proposed']['clean_BA']-fgs[r['dataset']]['metrics'][m]['clean_BA'] for r in summary['rows']]) for m in fgs_methods}
    output={'FGS120_contrasts':fgs_contrasts,'FGS120_audit_sha256':digest(fgsfolder/'complete-audit.json'),'FGS120_count_scope':'prior rescale counts, not atomic count matched','rows':details,'new_minus_original':paired([r['new_minus_original_pp']/100 for r in details]),
        'new_minus_rescale':paired([r['new_minus_rescale_pp']/100 for r in details]),
        'rescale_audit_sha256':digest(rescale_folder/'complete-audit.json'),
        'all_classifier_contrasts':summary['contrasts_clean_BA'],'primary_descriptive_holm':summary['primary_descriptive_holm'],
        'DSN_expanded_formula_contrasts':ds_contrasts,'mean_removed':summary['mean_removed'],
        'selection_count_vs_positive_weight_count':'Selected-cardinality controls can have smaller active supports; physical costs not measured.',
        'counts':summary['counts'],'pilot_gate_passed':summary['pilot_gate_passed'],'goal_achieved':False,
        'audit_sha256':digest(FOLDER/'complete-audit.json'),'source_audit_sha256':digest(SOURCE/'complete-audit.json'),
        'DSN_audit_sha256':digest(dsfolder/'complete-audit.json'),'review_script_sha256':digest(Path(__file__)),
        'limitations':['ALL15 and reused folds exposed development, p-values descriptive not confirmatory.',
            'Refined FGS/SFRIP counts can differ; exact-count controls are separate adaptations with extra tuning.',
            'Inherited seven source failures remain failures; new exact-control failures separately retained.',
            'Recent DSN is formula-controlled, compatible own-class example, author classifier conventions still unresolved.',
            'No original-full guarantee for own-anchor controls, and no population risk theorem.',
            'Additional author-faithful recent baselines and fresh confirmation remain required.']}
    (ROOT/'research/atomic-pruning-classifier-v2-review.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k not in ('rows','all_classifier_contrasts','DSN_expanded_formula_contrasts')},indent=2),flush=True)


if __name__=='__main__': main()
