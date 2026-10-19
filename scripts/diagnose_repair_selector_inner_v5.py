"""Inner-only selector sensitivity and label-free refit-drift diagnosis.

Reads only ``row['fit']`` from the v4 selector artifacts. Outer truth,
predictions and scores are never accessed. Leave-one-score-block-out results
are selection sensitivity, not unbiased new nested risk estimates: the frozen
inner models have overlapping supervised training sets. No source is mutated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"results/repair-relation-selector-v4"
OUT = ROOT/"research/repair-selector-inner-v5-diagnosis.json"
FAMILIES = ("uniform_sqrt", "half_majority", "learned_majority", "half_local_soft", "local_soft")
RULES = ("v4_certified_tolerance", "clean_primary", "clean_minus_SE", "paired_anchor_SE", "anchor_all_fold_noninferior")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def se(v):
    return float(np.std(v, ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0.


def options(blocks):
    records=[]
    for family in FAMILIES:
        for index in range(4):
            entries=[b['families'][family]['candidates'][index] for b in blocks]
            ba=np.array([e['score']['clean_BA'] for e in entries])
            certified=np.array([e['score']['query_certified_BA'] for e in entries])
            count=[len(e['model']['features']) for e in entries]
            records.append({'family':family,'index':index,'entries':entries,'BA':ba,
                            'certified':certified,'mean':float(ba.mean()),'SE':se(ba),
                            'certified_mean':float(certified.mean()),'count':float(np.mean(count)),
                            'restored':float(np.mean([e['model']['restored_count'] for e in entries])),
                            'logloss':float(np.mean([e['score']['balanced_logloss'] for e in entries]))})
    return records


def identify(record):
    return {'family':record['family'],'recipe_index':record['index']}


def clean_key(r):
    return (r['mean'], r['certified_mean'], -r['count'], -r['logloss'], -FAMILIES.index(r['family']), -r['index'])


def choose(opts, rule):
    anchor=next(o for o in opts if o['family']=='uniform_sqrt' and o['index']==3)
    if rule=='v4_certified_tolerance':
        best=max(o['mean'] for o in opts)
        eligible=[o for o in opts if o['mean']>=best-.005-1e-12]
        return max(eligible,key=lambda o:(o['certified_mean'],*clean_key(o)))
    if rule=='clean_primary':
        return max(opts,key=clean_key)
    if rule=='clean_minus_SE':
        return max(opts,key=lambda o:(o['mean']-o['SE'],*clean_key(o)))
    if rule=='paired_anchor_SE':
        qualified=[o for o in opts if float(np.mean(o['BA']-anchor['BA']))-se(o['BA']-anchor['BA'])>=-1e-12]
        return max(qualified,key=clean_key)
    if rule=='anchor_all_fold_noninferior':
        qualified=[o for o in opts if np.min(o['BA']-anchor['BA'])>=-.005-1e-12]
        return max(qualified,key=clean_key)
    raise ValueError(rule)


def model_drift(fit,chosen,d):
    inner=[b['families'][chosen['family']]['candidates'][chosen['recipe_index']]['model'] for b in fit['inner']]
    outer=fit['selected']
    wide=[]
    for m in inner+[outer]:
        v=np.zeros(d);v[m['features']]=m['weights'];wide.append(v/v.sum())
    reference=[np.array(b['families'][chosen['family']]['reference_weights']) for b in fit['inner']]+[np.array(fit['reference_weights'])]
    reference=[v/v.sum() for v in reference]
    supports=[set(m['features']) for m in inner]
    outer_support=set(outer['features'])
    return {'inner_realized_counts':[len(m['features']) for m in inner],
            'outer_realized_count':len(outer['features']),
            'inner_restoration':[m['restored_count'] for m in inner],
            'outer_restoration':outer['restored_count'],
            'inner_uniform_fallback':[m['uniform_fallback'] for m in inner],
            'outer_uniform_fallback':outer['uniform_fallback'],
            'outer_minus_inner_mean_count':float(len(outer['features'])-np.mean([len(m['features']) for m in inner])),
            'outer_to_inner_support_jaccard':[len(outer_support&s)/len(outer_support|s) for s in supports],
            'outer_to_inner_deployed_unit_weight_L1':[float(np.abs(wide[-1]-w).sum()) for w in wide[:-1]],
            'outer_to_inner_reference_unit_weight_L1':[float(np.abs(reference[-1]-w).sum()) for w in reference[:-1]],
            'outer_trace_lengths':len(fit['outer_trace'])}


def summarize(rows):
    dataset_names=sorted(set(r['dataset'] for r in rows))
    result={}
    for rule in RULES:
        by_dataset=[]
        for name in dataset_names:
            panels=[r for r in rows if r['dataset']==name]
            scores=[probe[rule] for r in panels for probe in r['leave_one_score_block_out']]
            by_dataset.append({'dataset':name,'clean_delta_anchor_pp':float(np.mean([v['heldout_clean_delta_anchor_pp'] for v in scores])),
                'certified_delta_anchor_pp':float(np.mean([v['heldout_certified_delta_anchor_pp'] for v in scores])),
                'heldout_count':float(np.mean([v['heldout_count'] for v in scores])),
                'selection_stability':float(np.mean([v['matches_full_inner_selection'] for v in scores])),
                'noninferior_score_blocks':int(sum(v['heldout_clean_delta_anchor_pp']>=-.5-1e-9 for v in scores)),
                'score_blocks':len(scores)})
        result[rule]={'dataset_equal_clean_delta_anchor_pp':float(np.mean([v['clean_delta_anchor_pp'] for v in by_dataset])),
                      'dataset_equal_certified_delta_anchor_pp':float(np.mean([v['certified_delta_anchor_pp'] for v in by_dataset])),
                      'mean_selection_stability':float(np.mean([v['selection_stability'] for v in by_dataset])),
                      'by_dataset':by_dataset}
    return result


def main():
    rows=[];hashes={};family_details={f:[] for f in FAMILIES}
    for path in sorted(SOURCE.glob('*.json')):
        if path.stem in ('source-freeze','development-summary') or path.stem.endswith('-audit'):
            continue
        saved=json.loads(path.read_text())
        if 'dataset' not in saved:continue
        hashes[str(path.relative_to(ROOT))]=sha(path)
        for row in saved['rows']:
            fit=row['fit'];opts=options(fit['inner']);anchor=next(o for o in opts if o['family']=='uniform_sqrt' and o['index']==3)
            selected={rule:choose(opts,rule) for rule in RULES}
            assert identify(selected['v4_certified_tolerance'])=={k:fit['chosen'][k] for k in ('family','recipe_index')}
            chosen=fit['chosen'];lookup=selected['v4_certified_tolerance']
            clean=selected['clean_primary']
            dominance=[]
            for family in FAMILIES:
                full=next(o for o in opts if o['family']==family and o['index']==3)
                local=max([o for o in opts if o['family']==family],key=clean_key)
                summary={'dataset':saved['dataset'],'fold':row['fold'],'best_fraction':local['index'],
                         'best_clean_delta_anchor_pp':100*(local['mean']-anchor['mean']),
                         'best_certified_delta_anchor_pp':100*(local['certified_mean']-anchor['certified_mean']),
                         'full_clean_delta_anchor_pp':100*(full['mean']-anchor['mean']),
                         'full_certified_delta_anchor_pp':100*(full['certified_mean']-anchor['certified_mean']),
                         'candidate_uniform_fallbacks':sum(e['model']['uniform_fallback'] for o in opts if o['family']==family for e in o['entries']),
                         'candidate_full_supports':sum(len(e['model']['features'])==saved['d'] for o in opts if o['family']==family for e in o['entries']),
                         'candidate_total':4*len(fit['inner']),
                         'candidate_mean_restoration':float(np.mean([e['model']['restored_count'] for o in opts if o['family']==family for e in o['entries']]))}
                family_details[family].append(summary)
                dominance.append(summary)
            probes=[]
            for held in range(len(fit['inner'])):
                train=[b for i,b in enumerate(fit['inner']) if i!=held]
                reduced=options(train);held_anchor=fit['inner'][held]['families']['uniform_sqrt']['candidates'][3]
                scores={}
                for rule in RULES:
                    pick=choose(reduced,rule)
                    value=fit['inner'][held]['families'][pick['family']]['candidates'][pick['index']]
                    scores[rule]={**identify(pick),'heldout_clean_BA':value['score']['clean_BA'],
                                  'heldout_clean_delta_anchor_pp':100*(value['score']['clean_BA']-held_anchor['score']['clean_BA']),
                                  'heldout_certified_delta_anchor_pp':100*(value['score']['query_certified_BA']-held_anchor['score']['query_certified_BA']),
                                  'heldout_count':len(value['model']['features']),
                                  'matches_full_inner_selection':identify(pick)==identify(selected[rule])}
                probes.append(scores)
            rows.append({'dataset':saved['dataset'],'fold':row['fold'],'d':saved['d'],
                         'v4_clean_sacrifice_vs_best_pp':100*(clean['mean']-lookup['mean']),
                         'v4_certified_gain_vs_clean_best_pp':100*(lookup['certified_mean']-clean['certified_mean']),
                         'v4_selected':identify(lookup),'clean_primary_selected':identify(clean),
                         'inner_selected_clean_SE_pp':100*lookup['SE'],
                         'inner_selected_clean_fold_delta_anchor_pp':(100*(lookup['BA']-anchor['BA'])).tolist(),
                         'number_distinct_candidate_models_per_inner':[len(set((tuple(c['model']['features']),tuple(c['model']['weights'])) for fam in b['families'].values() for c in fam['candidates'])) for b in fit['inner']],
                         'refit_label_free_drift':model_drift(fit,chosen,saved['d']),
                         'leave_one_score_block_out':probes})
    result={'rows':rows,'source_hashes':hashes,'script_sha256':sha(__file__),
            'datasets':len(set(r['dataset'] for r in rows)),'outer_training_panels':len(rows),
            'inner_score_selection_sensitivity':summarize(rows),'family_diagnostics':family_details,
            'v4_clean_sacrifice_mean_pp':float(np.mean([r['v4_clean_sacrifice_vs_best_pp'] for r in rows])),
            'v4_vs_clean_primary_changed_panels':sum(r['v4_selected']!=r['clean_primary_selected'] for r in rows),
            'mean_inner_selected_clean_SE_pp':float(np.mean([r['inner_selected_clean_SE_pp'] for r in rows])),
            'outer_labels_or_scores_accessed':False,'new_fitting_or_outer_scoring_performed':False,
            'limits':['Already exposed development artifacts.','Leave-one-score-block-out probes are sensitivity only: overlapping supervised fitting sets prevent an unbiased extra validation claim.','Three score blocks make SE estimates noisy.','Label-free outer refit model/support drift is a mechanism diagnosis, not test accuracy prediction.']}
    for path,expected in hashes.items():assert sha(ROOT/path)==expected
    if OUT.exists():raise FileExistsError(OUT)
    OUT.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','family_diagnostics','source_hashes')},indent=2))


if __name__=='__main__':main()
