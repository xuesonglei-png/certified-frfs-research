"""Transparent development review: complete cohorts, audit lineage and outliers."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from run_repair_relation_selector_v4 import ROOT,OUT,NAMES,digest
from run_anchor_guarded_selector_v5 import FOLDER as V5

def read(p):return json.loads(p.read_text()) if p.exists() else None

def audit_status(folder,name,artifact):
    sha=digest(artifact);matches=[];failures=[]
    for p in folder.glob(name+'*audit*.json'):
        a=read(p)
        if a.get('artifact_sha256')==sha and a.get('passed'):matches.append(p.relative_to(ROOT).as_posix())
        if a.get('passed') is False:failures.append(p.relative_to(ROOT).as_posix())
    return {'passed':bool(matches),'passing_reports':matches,'retained_failures':failures,'artifact_sha256':sha}

def effect(ds):
    ds=np.array(ds,float);rng=np.random.default_rng(20261012)
    boot=rng.choice(ds,size=(20000,len(ds)),replace=True).mean(axis=1)
    return {'mean_delta_pp':float(100*ds.mean()),'median_delta_pp':float(100*np.median(ds)),
        'wins':int(np.sum(ds>1e-12)),'ties':int(np.sum(abs(ds)<=1e-12)),'losses':int(np.sum(ds< -1e-12)),
        'dataset_bootstrap95_pp':(100*np.quantile(boot,[.025,.975])).tolist(),
        'exploratory_wilcoxon_two_sided_p':float(wilcoxon(ds).pvalue) if np.any(abs(ds)>1e-12) else 1.,
        'leave_one_dataset_out_mean_range_pp':[float(100*np.delete(ds,i).mean()) for i in range(len(ds))],
        'scope':'descriptive_exposed_development_selection_reused_no_confirmatory_significance'}

def main():
    rows=[];audits={};baseline_audits={};all_complete=True;files={}
    for n in NAMES:
        p4=OUT/(n+'.json');p5=V5/(n+'.json');pb=ROOT/'results/repair-relation-controls-v4'/(n+'.json')
        s4,s5,base=read(p4),read(p5),read(pb)
        if not all(s and s['complete'] for s in (s4,s5,base)):all_complete=False;continue
        for p in (p4,p5,pb):files[p.relative_to(ROOT).as_posix()]=digest(p)
        audits[n]={'v4':audit_status(OUT,n,p4),'v5':audit_status(V5,n,p5)}
        baseline_audits[n]=audit_status(pb.parent,n,pb)
        for a,b,c in zip(s4['rows'],s5['rows'],base['rows'],strict=True):
            assert a['train_indices']==b['train_indices']==c['train_indices']
            assert a['test_indices']==b['test_indices']==c['test_indices']
        r={'dataset':n,'d':s4['d']}
        for version,s,key in (('v4',s4,'proposed'),('v5',s5,'paired_anchor')):
            def score(row):return row[key] if version=='v4' else row['scores'][key]
            def model(row):return row['fit']['selected'] if version=='v4' else row['fits'][key]['selected']
            for metric in ('clean_BA','query_certified_BA','query_coverage','balanced_query_coverage','macro_F1','balanced_logloss'):
                r[version+'_'+metric]=float(np.mean([score(x)[metric] for x in s['rows']]))
            r[version+'_removed']=float(np.mean([1-len(model(x)['features'])/s['d'] for x in s['rows']]))
            r[version+'_fallbacks']=sum(model(x)['uniform_fallback'] for x in s['rows'])
            r[version+'_full_supports']=sum(len(model(x)['features'])==s['d'] for x in s['rows'])
        for metric in ('clean_BA','query_certified_BA','query_coverage','balanced_query_coverage'):
            r['full_'+metric]=float(np.mean([x['full'][metric] for x in s4['rows']]))
        for m in base['rows'][0]['scores']:
            vals=[x['scores'][m] for x in base['rows']]
            for metric in ('clean_BA','query_certified_BA','query_coverage'):
                r[m+'_'+metric]=float(np.mean([x[metric] for x in vals])) if all(x is not None for x in vals) else None
        rows.append(r)
    comparisons={}
    for v in ('v4','v5'):
        comparisons[v]={}
        for baseline in ('full','fgs_native_uniform','fgs_native_capped_transfer','fgs_native_local_soft',
                         'sfrip_native_uniform','sfrip_native_capped_transfer','sfrip_native_local_soft','full_local_soft_unguarded'):
            ds=[r[v+'_clean_BA']-r[baseline+'_clean_BA'] for r in rows if r.get(baseline+'_clean_BA') is not None]
            comparisons[v][baseline]=effect(ds) if ds else {'available_datasets':0,'unavailable':True,
                'reason':'no complete nested native support compatible with unchanged capped-transfer geometry'}
        for metric in ('query_certified_BA','query_coverage','balanced_query_coverage'):
            comparisons[v]['full_'+metric]=effect([r[v+'_'+metric]-r['full_'+metric] for r in rows])
    result={'complete':all_complete,'datasets':len(rows),'outer_folds':3*len(rows),'rows':rows,
        'comparisons':comparisons,'method_audits':audits,'baseline_audits':baseline_audits,
        'all_method_audits_passed':all_complete and all(x[v]['passed'] for x in audits.values() for v in ('v4','v5')),
        'all_baseline_audits_passed':all_complete and all(x['passed'] for x in baseline_audits.values()),
        'mean_removed':{v:float(np.mean([r[v+'_removed'] for r in rows])) for v in ('v4','v5')},
        'source_hashes':files,'classification_superiority_established':False,'goal_achieved':False,
        'limits':['Six exposed development datasets: no independent confirmation.',
                 'Native/capped/local-soft variants are distinct numerical reproductions/adaptations.',
                 'SFRIP capped native arm is unavailable on complete folds; no weighted-native win is claimed.',
                 'Positive mean against FGS native uniform is driven by one dataset; wins and leave-one-out range are essential.',
                 'Query certificates prove stability, not truth or population risk.',
                 'No additional runtime benefit established in this experiment.']}
    output=ROOT/'research/repair-iterations-v4-v5-review.json';output.write_text(json.dumps(result,indent=2))
    csv_path=ROOT/'research/repair-iterations-v4-v5-dataset-means.csv'
    with csv_path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'complete':all_complete,'datasets':len(rows),
        'method_audits':result['all_method_audits_passed'],'baseline_audits':result['all_baseline_audits_passed'],
        'mean_removed':result['mean_removed'],
        'contrasts':{v:{m:round(c.get('mean_delta_pp',0),4) for m,c in result['comparisons'][v].items()} for v in ('v4','v5')}},indent=2))

if __name__=='__main__':main()
