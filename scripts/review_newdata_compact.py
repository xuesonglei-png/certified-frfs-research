"""Secondary diagnostics; never changes the frozen primary comparison."""
import json
import numpy as np
from run_newdata_compact import ROOT, FOLDER, DATA, SOURCES


def main():
    result=json.loads((FOLDER/'complete.json').read_text()); rows=[]
    for name in SOURCES:
        p=json.loads((FOLDER/f'{name}.json').read_text()); counts={}; full_support=0; vacuous=0; test_counts=[]
        with np.load(DATA/f'{name}.npz') as data:y=data['y']
        for s in p['splits']:
            fit=s['fit']; selected=fit['compact_selected']; full_support+=len(selected['features'])==p['d']
            vacuous+=sum(selected['certificate']['required'])==0
            test_counts.append({str(label):int(sum(y[np.array(s['test_indices'])]==label)) for label in np.unique(y)})
            for c in fit['candidates']:
                if c['origin']!='capped_sparse':continue
                cert=c['certificate']; key=f"majority={cert['majority_feasible']},loss={cert['dependency_certificate']['upper_bound']<=.1}"
                counts[key]=counts.get(key,0)+1
        rows.append({'dataset':name,'full_support_splits':int(full_support),'vacuous_majority_splits':int(vacuous),
            'sparse_certificate_categories':counts,'test_class_counts':test_counts})
    sensitivity={}
    for method in ('fgs2023','sfrip2025'):
        delta=np.array([r['utility']['proposed']-r['utility'][method] for r in result['rows']])
        sensitivity[method]={'leave_one_dataset_out':{r['dataset']:float((delta.sum()-delta[i])/(len(delta)-1))
            for i,r in enumerate(result['rows'])},'secondary_metric_gains':{k:float(np.mean([r['metrics']['proposed'][k]-r['metrics'][method][k] for r in result['rows']]))
            for k in ('clean','noise01')}}
    output={'rows':rows,'sensitivity':sensitivity,'scope':'secondary_diagnostics_not_changed_primary_selection_or_inference',
        'methodology_note':'Parkinsons record_split_subject_overlap; Lymphography_missing_test_classes; see newdata-methodology-addendum.md'}
    (FOLDER/'complete-secondary.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':main()
