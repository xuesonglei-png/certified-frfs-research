"""Correct same-fuzzy-classifier attribution without mixing raw LP controls."""
import json
import numpy as np
from review_box_iterations import ROOT,paired
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_classifier import triangular_box_margins

FOLDER=ROOT/'results/common-fuzzy-classifier'

def main():
    old=json.loads((FOLDER/'complete.json').read_text());rows=[]
    for row in old['rows']:
        name=row['dataset'];p=json.loads((FOLDER/f'{name}.json').read_text());X,y=load_dataset(name);methods={}
        for method in ('full','proposed_support_logistic_diagnostic','fgs2023_native','sfrip2025_native','fgs2023_0.5','fgs2023_0.65','fgs2023_0.8','sfrip2025_0.5','sfrip2025_0.65','sfrip2025_0.8'):
            values=[];coverage=[];sizes=[]
            for s in p['splits']:
                if method in ('full','proposed_support_logistic_diagnostic'):r=s['results'][method]
                else:
                    family,mode=method.split('_',1);r=s['results'][family][mode]
                ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);_,B=preprocess_fit(X[ri],X[vi]);f=r['fit']
                margin=triangular_box_margins(B,y[vi],np.array(f['coef']),np.array(f['intercept']),np.array(f['classes']))
                coverage.append(float(np.mean([np.mean(margin[y[vi]==c]>1e-10) for c in np.unique(y[vi])])))
                values.append(r['metrics']);sizes.append(len(r['features']))
            methods[method]={'clean':float(np.mean([v['clean'] for v in values])),'noise':float(np.mean([v['query_noise01'] for v in values])),
                             'certified01':float(np.mean(coverage)),'removed':1-float(np.mean(sizes))/p['d']}
        rows.append({'dataset':name,'methods':methods})
    reference='proposed_support_logistic_diagnostic'
    result={'rows':rows,'means':{m:{k:float(np.mean([r['methods'][m][k] for r in rows])) for k in ('clean','noise','certified01','removed')} for m in rows[0]['methods']},
        'comparisons':{m:paired([r['methods'][reference]['clean']-r['methods'][m]['clean'] for r in rows]) for m in rows[0]['methods'] if m!=reference},
        'scope':'same_fuzzy_classifier_saved_proposed_support_post_selection_diagnostic_not_new_certified_reduction_or_confirmation'}
    (FOLDER/'complete-fuzzy-review.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print({k:v for k,v in result.items() if k!='rows'})

if __name__=='__main__':main()
