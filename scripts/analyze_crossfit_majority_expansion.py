"""Dataset-level development sensitivity analysis; no confirmatory claims."""
import json
from itertools import product
import numpy as np
from scipy.stats import wilcoxon
from run_decision_iteration import ROOT


def signflip_p(delta):
    delta=np.asarray(delta,float);nonzero=delta[np.abs(delta)>1e-12]
    if not len(nonzero):return 1.
    signs=np.array(list(product([-1.,1.],repeat=len(nonzero))))
    observed=abs(nonzero.sum())
    return float(np.mean(abs(signs@nonzero)>=observed-1e-12))


def analyze(grouped,excluded,metric='clean'):
    names=sorted(set(grouped['protected'])-set(excluded));methods={}
    for method,data in grouped.items():
        means={name:{key:float(np.mean([r[key] for r in data[name]])) for key in data[name][0]}
               for name in names}
        methods[method]={'datasets':means,'means':{key:float(np.mean([r[key] for r in means.values()]))
                        for key in next(iter(means.values()))}}
    pairs=[]
    for method in methods:
        if method=='protected':continue
        delta=np.array([methods['protected']['datasets'][name][metric]-
                        methods[method]['datasets'][name][metric] for name in names])
        rng=np.random.default_rng(20261009)
        boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
        pairs.append({'comparator':method,'mean_delta':float(delta.mean()),
            'median_delta':float(np.median(delta)),
            'wins_ties_losses':[int((delta>1e-12).sum()),int((abs(delta)<=1e-12).sum()),int((delta< -1e-12).sum())],
            'raw_wilcoxon_p':float(wilcoxon(delta).pvalue) if np.any(abs(delta)>1e-12) else 1.,
            'raw_signflip_p':signflip_p(delta),
            'descriptive_bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist()})
    for field in ('wilcoxon','signflip'):
        maximum=0.
        for rank,i in enumerate(sorted(range(len(pairs)),key=lambda i:pairs[i]['raw_'+field+'_p'])):
            maximum=max(maximum,min(1.,(len(pairs)-rank)*pairs[i]['raw_'+field+'_p']))
            pairs[i]['holm_'+field+'_p']=maximum
    return {'datasets':names,'methods':methods,'pairs':pairs}


def main():
    directory=ROOT/'results/predictive-iteration26';grouped={};count=0
    for path in sorted(directory.glob('*-s101-f*.json')):
        obj=json.loads(path.read_text());count+=1
        for row in obj['rows']:
            grouped.setdefault(row['method'],{}).setdefault(obj['dataset'],[]).append({
                **row['metrics'],'removed':1-len(row['features'])/obj['d'],
                'candidate_count':len(row['trace'])})
    expected={'banknote','ecoli','glass','haberman','heart_statlog','ionosphere','iris','seeds',
              'sonar','transfusion','vehicle','vertebral','wdbc','wine','yeast'}
    assert set(grouped['protected'])==expected
    assert count==45 and all(len(rows)==3 for data in grouped.values() for rows in data.values())
    result={'status':'repeated_development_not_confirmation','all15':analyze(grouped,[]),
        'without_ecoli':analyze(grouped,['ecoli']),
        'limitations':'unequal_recipe_budgets; repeated_outer_test_exposure; bootstrap does not correct algorithm selection bias'}
    (directory/'expansion-analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for key in ('all15','without_ecoli'):
        analysis=result[key]
        print(json.dumps({'scope':key,'means':{m:r['means'] for m,r in analysis['methods'].items()},
                          'pairs':analysis['pairs']},indent=2))

if __name__=='__main__':main()
