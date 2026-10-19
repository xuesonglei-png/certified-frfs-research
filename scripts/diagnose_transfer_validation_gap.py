"""Inner-only search/validation drift for transferred-metric selectors.

Candidate observations are dependent; no inferential tests. Only completed
jobs common to38 and39 are compared, and completion status is explicit.
Outer scores are deliberately not read.
"""
import json
import numpy as np
from run_decision_iteration import ROOT


def main():
    dirs={i:ROOT/f'results/predictive-iteration{i}' for i in (38,39)}
    files=sorted({f.name for f in dirs[38].glob('*-s101-f0.json')} &
                 {f.name for f in dirs[39].glob('*-s101-f0.json')})
    rows=[];missing=0
    for iteration,directory in dirs.items():
        for filename in files:
            obj=json.loads((directory/filename).read_text())
            for row in obj['rows']:
                if row['method'] not in ('protected','utility'):continue
                for config in row['trace']:
                    for value in config['folds']:
                        record=obj['inner_splits'][value['inner']]
                        fit=next(f for f in record['redistributed_fits']
                                 if f['method']==row['method'] and f['family']==config['family'])
                        lookup={tuple(m['features']):m['utility'] for step in fit['path']['trace']
                                for m in step['ranked_moves']}
                        utility=lookup.get(tuple(value['features']))
                        if utility is None:missing+=1;continue
                        rows.append({'iteration':iteration,'dataset':obj['dataset'],
                            'method':row['method'],'family':config['family'],
                            'inner':value['inner'],'fraction':config['config']['fraction'],
                            'size':len(value['features']),'search_utility':utility,
                            'validation':value['score'],'gap':utility-value['score']})
    summary={}
    for iteration in (38,39):
        for method in ('protected','utility'):
            data=[r for r in rows if r['iteration']==iteration and r['method']==method]
            by_dataset={name:float(np.mean([r['gap'] for r in data if r['dataset']==name]))
                        for name in sorted({r['dataset'] for r in data})}
            summary[f'{iteration}:{method}']={'candidate_observations':len(data),
                'dataset_mean_gaps':by_dataset,'mean_dataset_gap':float(np.mean(list(by_dataset.values())))}
    result={'scope':'inner_only_dependent_search_validation_diagnostic_not_generalization',
        'common_completed_files':files,'complete_five_dataset_pilot':len(files)==5,
        'unscored_full_candidate_observations':missing,'summary':summary,'rows':rows}
    out=dirs[39]/'inner-transfer-gap-diagnostic.json'
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
