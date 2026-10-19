"""Describe search-to-validation gaps using inner data, never outer scores.

Full-feature candidates have no cached deletion score and are excluded,
explicitly counted. Candidate observations are dependent; no significance
tests or generalization claims are computed here.
"""
import json
import numpy as np
from run_decision_iteration import ROOT


def main():
    rows=[];summary={}
    for iteration in (35,36):
        directory=ROOT/f'results/predictive-iteration{iteration}'
        for file in sorted(directory.glob('*-s101-f*.json')):
            obj=json.loads(file.read_text())
            for method in ('protected','utility'):
                row=next(r for r in obj['rows'] if r['method']==method)
                fits={(index,fit['family']):fit['path'] for index,r in enumerate(obj['inner_splits'])
                      for fit in r['utility_fits'] if fit['method']==method}
                for config in row['trace']:
                    for f in config['folds']:
                        path=fits[f['inner'],config['family']]
                        lookup={tuple(move['features']):move['utility'] for step in path['trace']
                                for move in step['ranked_moves']}
                        search_score=lookup.get(tuple(f['features']))
                        rows.append({'iteration':iteration,'dataset':obj['dataset'],'fold':obj['fold'],
                            'method':method,'family':config['family'],'fraction':config['config']['fraction'],
                            'inner':f['inner'],'size':len(f['features']),'search_score':search_score,
                            'validation_score':f['score'],
                            'gap':None if search_score is None else search_score-f['score']})
        for method in ('protected','utility'):
            selected=[r for r in rows if r['iteration']==iteration and r['method']==method]
            available=[r for r in selected if r['gap'] is not None]
            groups={name:float(np.mean([r['gap'] for r in available if r['dataset']==name]))
                    for name in sorted({r['dataset'] for r in available})}
            summary[f'{iteration}:{method}']={'available_candidate_scores':len(available),
                'full_feature_unscored':len(selected)-len(available),'dataset_mean_gaps':groups,
                'mean_dataset_gap':float(np.mean(list(groups.values())))}
    out=ROOT/'results/predictive-iteration36/search-validation-diagnostic.json'
    out.write_text(json.dumps({'scope':'dependent_inner_candidate_descriptive_diagnostic_no_outer_scores',
                               'summary':summary,'rows':rows},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
