"""Training-side beam versus greedy utility at identical cardinalities.

No held-out labels or scores; excludes full subset if greedy never scored it.
Beam width3 is not guaranteed to dominate width1 at every later layer.
"""
import json
from run_decision_iteration import ROOT


def objective(path):
    if 'candidate_utilities' in path:
        return {tuple(r['features']):r['utility'] for r in path['candidate_utilities']}
    return {tuple(m['features']):m['utility'] for s in path['trace'] for m in s['ranked_moves']}


def compare(beam,greedy,scope):
    a,b=objective(beam),objective(greedy);rows=[]
    sizes=sorted(set(map(len,beam['candidates']))&set(map(len,greedy['candidates'])))
    for size in sizes:
        left=[a[tuple(fs)] for fs in beam['candidates'] if len(fs)==size and tuple(fs) in a]
        right=[b[tuple(fs)] for fs in greedy['candidates'] if len(fs)==size and tuple(fs) in b]
        if left and right:rows.append({**scope,'cardinality':size,'beam_best':max(left),
                                      'greedy':max(right),'delta':max(left)-max(right)})
    return rows


def main():
    directory=ROOT/'results/predictive-iteration37';rows=[]
    for file in sorted(directory.glob('*-s101-f*.json')):
        o=json.loads(file.read_text());old=json.loads((ROOT/'results/predictive-iteration36'/file.name).read_text())
        for index,(new,prior) in enumerate(zip(o['inner_splits'],old['inner_splits'])):
            for fit in new['utility_fits']:
                p=next(p for p in prior['utility_fits'] if p['method']==fit['method'] and p['family']==fit['family'])
                rows.extend(compare(fit['path'],p['path'],{'dataset':o['dataset'],'fold':o['fold'],
                    'inner':index,'method':fit['method'],'family':fit['family']}))
    summary={method:{'matched_cardinalities':len(data),
        'better_equal_worse':[sum(r['delta']>1e-12 for r in data),sum(abs(r['delta'])<=1e-12 for r in data),sum(r['delta']< -1e-12 for r in data)],
        'descriptive_mean_delta':sum(r['delta'] for r in data)/len(data)}
        for method in ('protected','utility') if (data:=[r for r in rows if r['method']==method])}
    result={'scope':'training_inner_search_same_cardinality_dependent_descriptive_not_test_gain',
            'summary':summary,'rows':rows}
    (directory/'training-beam-diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
