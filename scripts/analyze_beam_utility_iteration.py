"""Frozen beam pilot: nine corrected clean comparisons, no confirmation."""
import json
from run_decision_iteration import ROOT
from analyze_crossfit_majority_expansion import analyze


def main():
    directory=ROOT/'results/predictive-iteration37';files=sorted(directory.glob('*-s101-f*.json'))
    assert len(files)==15;grouped={};changes=[]
    for file in files:
        o=json.loads(file.read_text());old=json.loads((ROOT/'results/predictive-iteration36'/file.name).read_text())
        assert o['beam_width']==3 and o['crossfit_ranking'] and o['dependency_eta']==.1 and o['shared_loss']
        for r in o['rows']:
            if r['method'] not in ('protected','utility'):
                previous=next(p for p in old['rows'] if p['method']==r['method'])
                for k in ('features','weights','metrics','chosen','trace'):assert r[k]==previous[k]
            grouped.setdefault(r['method'],{}).setdefault(o['dataset'],[]).append({**r['metrics'],'removed':1-len(r['features'])/o['d']})
        for method,alias in [('protected','greedy_protected36'),('utility','greedy_utility36')]:
            r=next(r for r in old['rows'] if r['method']==method)
            grouped.setdefault(alias,{}).setdefault(o['dataset'],[]).append({**r['metrics'],'removed':1-len(r['features'])/o['d']})
        r=next(r for r in o['rows'] if r['method']=='protected');p=next(r for r in old['rows'] if r['method']=='protected')
        changes.append({'dataset':o['dataset'],'fold':o['fold'],'old_selected_size':len(p['features']),
            'new_selected_size':len(r['features']),'old_minimum_size':min(map(len,p['path']['candidates'])),
            'new_minimum_size':min(map(len,r['path']['candidates'])),
            'clean_delta':r['metrics']['clean']-p['metrics']['clean'],
            'old_training_search_evaluations':p['path']['training_utility_evaluations'],
            'new_training_search_evaluations':r['path']['training_utility_evaluations']})
    primary=analyze(grouped,[]);means={m:r['means'] for m,r in primary['methods'].items()}
    gates={'clean_gain_over36_at_least_half_point':means['protected']['clean']-means['greedy_protected36']['clean']>=.005,
        'compression_no_collapse':means['protected']['removed']>=means['greedy_protected36']['removed']-.02,
        'no_mean_deterioration_vs_full':means['protected']['clean']>=means['all']['clean']}
    result={'status':'frozen_beam_pilot_repeated_development_not_confirmation','primary':primary,
        'noisy_secondary':analyze(grouped,[],metric='noise01'),'gates':gates,'gate_passed':all(gates.values()),'changes':changes}
    (directory/'beam-analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'means':means,'gates':gates,'pairs':primary['pairs']},indent=2))

if __name__=='__main__':main()
