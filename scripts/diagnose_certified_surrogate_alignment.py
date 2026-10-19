"""Adaptive training diagnostic, never a validation-driven iteration selector."""
import json,hashlib
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit,validation_scores

FOLDER=ROOT/'results/certified-surrogate-alignment'


def main():
    threadpool_limits(1);FOLDER.mkdir(exist_ok=True);summary=[]
    prior=json.loads((ROOT/'results/certified-query-refinement/complete.json').read_text())
    for row in prior['rows']:
        name=row['dataset'];out=FOLDER/f'{name}.json'
        if out.exists():raise FileExistsError(out)
        path=ROOT/'results/certified-query-refinement'/f'{name}.json';p=json.loads(path.read_text());X,y=load_dataset(name);records=[]
        for index,s in enumerate(p['splits']):
            ri=np.array(s['reference_indices']);raw,labels=X[ri],y[ri];nodes=[]
            folds=[]
            for inner,split in enumerate(s['fit']['training_splits']):
                ti,qi=np.array(split['reference']),np.array(split['query']);assert set(ti).isdisjoint(qi)
                A,B=preprocess_fit(raw[ti],raw[qi]);folds.append((inner,ti,qi,A,B))
            for node in s['fit']['trace']:
                w=np.array(node['weights']);fs=np.flatnonzero(w>0).tolist();scores=[]
                for inner,ti,qi,A,B in folds:
                    metrics=validation_scores(A,labels[ti],B,labels[qi],fs,w,31001+inner,joint=True)
                    scores.append(metrics)
                means={k:float(np.mean([m[k] for m in scores])) for k in ('clean','noise01')}
                nodes.append({'iteration':node['iteration'],'surrogate':node['objective'],
                    'adaptive_training_metrics':means,'utility':(means['clean']+means['noise01'])/2})
            changes=[{'iteration':b['iteration'],'surrogate_decrease':a['surrogate']-b['surrogate'],
                'vote_change':b['utility']-a['utility']} for a,b in zip(nodes,nodes[1:])]
            assert all(c['surrogate_decrease']>0 for c in changes)
            records.append({'split':index,'nodes':nodes,'changes':changes,'external_final_minus_initial':
                (s['metrics']['clean']+s['metrics']['noise01']-s['initial_metrics']['clean']-s['initial_metrics']['noise01'])/2})
        result={'dataset':name,'splits':records,'prior_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'adaptive_training_alignment_diagnostic_not_crossvalidation_estimate_or_iteration_selection'}
        out.write_text(json.dumps(result,indent=2),encoding='utf-8')
        changes=[c for s in records for c in s['changes']]
        summary.append({'dataset':name,'updates':len(changes),'vote_improved':sum(c['vote_change']>1e-12 for c in changes),
            'vote_decreased':sum(c['vote_change'] < -1e-12 for c in changes),
            'vote_unchanged':sum(abs(c['vote_change'])<=1e-12 for c in changes),
            'mean_adaptive_vote_final_minus_initial':float(np.mean([s['nodes'][-1]['utility']-s['nodes'][0]['utility'] for s in records])),
            'mean_external_final_minus_initial':float(np.mean([s['external_final_minus_initial'] for s in records]))})
        print(name,summary[-1],flush=True)
    result={'rows':summary,'updates':sum(r['updates'] for r in summary),
        'vote_improved':sum(r['vote_improved'] for r in summary),'vote_decreased':sum(r['vote_decreased'] for r in summary),
        'vote_unchanged':sum(r['vote_unchanged'] for r in summary),
        'scope':'descriptive_adaptive_training_alignment_not_independent_accuracy_or_causal_proof'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':main()
