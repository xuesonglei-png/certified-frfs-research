"""Compare cached scores/order with every recorded candidate in completed39 jobs.

Does not reexecute certificate search or claim exact full-path equivalence.
Saves disagreements rather than changing tolerances or ignoring near ties.
"""
import argparse,json,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.cached_transfer_utility import CachedCrossfitTransferUtility


def audit(X,y,family,path):
    ws=CachedCrossfitTransferUtility(X,y,family=family)
    assert ws.splits==path['ranking_splits'];mismatch=[];orders=[];checked=0
    for index,step in enumerate(path['trace']):
        rescored=[]
        for move in step['ranked_moves']:
            actual=ws.evaluate(move['features']);checked+=1
            delta={k:actual[k]-move[k] for k in ['clean_LOO','shared_noise_LOO','utility']}
            if any(abs(v)>1e-12 for v in delta.values()):
                mismatch.append({'step':index,'features':move['features'],'delta':delta})
            rescored.append({**move,**actual})
        rescored.sort(key=lambda r:(-r['utility'],-r['clean_LOO'],r['feature']))
        old=[r['feature'] for r in step['ranked_moves']];new=[r['feature'] for r in rescored]
        if old!=new:orders.append({'step':index,'old_order':old,'new_order':new})
    return {'candidate_scores':checked,'score_disagreements':mismatch,'rank_disagreements':orders}


def main(datasets):
    threadpool_limits(1);directory=ROOT/'results/predictive-iteration39';rows=[]
    files=[directory/f'{name}-s101-f0.json' for name in datasets]
    assert all(f.exists() for f in files),'requested tasks incomplete'
    for file in files:
        o=json.loads(file.read_text());X,y=load_dataset(o['dataset']);tr=np.array(o['train_indices']);checks=[]
        for index,rec in enumerate(o['inner_splits']):
            ii=np.array(rec['train']);workspaces={}
            for fit in rec['redistributed_fits']:
                result=audit(X[tr][ii],y[tr][ii],fit['family'],fit['path'])
                checks.append({'inner':index,'method':fit['method'],'family':fit['family'],**result})
        for r in o['rows']:
            if r['method'] not in ('protected','utility'):continue
            checks.append({'inner':None,'method':r['method'],'family':r['chosen']['family'],
                **audit(X[tr],y[tr],r['chosen']['family'],r['path'])})
        rows.append({'file':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'checks':checks})
        print('cached candidates checked',file.name,flush=True)
    checks=[c for r in rows for c in r['checks']]
    result={'scope':'all_recorded_candidate_scores_and_layer_orders_not_complete_certificate_search_regeneration',
        'complete_five_dataset_pilot':len(files)==5,'rows':rows,
        'totals':{'paths':len(checks),'candidate_scores':sum(c['candidate_scores'] for c in checks),
            'score_disagreements':sum(len(c['score_disagreements']) for c in checks),
            'rank_disagreements':sum(len(c['rank_disagreements']) for c in checks)}}
    out=directory/'cached-candidate-audit-selected.json'
    out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result['totals']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['iris','wine','wdbc','ionosphere']);main(p.parse_args().datasets)
