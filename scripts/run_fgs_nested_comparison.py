"""Independent FGS-RFRAS comparison, native selection geometry preserved.

Graph relations use normalized unweighted inputs per the source paper.
The same uniform/softvote classification metrics as iteration31 are tuned
on inner folds. The documented empty-s1 convention remains a fidelity gap.
"""
import argparse,hashlib,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from run_softvote_iteration import metric
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit,validation_scores
from certified_frfs.graph_recent import select_fgs

PARAMETERS=[(b,x) for b in (.6,.8,.9) for x in (.5,1.,2.)]


def run(name,fold):
    threadpool_limits(1);directory=ROOT/'results/fgs-nested-comparison';out=directory/f'{name}-s101-f{fold}.json'
    if out.exists():raise ValueError('preserve FGS comparison')
    source=ROOT/'results/predictive-iteration31'/out.name;prior=json.loads(source.read_text())
    X,y=load_dataset(name);tr,te=np.array(prior['train_indices']),np.array(prior['test_indices']);trace={}
    for index,record in enumerate(prior['inner_splits']):
        ii,jj=np.array(record['train']),np.array(record['validation']);A,B=preprocess_fit(X[tr][ii],X[tr][jj])
        selections={(b,x):select_fgs(A,y[tr][ii],beta=b,xi=x) for b,x in PARAMETERS}
        for family,values in record['weights'].items():
            w=np.array(values);cache={}
            for (beta,xi),selected in selections.items():
                fs=selected['features'];key=tuple(fs)
                if key not in cache:cache[key]=validation_scores(A,y[tr][ii],B,y[tr][jj],fs,w,144+index,joint=True)
                scores=cache[key];config={'beta':beta,'xi':xi};k=json.dumps([family,config],sort_keys=True)
                trace.setdefault(k,{'family':family,'config':config,'folds':[]})['folds'].append({
                    'inner':index,'features':fs,'validation':scores,'score':(scores['clean']+scores['noise01'])/2})
    for c in trace.values():
        c['score']=float(np.mean([r['score'] for r in c['folds']]))
        c['mean_size']=float(np.mean([len(r['features']) for r in c['folds']]))
    chosen=max(trace.values(),key=lambda c:(c['score'],-c['mean_size'],c['family']=='uniform'))
    A,B=preprocess_fit(X[tr],X[te]);w=metric(A,y[tr],chosen['family']);selection=select_fgs(A,y[tr],**chosen['config'])
    fs=selection['features'];scores=validation_scores(A,y[tr],B,y[te],fs,w,104,joint=True)
    directory.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),ROOT/'src/certified_frfs/graph_recent.py',ROOT/'src/certified_frfs/softvote_metric.py']
    out.write_text(json.dumps({'dataset':name,'fold':fold,'d':prior['d'],'train_indices':tr.tolist(),
        'test_indices':te.tolist(),'chosen':chosen,'trace':list(trace.values()),'features':fs,'selection':selection,
        'weights':w.tolist(),'metrics':scores,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'status':'independent_FGS_RFRAS_documented_empty_s1_convention_development',
        'limitations':'not_author_code; empty_s1_convention; 18_graph_configs_vs_36_protected_configs'},indent=2),encoding='utf-8')
    print(name,fold,'FGS',chosen['family'],scores['clean'],len(fs),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['iris','wine','sonar','wdbc','ionosphere'])
    p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,name,fold) for name in a.datasets for fold in range(3)]
        for f in as_completed(futures):f.result()
