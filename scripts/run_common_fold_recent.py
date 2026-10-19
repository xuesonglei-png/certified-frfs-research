"""Recent native/fixed-budget baselines on the proposed inner raw folds."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from compare_trajectory_recent import GRIDS
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.linear_retention import linear_parameters
from certified_frfs.recent_budget import fgs_budget_path,sfrip_budget_path
from review_box_iterations import paired

FOLDER=ROOT/'results/common-fold-recent';CS=(.1,1.,10.);MODES=('native','0.5','0.65','0.8')
PATHS={'fgs2023':fgs_budget_path,'sfrip2025':sfrip_budget_path}

def features(path,mode,d):
    return path['native']['features'] if mode=='native' else path['ranking'][:int(np.ceil(float(mode)*d))]

def train(A,y,fs,C):
    if fs:
        model=LogisticRegression(C=C,max_iter=2000,random_state=0).fit(A[:,fs],y)
        coef,b,classes=linear_parameters(model);W=np.zeros((len(classes),A.shape[1]));W[:,fs]=coef
    else:
        classes,counts=np.unique(y,return_counts=True);W=np.zeros((len(classes),A.shape[1]));b=np.log(counts/counts.sum())
    return {'coef':W.tolist(),'intercept':b.tolist(),'classes':classes.tolist()}

def score(B,truth,fit):
    classes=np.array(fit['classes']);pred=classes[(B@np.array(fit['coef']).T+np.array(fit['intercept'])).argmax(axis=1)]
    return float(np.mean([np.mean(pred[truth==c]==c) for c in np.unique(truth)]))

def final_metrics(A,y,B,truth,fs,C,seed):
    fit=train(A,y,fs,C);rng=np.random.default_rng(seed)
    noise=[score(np.clip(B+rng.uniform(-.01,.01,B.shape),0,1),truth,fit) for _ in range(3)]
    return fit,{'clean':score(B,truth,fit),'query_noise01':float(np.mean(noise))}

def choose(trace):
    for r in trace:
        r['mean_clean']=float(np.mean([f['clean'] for f in r['folds']]))
        r['mean_size']=float(np.mean([len(f['features']) for f in r['folds']]))
    return max(range(len(trace)),key=lambda j:(trace[j]['mean_clean'],-trace[j]['mean_size'],-j))

def run(name):
    threadpool_limits(1);out=FOLDER/f'{name}.json'
    if out.exists():raise FileExistsError(out)
    stagepath=ROOT/'results/nested-teacher-regularized'/f'{name}.json';proposedpath=ROOT/'results/hierarchical-teacher-selection'/f'{name}.json'
    p=json.loads(stagepath.read_text());proposed=json.loads(proposedpath.read_text());X,y=load_dataset(name);records=[]
    for index,s in enumerate(p['splits']):
        start=time.perf_counter();ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);raw,labels=X[ri],y[ri];d=X.shape[1]
        trace={m:{mode:[{'config':cfg,'C':C,'folds':[]} for cfg in GRIDS[m] for C in CS] for mode in MODES} for m in GRIDS}
        simple={m:[{'C':C,'folds':[]} for C in CS] for m in ('full','proposed_support_logistic_diagnostic')};innerrecords=[]
        ps=proposed['splits'][index];assert ps['reference_indices']==ri.tolist() and ps['validation_indices']==vi.tolist()
        for j,inner in enumerate(s['inner']):
            a,b=np.array(inner['reference_indices']),np.array(inner['validation_indices']);A,B=preprocess_fit(raw[a],raw[b]);paths={}
            for method,grid in GRIDS.items():
                paths[method]=[PATHS[method](A,labels[a],**cfg) for cfg in grid]
                for mode in MODES:
                    for k,candidate in enumerate(trace[method][mode]):
                        fs=features(paths[method][k//len(CS)],mode,d);fit=train(A,labels[a],fs,candidate['C'])
                        candidate['folds'].append({'features':fs,'clean':score(B,labels[b],fit)})
            diagnosticfs=inner['by_C'][str(ps['teacher_C'])]['stage']['policies'][ps['chosen']]['candidate']['features']
            for method,fs in (('full',list(range(d))),('proposed_support_logistic_diagnostic',diagnosticfs)):
                for candidate in simple[method]:candidate['folds'].append({'features':fs,'clean':score(B,labels[b],train(A,labels[a],fs,candidate['C']))})
            innerrecords.append({'reference_indices':a.tolist(),'validation_indices':b.tolist(),'paths':paths})
            print(name,index,'recent inner',j,'all modes scored',flush=True)
        A,B=preprocess_fit(raw,X[vi]);results={};outerpaths={}
        for method,modes in trace.items():
            results[method]={};outerpaths[method]={}
            for mode,choices in modes.items():
                selected=choose(choices);config=choices[selected];cfgkey=json.dumps(config['config'],sort_keys=True)
                if cfgkey not in outerpaths[method]:outerpaths[method][cfgkey]=PATHS[method](A,labels,**config['config'])
                path=outerpaths[method][cfgkey];fs=features(path,mode,d)
                fit,metrics=final_metrics(A,labels,B,y[vi],fs,config['C'],144+index)
                results[method][mode]={'chosen_index':selected,'features':fs,'C':config['C'],'config':config['config'],'fit':fit,'metrics':metrics}
        for method,choices in simple.items():
            selected=choose(choices);fs=list(range(d)) if method=='full' else ps['selected']['candidate']['features']
            fit,metrics=final_metrics(A,labels,B,y[vi],fs,choices[selected]['C'],144+index)
            results[method]={'chosen_index':selected,'features':fs,'C':choices[selected]['C'],'fit':fit,'metrics':metrics}
        records.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'inner':innerrecords,'trace':trace,'simple_trace':simple,
            'results':results,'outer_paths':outerpaths,'proposed_metrics':ps['metrics'],'proposed_features':ps['selected']['candidate']['features'],
            'matched_full_metrics':ps['full_metrics'],'seconds':time.perf_counter()-start})
        print(name,index,'all recent native/budget outcomes saved in memory',flush=True)
    files=[Path(__file__),ROOT/'research/common-fold-recent-protocol.md',stagepath,proposedpath,ROOT/'scripts/compare_trajectory_recent.py',ROOT/'src/certified_frfs/recent_budget.py',ROOT/'src/certified_frfs/recent.py',ROOT/'src/certified_frfs/graph_recent.py',ROOT/'src/certified_frfs/validated_pool.py',ROOT/'data/processed'/f'{name}.npz']
    out.write_text(json.dumps({'dataset':name,'d':X.shape[1],'splits':records,'fingerprints':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}},indent=2),encoding='utf-8')

def summarize(names):
    rows=[]
    for name in names:
        p=json.loads((FOLDER/f'{name}.json').read_text());splits=p['splits'];methods={}
        for key in ('proposed','matched_full','full','proposed_support_logistic_diagnostic'):
            values=[s['proposed_metrics'] if key=='proposed' else s['matched_full_metrics'] if key=='matched_full' else s['results'][key]['metrics'] for s in splits]
            fs=[s['proposed_features'] if key=='proposed' else list(range(p['d'])) if key=='matched_full' else s['results'][key]['features'] for s in splits]
            methods[key]={'clean':float(np.mean([r['clean'] for r in values])),'noise':float(np.mean([r['query_noise01'] for r in values])),
                          'removed':float(np.mean([1-len(f)/p['d'] for f in fs]))}
        for method in GRIDS:
            for mode in MODES:
                values=[s['results'][method][mode] for s in splits]
                methods[f'{method}_{mode}']={'clean':float(np.mean([r['metrics']['clean'] for r in values])),
                    'noise':float(np.mean([r['metrics']['query_noise01'] for r in values])),
                    'removed':float(np.mean([1-len(r['features'])/p['d'] for r in values]))}
        rows.append({'dataset':name,'methods':methods,'seconds':sum(s['seconds'] for s in splits)})
    means={m:{k:float(np.mean([r['methods'][m][k] for r in rows])) for k in ('clean','noise','removed')} for m in rows[0]['methods']}
    comparisons={m:paired([r['methods']['proposed']['clean']-r['methods'][m]['clean'] for r in rows]) for m in means if m!='proposed'}
    result={'rows':rows,'means':means,'comparisons':comparisons,'scope':'common_inner_folds_recent_native_plus_explicit_budget_adaptations_exposed_development_not_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print({'means':means,'comparisons':comparisons},flush=True)

if __name__=='__main__':
    FOLDER.mkdir(exist_ok=True);names=[r['dataset'] for r in json.loads((ROOT/'results/hierarchical-teacher-selection/complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]):f.result()
    summarize(names)
