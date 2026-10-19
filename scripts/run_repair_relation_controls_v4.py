"""Full-grid recent baselines plus equally trained predictive metric controls."""
import argparse,json,time
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.relation_support_selector_v1 import weighted_knn,_candidate
from certified_frfs.selective_support_selector_v3 import query_scores
from run_selective_recent_controls_v2 import paths,build_metric,empty_score
from repair_relation_selector_v4 import local_soft_relation
from run_repair_relation_selector_v4 import ROOT,OUT,NAMES,digest,freeze

FOLDER=ROOT/'results/repair-relation-controls-v4'
VARIANTS=('native_uniform','native_capped_transfer','matched_uniform',
          'matched_capped_transfer','native_local_soft','matched_local_soft')

def metrics(A,path,w,count):
    result={v:build_metric(A,path,v,count) for v in VARIANTS[:4]}
    for prefix in ('native','matched'):
        fs=result[prefix+'_uniform']['features']
        reduced=w[fs];norm=np.linalg.norm(reduced)
        result[prefix+'_local_soft']={'features':fs,'weights':(reduced/norm).tolist() if norm else [],
            'count':len(fs),'status':'available',
            'adaptation':'same_training_local_soft_relation_restricted_L2_normalized_no_guard'}
    return result

def simple_score(A,y,B,truth,c):
    if c['weights'] is None:return None
    if not c['features']:return empty_score(y,truth)
    return weighted_knn(A,y,B,truth,c['features'],c['weights'])

def candidate_block(A,y,B,truth,count):
    registry=paths(A,y);soft,record=local_soft_relation(A,y);cache={};options={}
    for algorithm,rows in registry.items():
        for variant in VARIANTS:options[algorithm+'_'+variant]=[]
        for index,(config,path) in enumerate(rows):
            arms=metrics(A,path,soft,count)
            for variant,c in arms.items():
                key=(tuple(c['features']),None if c['weights'] is None else tuple(c['weights']))
                if key not in cache:cache[key]=simple_score(A,y,B,truth,c)
                options[algorithm+'_'+variant].append({'config_index':index,'config':config,
                    'metric':c,'score':cache[key]})
    return {'options':options,'local_soft_fit':record,'raw_paths':registry,
            'matched_count':count,'unique_metrics':len(cache)}

def select(blocks,method):
    rows=[]
    for index in range(len(blocks[0]['options'][method])):
        cc=[b['options'][method][index] for b in blocks]
        if any(c['score'] is None for c in cc):
            rows.append({'config_index':index,'available':False});continue
        rows.append({'config_index':index,'available':True,
            'mean_BA':float(np.mean([c['score']['clean_BA'] for c in cc])),
            'mean_logloss':float(np.mean([c['score']['balanced_logloss'] for c in cc])),
            'mean_count':float(np.mean([len(c['metric']['features']) for c in cc]))})
    valid=[r for r in rows if r['available']]
    chosen=min(valid,key=lambda r:(-round(r['mean_BA'],12),round(r['mean_logloss'],12),
                                    r['mean_count'],r['config_index'])) if valid else None
    return {'options':rows,'chosen':chosen}

def run(name):
    threadpool_limits(1);freeze();FOLDER.mkdir(exist_ok=True)
    source=OUT/(name+'.json');parent=json.loads(source.read_text());assert parent['complete']
    sha=digest(source);out=FOLDER/(name+'.json');saved=json.loads(out.read_text()) if out.exists() else None
    if saved:
        assert saved['parent_sha256']==sha
        if saved['complete']:return
    result_rows=saved['rows'] if saved else [];X,y=load_dataset(name);start=time.perf_counter()
    for row in parent['rows']:
        fold=row['fold']
        if fold<len(result_rows):continue
        ri,vi=np.array(row['train_indices']),np.array(row['test_indices']);raw,labels=X[ri],y[ri]
        chosen=row['fit']['chosen'];inners=[]
        for b in row['fit']['inner']:
            ii,jj=np.array(b['fit_indices']),np.array(b['validation_indices'])
            count=len(b['families'][chosen['family']]['candidates'][chosen['recipe_index']]['model']['features'])
            A,B=preprocess_fit(raw[ii],raw[jj]);stage=candidate_block(A,labels[ii],B,labels[jj],count)
            inners.append({'fit_indices':ii.tolist(),'validation_indices':jj.tolist(),**stage})
            print(name,fold,'baseline inner',len(inners),'unique',stage['unique_metrics'],flush=True)
        picks={m:select(inners,m) for m in inners[0]['options']}
        A,B=preprocess_fit(raw,X[vi]);registry=paths(A,labels);soft,record=local_soft_relation(A,labels)
        count=len(row['fit']['selected']['features']);models={};scores={};cache={}
        for method,pick in picks.items():
            c=pick['chosen']
            if c is None:models[method]=None;scores[method]=None;continue
            algorithm,variant=method.split('_',1);config,path=registry[algorithm][c['config_index']]
            metric=metrics(A,path,soft,count)[variant];models[method]=metric
            if metric['weights'] is None:scores[method]=None;continue
            key=(tuple(metric['features']),tuple(metric['weights']))
            if key not in cache:
                cache[key]=query_scores(A,labels,B,y[vi],metric) if metric['features'] else empty_score(labels,y[vi])
            scores[method]=cache[key]
        full={'features':list(range(A.shape[1])),'weights':soft.tolist()}
        guarded=_candidate(A,labels,soft,list(range(A.shape[1])),A.shape[1])
        original={'features':list(range(A.shape[1])),'weights':(np.ones(A.shape[1])/np.sqrt(A.shape[1])).tolist()}
        models['full_local_soft_unguarded']=full
        models['full_local_soft_guarded']=guarded if guarded['joint_feasible'] else original
        for m in ('full_local_soft_unguarded','full_local_soft_guarded'):scores[m]=query_scores(A,labels,B,y[vi],models[m])
        noise={m:[] for m in models}
        for draw in range(8):
            rng=np.random.default_rng(20262000+10*fold+draw)
            na=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1);nb=np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
            for m,c in models.items():
                if c is None or c['weights'] is None:continue
                s=simple_score(na,labels,nb,y[vi],c);noise[m].append(s)
                if c['features']:
                    flags=np.array(scores[m]['query_certificate']['certified'])
                    assert not np.any(flags & (np.array(s['predictions'])!=scores[m]['predictions']))
        result_rows.append({'fold':fold,'train_indices':ri.tolist(),'test_indices':vi.tolist(),
            'inner':inners,'selection':picks,'models':models,'scores':scores,
            'outer_paths':registry,'local_soft_fit':record,'full_local_soft_attempt':guarded,
            'shared_perturbations':noise,'perturbed_BA':{m:float(np.mean([s['clean_BA'] for s in vals])) if vals else None for m,vals in noise.items()}})
        assert sha==digest(source);freeze()
        final={'dataset':name,'d':A.shape[1],'rows':result_rows,'parent_sha256':sha,
            'complete':len(result_rows)==3,'seconds':time.perf_counter()-start,
            'grid_sizes':{'fgs':120,'sfrip':6},'independent_audit_passed':False,
            'scope':'complete_grid_nested_exposed_development_numeric_reproductions_and_labelled_adaptations'}
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(final,separators=(',',':')));tmp.replace(out)
        print(name,fold,'recent baselines', {m:round(100*s['clean_BA'],3) if s else None for m,s in scores.items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--only',choices=NAMES);args=p.parse_args()
    for n in ((args.only,) if args.only else NAMES):run(n)
