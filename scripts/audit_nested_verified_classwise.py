"""Full nested teacher refit, primal/deployed coverage and selector replay."""
import hashlib,json
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.linear_retention import linear_parameters
from certified_frfs.feasible_transport import weight_certificate
from audit_classwise_box_pilot import deployed_check
from audit_verified_classwise_box_pilot import verify_point
from audit_nested_teacher_support import independent_score

FOLDER=ROOT/'results/nested-verified-classwise'

def stage_audit(raw,y,query,truth,stage,stored,counts,seed=None):
    A,B=preprocess_fit(raw,query);model=LogisticRegression(C=stage['teacher_C'],max_iter=2000,random_state=0).fit(A,y)
    W,b,classes=linear_parameters(model)
    np.testing.assert_allclose(W,stage['teacher_coef'],atol=1e-12);np.testing.assert_allclose(b,stage['teacher_intercept'],atol=1e-12)
    np.testing.assert_array_equal(classes,stage['teacher_classes'])
    mass=np.sqrt(stage['d']);full=np.ones(stage['d'])/mass;seen=set();queries=[B]
    if seed is not None:
        rng=np.random.default_rng(seed);queries += [np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
    scores={}
    for k,r in stage['policies'].items():
        c=r['candidate'];fs=c['features'];fit=r['fit'];identity=(tuple(fs),json.dumps(fit['coef']),json.dumps(fit['intercept']))
        if identity not in seen:
            v=np.array(c['weights']);assert abs(v.sum()-mass)<1e-8 and np.all(v>=0) and v.max()<=2/mass+1e-8
            assert weight_certificate(A,y,fs,full,v[fs])['feasible']
            deployed_check(A,y,classes,W,b,fit,bounded='primal_solution' in fit)
            assert np.all(np.array(fit['coef'])[:,np.setdiff1d(np.arange(stage['d']),fs)]==0)
            if 'primal_solution' in fit:
                verify_point(A,y,stage,{'candidate':c,'fit':fit});counts['primal_points']+=1
                counts['nonoptimal_points']+=not fit['solver_optimal']
            else:
                np.testing.assert_array_equal(fit['coef'],W);np.testing.assert_array_equal(fit['intercept'],b)
                assert len(fs)==stage['d'];counts['teacher_fallback_models']+=1
            seen.add(identity);counts['models']+=1
        values=[independent_score(Q,truth,classes,fit) for Q in queries]
        scores[k]={'clean':values[0],'retained':len(fs)/stage['d'],'query_noise01':float(np.mean(values[1:])) if seed is not None else None}
        if stored is not None:
            for field in ('clean','retained'):np.testing.assert_allclose(scores[k][field],stored[k][field],atol=1e-12)
        if r['fallback']:assert len(fs)==stage['d'] and r['failure'] is not None
        counts['prediction_replays']+=len(queries)
    counts['stages']+=1
    return scores

def main(names=None):
    threadpool_limits(1)
    summary=json.loads((FOLDER/'complete.json').read_text()) if names is None else {'rows':[{'dataset':name} for name in names]}
    counts={'outer_splits':0,'inner_splits':0,'stages':0,'models':0,'primal_points':0,'nonoptimal_points':0,'teacher_fallback_models':0,'prediction_replays':0}
    for row in summary['rows']:
        name=row['dataset'];p=json.loads((FOLDER/f'{name}.json').read_text());X,y=load_dataset(name)
        for rel,digest in p['fingerprints'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
        for i,s in enumerate(p['splits']):
            ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);assert set(ri).isdisjoint(vi);raw,labels=X[ri],y[ri];queryids=[]
            labels_unique,labelcounts=np.unique(labels,return_counts=True);single=np.flatnonzero(np.isin(labels,labels_unique[labelcounts==1]))
            for inner in s['inner']:
                a,b=np.array(inner['reference_indices']),np.array(inner['validation_indices'])
                assert set(a).isdisjoint(b) and set(a)|set(b)==set(range(len(ri)))
                assert set(single).issubset(a) and set(single).isdisjoint(b)
                for C,r in inner['by_C'].items():
                    assert float(C)==r['stage']['teacher_C'];stage_audit(raw[a],labels[a],raw[b],labels[b],r['stage'],r['policies'],counts)
                queryids.extend(b.tolist());counts['inner_splits']+=1
            assert sorted(queryids)==sorted(set(range(len(ri)))-set(single))
            options=[]
            for C in (.1,1.,10.):
                for fraction in ('0.5','0.65','0.8','1.0'):
                    values=[r['by_C'][str(C)]['policies'][fraction] for r in s['inner']]
                    options.append({'C':C,'fraction':fraction,'score':float(np.mean([r['clean'] for r in values])),
                                    'retained':float(np.mean([r['retained'] for r in values]))})
            full=min((r for r in options if r['fraction']=='1.0'),key=lambda r:(-r['score'],r['C']))
            family=[r for r in options if r['C']==full['C']];best=max(r['score'] for r in family)
            eligible=[r for r in family if r['score']>=best-.005-1e-12]
            selected=min(eligible,key=lambda r:(r['retained'],-r['score'],float(r['fraction'])))
            assert selected==s['selected_config'] and full==s['full_config'] and selected['C']==full['C']
            assert s['selection']=={'options':options,'eligible':eligible,'tolerance':.005,'hierarchical':True}
            assert set(s['outer_stages'])=={str(full['C'])}
            stage=s['outer_stages'][str(full['C'])];assert stage==s['stage']
            scores=stage_audit(raw,labels,X[vi],y[vi],stage,None,counts,seed=144+i)
            for expected,stored in ((scores[s['chosen']],s['metrics']),(scores['1.0'],s['full_metrics']),(scores['1.0'],s['same_C_full_metrics'])):
                for field in stored:np.testing.assert_allclose(expected[field],stored[field],atol=1e-12)
            counts['outer_splits']+=1
        print(name,'nested90% teachers/primal points/selection verified',flush=True)
    if names is None:
        assert counts['outer_splits']==44 and counts['inner_splits']==128
    else:
        assert counts['outer_splits']>0 and counts['inner_splits']>0
    counts['partial']=names is not None
    counts['datasets']=[row['dataset'] for row in summary['rows']]
    counts['scope']='all_nested_teacher_refit_fuzzy_classwise_primal_and_prediction_selection_replay_not_global_MILP_optimality'
    target='complete-audit.json' if names is None else 'partial-audit.json'
    (FOLDER/target).write_text(json.dumps(counts,indent=2),encoding='utf-8');print(counts)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--completed-only',action='store_true')
    args=parser.parse_args()
    if args.completed_only:
        names=[p.stem for p in sorted(FOLDER.glob('*.json')) if not p.name.startswith(('complete','partial'))]
        main(names)
    else:main()
