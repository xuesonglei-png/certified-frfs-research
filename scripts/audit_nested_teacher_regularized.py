"""Independent audit of joint inner C/support choice and matched-full C choice."""
import hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from audit_nested_teacher_support import audit_stage,independent_score

FOLDER=ROOT/'results/nested-teacher-regularized'

def outer_scores(raw,query,truth,stage,seed):
    _,B=preprocess_fit(raw,query);rng=np.random.default_rng(seed)
    queries=[B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
    out={}
    for k,r in stage['policies'].items():
        values=[independent_score(Q,truth,np.array(stage['teacher_classes']),r['fit']) for Q in queries]
        out[k]={'clean':values[0],'query_noise01':float(np.mean(values[1:])),'retained':len(r['candidate']['features'])/stage['d']}
    return out

def main():
    threadpool_limits(1);summary=json.loads((FOLDER/'complete.json').read_text())
    counts={'outer_splits':0,'inner_splits':0,'stages':0,'unique_models':0,'predictions':0,'fallback_policies':0}
    for row in summary['rows']:
        name=row['dataset'];p=json.loads((FOLDER/f'{name}.json').read_text());X,y=load_dataset(name)
        for rel,digest in p['fingerprints'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
        for i,s in enumerate(p['splits']):
            ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);assert set(ri).isdisjoint(vi)
            raw,labels=X[ri],y[ri];query_ids=[]
            classes,classcounts=np.unique(labels,return_counts=True);single=np.flatnonzero(np.isin(labels,classes[classcounts==1]))
            for inner in s['inner']:
                a,b=np.array(inner['reference_indices']),np.array(inner['validation_indices'])
                assert set(a).isdisjoint(b) and set(a)|set(b)==set(range(len(ri)))
                assert set(single).issubset(a) and not set(single)&set(b)
                assert set(ri[a]).isdisjoint(vi) and set(ri[b]).isdisjoint(vi)
                assert set(inner['by_C'])=={'0.1','1.0','10.0'}
                for C,r in inner['by_C'].items():
                    assert float(C)==r['stage']['teacher_C']
                    audit_stage(raw[a],labels[a],raw[b],labels[b],r['stage'],r['policies'],counts)
                query_ids.extend(b.tolist());counts['inner_splits']+=1
            assert sorted(query_ids)==sorted(set(range(len(ri)))-set(single))
            options=[]
            for C in (.1,1.,10.):
                for fraction in ('0.5','0.65','0.8','1.0'):
                    scores=[r['by_C'][str(C)]['policies'][fraction] for r in s['inner']]
                    options.append({'C':C,'fraction':fraction,'score':float(np.mean([r['clean'] for r in scores])),
                                    'retained':float(np.mean([r['retained'] for r in scores]))})
            eligible=[r for r in options if r['score']>=max(z['score'] for z in options)-.005-1e-12]
            proposed=min(eligible,key=lambda r:(r['retained'],-r['score'],r['C'],float(r['fraction'])))
            full=min((r for r in options if r['fraction']=='1.0'),key=lambda r:(-r['score'],r['C']))
            assert proposed==s['selected_config'] and full==s['full_config']
            assert s['selection']=={'options':options,'eligible':eligible,'tolerance':.005}
            byC={}
            for C,stage in s['outer_stages'].items():
                assert float(C)==stage['teacher_C']
                scores=outer_scores(raw,X[vi],y[vi],stage,144+i)
                audit_stage(raw,labels,X[vi],y[vi],stage,scores,counts,seed=144+i);byC[C]=scores
            expected=[byC[str(proposed['C'])][proposed['fraction']],byC[str(full['C'])]['1.0'],byC[str(proposed['C'])]['1.0']]
            for scores,stored in zip(expected,[s['metrics'],s['full_metrics'],s['same_C_full_metrics']]):
                for field,value in stored.items():np.testing.assert_allclose(scores[field],value,atol=1e-12)
            assert s['stage']==s['outer_stages'][str(proposed['C'])]
            counts['outer_splits']+=1
        print(name,'C/support selection and refitted stages audited',flush=True)
    assert counts['outer_splits']==44
    counts['scope']='all_nested_teacher_refits_certificates_and_selection_prediction_replay_not_LP_optimality_replay'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(counts)

if __name__=='__main__':main()
