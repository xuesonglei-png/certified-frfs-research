"""Hierarchical inner choice; no new test-informed configurations."""
import hashlib,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from audit_nested_teacher_support import independent_score
from review_box_iterations import paired

FOLDER=ROOT/'results/hierarchical-teacher-selection'

def main():
    threadpool_limits(1);FOLDER.mkdir(exist_ok=True)
    prior=ROOT/'results/nested-teacher-regularized';summary=json.loads((prior/'complete.json').read_text())
    audit=json.loads((prior/'complete-audit.json').read_text());assert audit['outer_splits']==44
    rows=[];counts={'splits':0,'prediction_replays':0}
    for row in summary['rows']:
        name=row['dataset'];source=prior/f'{name}.json';target=FOLDER/f'{name}.json'
        if target.exists():raise FileExistsError(target)
        p=json.loads(source.read_text());X,y=load_dataset(name);records=[]
        for i,s in enumerate(p['splits']):
            fullC=min((.1,1.,10.),key=lambda C:(-np.mean([r['by_C'][str(C)]['policies']['1.0']['clean'] for r in s['inner']]),C))
            assert fullC==s['full_config']['C']
            options=[]
            for fraction in ('0.5','0.65','0.8','1.0'):
                inner=[r['by_C'][str(fullC)]['policies'][fraction] for r in s['inner']]
                options.append({'fraction':fraction,'score':float(np.mean([r['clean'] for r in inner])),
                                'retained':float(np.mean([r['retained'] for r in inner]))})
            eligible=[r for r in options if r['score']>=max(z['score'] for z in options)-.005-1e-12]
            chosen=min(eligible,key=lambda r:(r['retained'],-r['score'],float(r['fraction'])))
            stage=s['outer_stages'][str(fullC)];selected=stage['policies'][chosen['fraction']]
            ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);_,B=preprocess_fit(X[ri],X[vi]);rng=np.random.default_rng(144+i)
            queries=[B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)];values={}
            for key,fit in (('selected',selected['fit']),('full',stage['policies']['1.0']['fit'])):
                scores=[independent_score(Q,y[vi],np.array(stage['teacher_classes']),fit) for Q in queries]
                values[key]={'clean':scores[0],'query_noise01':float(np.mean(scores[1:]))};counts['prediction_replays']+=4
            np.testing.assert_allclose(values['full']['clean'],s['full_metrics']['clean'],atol=1e-12)
            records.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'teacher_C':fullC,
                'chosen':chosen['fraction'],'inner_options':options,'eligible':eligible,'selected':selected,
                'metrics':values['selected'],'full_metrics':values['full'],'removed':1-len(selected['candidate']['features'])/p['d']})
            counts['splits']+=1
        files=[Path(__file__),ROOT/'research/hierarchical-teacher-protocol.md',source,prior/'complete-audit.json',ROOT/'data/processed'/f'{name}.npz']
        target.write_text(json.dumps({'dataset':name,'splits':records,'fingerprints':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}},indent=2),encoding='utf-8')
        rows.append({'dataset':name,'clean':float(np.mean([r['metrics']['clean'] for r in records])),
            'full':float(np.mean([r['full_metrics']['clean'] for r in records])),
            'noise':float(np.mean([r['metrics']['query_noise01'] for r in records])),
            'full_noise':float(np.mean([r['full_metrics']['query_noise01'] for r in records])),
            'removed':float(np.mean([r['removed'] for r in records])),
            'fallbacks':sum(r['selected']['fallback'] for r in records),'historical':row['historical']})
    result={'rows':rows,'mean_clean':float(np.mean([r['clean'] for r in rows])),'mean_removed':float(np.mean([r['removed'] for r in rows])),
        'vs_identical_teacher_full':paired([r['clean']-r['full'] for r in rows]),
        'vs_identical_teacher_full_noise':paired([r['noise']-r['full_noise'] for r in rows]),
        'vs_historical':{m:paired([r['clean']-r['historical'][m] for r in rows]) for m in ('full','fgs2023','sfrip2025')},
        'prediction_replay':counts,'scope':'hierarchical_inner_selector_replay_previously_audited_independent_stages_exposed_development_not_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print({k:v for k,v in result.items() if k!='rows'})

if __name__=='__main__':main()
