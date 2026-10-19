import json,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_certified_surrogate_alignment import ROOT,FOLDER
from audit_trajectory_recent import scaling
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import validation_scores


def main():
    threadpool_limits(1);counts={'splits':0,'nodes':0,'fold_score_values':0,'updates':0,'vote_improved':0,'vote_decreased':0,'vote_unchanged':0}
    for path in sorted(FOLDER.glob('*.json')):
        if path.name.startswith('complete'):continue
        p=json.loads(path.read_text());name=p['dataset'];prior_path=ROOT/'results/certified-query-refinement'/path.name
        assert hashlib.sha256(prior_path.read_bytes()).hexdigest()==p['prior_sha256']
        assert hashlib.sha256((ROOT/'scripts/diagnose_certified_surrogate_alignment.py').read_bytes()).hexdigest()==p['script_sha256']
        old=json.loads(prior_path.read_text());X,y=load_dataset(name)
        for s,saved in zip(old['splits'],p['splits']):
            ri=np.array(s['reference_indices']);raw,labels=X[ri],y[ri]
            assert len(saved['nodes'])==len(s['fit']['trace'])
            for node,record in zip(s['fit']['trace'],saved['nodes']):
                scores=[];w=np.array(node['weights']);fs=np.flatnonzero(w>0).tolist()
                for inner,fold in enumerate(s['fit']['training_splits']):
                    ti,qi=np.array(fold['reference']),np.array(fold['query'])
                    assert set(ti).isdisjoint(qi) and set(ti).union(qi)==set(range(len(ri)))
                    A,B=scaling(raw[ti],raw[qi]);scores.append(validation_scores(A,labels[ti],B,labels[qi],fs,w,31001+inner,joint=True))
                    counts['fold_score_values']+=2
                values={k:float(np.mean([m[k] for m in scores])) for k in ('clean','noise01')}
                assert all(abs(values[k]-record['adaptive_training_metrics'][k])<1e-12 for k in values)
                assert record['surrogate']==node['objective'] and record['iteration']==node['iteration']
                assert abs(record['utility']-(values['clean']+values['noise01'])/2)<1e-12
                counts['nodes']+=1
            expected=[{'iteration':b['iteration'],'surrogate_decrease':a['surrogate']-b['surrogate'],'vote_change':b['utility']-a['utility']}
                for a,b in zip(saved['nodes'],saved['nodes'][1:])]
            assert expected==saved['changes']
            for c in expected:
                assert c['surrogate_decrease']>0;counts['updates']+=1
                counts['vote_improved']+=c['vote_change']>1e-12;counts['vote_decreased']+=c['vote_change'] < -1e-12
                counts['vote_unchanged']+=abs(c['vote_change'])<=1e-12
            counts['splits']+=1
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    print(json.dumps(counts,indent=2),flush=True)


if __name__=='__main__':main()
