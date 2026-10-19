"""Independently rebuild holdout choice, certificates and external scoring."""
import json
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.model_selection import train_test_split
from diagnose_validated_capped_sparse import ROOT,FOLDER,fingerprints
from audit_trajectory_recent import scaling
from certified_frfs.datasets import load_dataset
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.query_probability import QueryProbabilityObjective
from certified_frfs.validated_pool import validation_scores


def verify_fit(raw,labels,fit,counts):
    A,_=scaling(raw,raw);d=raw.shape[1];full=np.ones(d)/np.sqrt(d);mass=full.sum();obj=QueryProbabilityObjective(raw,labels)
    assert obj.splits==fit['training_splits']
    for c in fit['candidates']:
        v=np.array(c['weights']);fs=c['features']
        assert fs==np.flatnonzero(v>0).tolist() and abs(v.sum()-mass)<1e-10 and v.min()>=0 and v.max()<=2*mass/d+1e-10
        assert abs(obj(v/mass)[0]-c['objective'])<1e-10
        cert=weight_certificate(A,labels,fs,full,v[fs]);saved=c['certificate']
        for k in ('feasible','majority_feasible','retained_counts','required'):assert cert[k]==saved[k]
        assert abs(cert['dependency_certificate']['upper_bound']-saved['dependency_certificate']['upper_bound'])<1e-12
        counts['certificates']+=1
    eligible=[c for c in fit['candidates'] if c['certificate']['feasible']]
    exact=min(eligible,key=lambda c:(c['objective'],len(c['features'])))
    compact=min((c for c in eligible if c['objective']<=exact['objective']+.01),key=lambda c:(len(c['features']),c['objective']))
    assert fit['selected']==exact and fit['compact_selected']==compact


def main():
    threadpool_limits(1);counts={'splits':0,'certificates':0,'holdout_scores':0,'outer_scores':0}
    for path in sorted(FOLDER.glob('*.json')):
        if path.name.startswith('complete'):continue
        p=json.loads(path.read_text());name=p['dataset'];assert p['fingerprints']==fingerprints(name)
        old=json.loads((ROOT/'results/long-capped-sparse'/f'{name}.json').read_text());X,y=load_dataset(name)
        for index,s in enumerate(p['splits']):
            original=old['splits'][index];ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices'])
            assert ri.tolist()==original['reference_indices'] and vi.tolist()==original['validation_indices'] and set(ri).isdisjoint(vi)
            raw,labels=X[ri],y[ri];fit=s['fit'];assert fit['seed']==14100
            classes,n=np.unique(labels,return_counts=True);repeat=np.flatnonzero(np.isin(labels,classes[n>1]));single=np.flatnonzero(np.isin(labels,classes[n==1]))
            nc=len(classes[n>1]);size=min(max(nc,int(np.ceil(.3*len(repeat)))),len(repeat)-nc)
            di,qi=train_test_split(repeat,test_size=size,stratify=labels[repeat],random_state=14100);di=np.r_[di,single]
            assert di.tolist()==fit['discovery_indices'] and qi.tolist()==fit['validation_indices']
            assert set(di).isdisjoint(qi) and set(di).union(qi)==set(range(len(ri)))
            A,B=scaling(raw[di],raw[qi]);full=np.ones(p['d'])/np.sqrt(p['d'])
            assert [r['recipe'] for r in fit['trace']]==['full','3','10']
            for row in fit['trace']:
                if row['recipe']=='full':assert row['features']==list(range(p['d'])) and np.array_equal(row['weights'],full)
                else:
                    learned=fit['training_fits'][row['recipe']];verify_fit(raw[di],labels[di],learned['fit'],counts)
                    assert learned['fit']['compact_selected']['weights']==row['weights']
                    assert learned['fit']['compact_selected']['features']==row['features']
                metrics=validation_scores(A,labels[di],B,labels[qi],row['features'],np.array(row['weights']),14101,joint=True)
                assert all(abs(metrics[k]-row['validation'][k])<1e-12 for k in metrics)
                assert abs(row['utility']-(metrics['clean']+metrics['noise01'])/2)<1e-12;counts['holdout_scores']+=2
            chosen=max(range(3),key=lambda j:(fit['trace'][j]['utility'],-len(fit['trace'][j]['features']),-j))
            assert fit['chosen_recipe']==fit['trace'][chosen]['recipe']
            if fit['refit'] is not None:
                verify_fit(raw,labels,fit['refit']['fit'],counts)
                assert fit['selected']==fit['refit']['fit']['compact_selected']
            else:
                assert fit['chosen_recipe']=='full' and np.array_equal(fit['selected']['weights'],full)
                Z,_=scaling(raw,raw);assert weight_certificate(Z,labels,list(range(p['d'])),full,full)==fit['selected']['certificate']
            A,B=scaling(raw,X[vi]);c=fit['selected']
            metrics=validation_scores(A,labels,B,y[vi],c['features'],np.array(c['weights']),144+index,joint=True)
            assert all(abs(metrics[k]-s['metrics'][k])<1e-12 for k in metrics)
            counts['splits']+=1;counts['outer_scores']+=2
        print(name,'validated recipe audit complete',flush=True)
    result={**counts,'scope':'independent_holdout_splits_certificates_objectives_choice_refit_certification_scores_not_full_optimizer_refit'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ROOT/'output/validated-capped-sparse-evidence.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in FOLDER.glob('*.json'):archive.write(path,str(path.relative_to(ROOT)))
        for path in (ROOT/'src/certified_frfs').glob('*.py'):archive.write(path,str(path.relative_to(ROOT)))
        for rel in ('scripts/diagnose_validated_capped_sparse.py','scripts/audit_validated_capped_sparse.py','research/validated-capped-sparse-protocol.md','tests/test_validated_capped_sparse.py'):
            archive.write(ROOT/rel,rel)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
