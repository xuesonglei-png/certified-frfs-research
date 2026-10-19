"""Recheck frozen new-data split integrity, certificates and outer scoring."""
import json
import zipfile
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from run_newdata_compact import ROOT, FOLDER, DATA, SOURCES, sources, GRIDS, SELECTORS, score, utility
from audit_trajectory_recent import scaling
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.query_probability import QueryProbabilityObjective
from certified_frfs.validated_pool import validation_scores

ARCHIVE = ROOT/'output/newdata-compact-evidence.zip'

def main():
    threadpool_limits(1); counts={'splits':0,'certificate_checks':0,'outer_scores':0,'baseline_final_refits':0}
    for name in SOURCES:
        p=json.loads((FOLDER/f'{name}.json').read_text()); assert p['fingerprints']==sources(name)
        with np.load(DATA/f'{name}.npz') as data:X,y=data['X'],data['y']
        expected=list(StratifiedKFold(3,shuffle=True,random_state=13101).split(X,y))
        full=np.ones(X.shape[1])/np.sqrt(X.shape[1]);mass=full.sum()
        for fold,s in enumerate(p['splits']):
            tr,te=np.array(s['train_indices']),np.array(s['test_indices'])
            assert np.array_equal(tr,expected[fold][0]) and np.array_equal(te,expected[fold][1])
            assert set(tr).isdisjoint(te) and set(tr).union(te)==set(range(len(y)))
            A,B=scaling(X[tr],X[te]);obj=QueryProbabilityObjective(X[tr],y[tr]);fit=s['fit']
            assert fit['training_splits']==obj.splits
            for c in fit['candidates']:
                v=np.array(c['weights']);fs=c['features']
                assert abs(v.sum()-mass)<1e-10 and v.min()>=0 and v.max()<=2*mass/X.shape[1]+1e-10
                assert fs==np.flatnonzero(v>0).tolist() and abs(obj(v/mass)[0]-c['objective'])<1e-10
                cert=weight_certificate(A,y[tr],fs,full,v[fs]);saved=c['certificate']
                for k in ('feasible','majority_feasible','retained_counts','required'):assert cert[k]==saved[k]
                assert abs(cert['dependency_certificate']['upper_bound']-saved['dependency_certificate']['upper_bound'])<1e-12
                counts['certificate_checks']+=1
            eligible=[c for c in fit['candidates'] if c['certificate']['feasible']]
            exact=min(eligible,key=lambda c:(c['objective'],len(c['features'])))
            compact=min((c for c in eligible if c['objective']<=exact['objective']+.01),key=lambda c:(len(c['features']),c['objective']))
            assert fit['selected']==exact and fit['compact_selected']==compact
            for key,fs,w in [('proposed_metrics',compact['features'],np.array(compact['weights'])),('full_metrics',list(range(X.shape[1])),full)]:
                metrics=validation_scores(A,y[tr],B,y[te],fs,w,144+fold,joint=True)
                assert all(abs(metrics[k]-s[key][k])<1e-12 for k in metrics);counts['outer_scores']+=2
            for method,grid in GRIDS.items():
                saved=s['baselines'][method];assert [c['config'] for c in saved['trace']]==grid
                for c in saved['trace']:
                    assert abs(c['utility']-np.mean([utility(f['metrics']) for f in c['folds']]))<1e-12
                    assert abs(c['mean_size']-np.mean([len(f['features']) for f in c['folds']]))<1e-12
                chosen=max(range(len(grid)),key=lambda j:(saved['trace'][j]['utility'],-saved['trace'][j]['mean_size'],-j))
                assert chosen==saved['chosen_index']
                selection=SELECTORS[method](A,y[tr],**grid[chosen]);assert json.loads(json.dumps(selection))==saved['selection']
                metrics=score(A,y[tr],B,y[te],selection['features'],144+fold)
                assert all(abs(metrics[k]-saved['metrics'][k])<1e-12 for k in metrics)
                counts['baseline_final_refits']+=1;counts['outer_scores']+=2
            counts['splits']+=1
        print(name,'new-data audit complete',flush=True)
    result={**counts,'scope':'certificate_selection_split_outer_score_final_baseline_rebuild_not_inner_optimizer_refit_or_author_fidelity'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ARCHIVE,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in (FOLDER,DATA):
            for path in folder.iterdir():
                if path.is_file():archive.write(path,str(path.relative_to(ROOT)))
        for path in (ROOT/'src/certified_frfs').glob('*.py'):archive.write(path,str(path.relative_to(ROOT)))
        for rel in ('scripts/run_newdata_compact.py','scripts/audit_newdata_compact.py','scripts/compare_trajectory_recent.py','research/newdata-compact-protocol.md'):
            archive.write(ROOT/rel,rel)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
