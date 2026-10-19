"""Four-arm first-panel inner-only restoration; automatic audited-backend gate."""
import argparse,hashlib,json,time
from pathlib import Path
from collections import Counter
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.relation_support_selector_v1 import _candidate
from certified_frfs.selective_support_selector_v3 import query_scores
from graph_certified_restoration_v2 import FAMILIES,FRACTIONS,MODES,GuardCache,restored,summarize
from run_repair_relation_selector_v4 import ROOT,OUT,NAMES
from audit_selective_support_selector_v3 import check_candidate,check_score
from audit_relation_support_selector_v1 import equivalent

FOLDER=ROOT/'results/graph-certified-restoration-inner-v2'
AUDIT_REL='results/shared-graph-dependency-v4/endpoint-audit.json'


def save(path,value):
    path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,separators=(',',':')));tmp.replace(path)


def readiness():
    # No user-consent step: the scientifically required independent audit is
    # the only promotion prerequisite. Absence/failure never becomes approval.
    p=ROOT/AUDIT_REL
    if not p.exists():raise RuntimeError('Do not fit: independent graph-v4 endpoint audit is absent')
    audit=json.loads(p.read_text());assert audit.get('passed') is True,'graph-v4 independent audit must explicitly pass'
    current=hashlib.sha256((ROOT/'scripts/shared_graph_dependency_v4.py').read_bytes()).hexdigest()
    hashes=audit.get('source_hashes',audit.get('hashes',{}))
    recorded=hashes.get('scripts/shared_graph_dependency_v4.py',hashes.get('scripts\\shared_graph_dependency_v4.py'))
    assert recorded==current,'audit must match actual imported graph-v4 backend'
    for rel,want in hashes.items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==want,rel
    return p


def freeze():
    audit=readiness();files=list((ROOT/'src/certified_frfs').glob('*.py'))
    files += [ROOT/'scripts'/s for s in ('graph_certified_restoration_v2.py','run_graph_certified_restoration_inner_v2.py',
      'shared_graph_dependency_v4.py','signed_classwise_dependency.py','classwise_star_dependency.py',
      'audit_selective_support_selector_v3.py','audit_relation_support_selector_v1.py','audit_majority_bound_prediction_v1.py',
      'run_repair_relation_selector_v4.py')]
    files += [ROOT/'tests/test_graph_certified_restoration_v2.py',
              ROOT/'research/graph-certified-restoration-inner-v2-protocol.md',audit]
    for name in NAMES:files += [OUT/(name+'.json'),ROOT/'results/repair-relation-controls-v4'/(name+'.json'),ROOT/'data/processed'/(name+'.npz')]
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    path=FOLDER/'source-freeze.json'
    if path.exists():assert json.loads(path.read_text())['hashes']==hashes
    else:save(path,{'hashes':hashes,'created_unix':time.time(),'new_outer_scoring':False})
    return hashes


def inherited(part,d):
    """Recover ordinary family prefixes, distinguish V4 uniform fallback."""
    bycount={};fallback=None
    for recipe in part['candidates']:
        for i,c in enumerate(recipe['trace']):
            isfallback=recipe['model']['uniform_fallback'] and i==len(recipe['trace'])-1
            if isfallback:
                if fallback is not None:equivalent(c,fallback)
                fallback=c;continue
            count=len(c['features'])
            assert c['features']==sorted(part['ranking'][:count])
            if count in bycount:equivalent(c,bycount[count])
            bycount[count]=c
    return bycount,fallback


def full_baselines(blocks):
    # Full native grid tuning only, no outer score fields and no reduction of
    # authors' documented candidate grids to one favorable old option.
    out={}
    for method in ('fgs_native_uniform','fgs_native_capped_transfer','fgs_native_local_soft',
                   'sfrip_native_uniform','sfrip_native_capped_transfer','sfrip_native_local_soft'):
        n=len(blocks[0]['options'][method]);assert n==(120 if method.startswith('fgs') else 6)
        options=[]
        for i in range(n):
            cc=[b['options'][method][i] for b in blocks]
            if any(r['score'] is None for r in cc):
                options.append({'config_index':i,'available':False});continue
            options.append({'config_index':i,'available':True,
                'mean_BA':float(np.mean([r['score']['clean_BA'] for r in cc])),
                'mean_logloss':float(np.mean([r['score']['balanced_logloss'] for r in cc])),
                'mean_count':float(np.mean([len(r['metric']['features']) for r in cc]))})
        valid=[r for r in options if r['available']]
        pick=min(valid,key=lambda r:(-round(r['mean_BA'],12),round(r['mean_logloss'],12),r['mean_count'],r['config_index'])) if valid else None
        out[method]={'all_options':options,'chosen':pick,'full_grid_size':n,
                     'scope':'cached_complete_native_grid_mechanism_reference_not_confirmation'}
    return out


def run(name,hashes):
    path=FOLDER/(name+'.json')
    saved=json.loads(path.read_text()) if path.exists() else None
    if saved:
        assert saved['source_hashes']==hashes
        if saved['complete']:return saved
    source=json.loads((OUT/(name+'.json')).read_text());control=json.loads((ROOT/'results/repair-relation-controls-v4'/(name+'.json')).read_text())
    panel=source['rows'][0];basepanel=control['rows'][0]
    assert panel['train_indices']==basepanel['train_indices'];ids=np.asarray(panel['train_indices']);X,y=load_dataset(name)
    raw,labels=X[ids],y[ids];blocks=saved['blocks'] if saved else [];previous=saved['seconds'] if saved else 0.;start=time.perf_counter()
    for index,(oldblock,baseline) in enumerate(zip(panel['fit']['inner'],basepanel['inner'],strict=True)):
        if index<len(blocks):continue
        ii,jj=np.asarray(oldblock['fit_indices']),np.asarray(oldblock['validation_indices'])
        assert ii.tolist()==baseline['fit_indices'] and jj.tolist()==baseline['validation_indices']
        A,B=preprocess_fit(raw[ii],raw[jj]);d=A.shape[1];counts=Counter();families={};scored={};freshchecks=set()
        statepath=FOLDER/name/(str(index)+'-class-cache.json')
        state=json.loads(statepath.read_text()) if statepath.exists() else None
        if state is not None:assert state['source_hashes']==hashes
        def checkpoint(value):save(statepath,{'source_hashes':hashes,'state':value})
        guard=GuardCache(A,labels[ii],state=None if state is None else state['state'],checkpoint=checkpoint)
        uniform=np.ones(d)/np.sqrt(d)
        uniform_part=oldblock['families']['uniform_sqrt'];uniformfull=uniform_part['candidates'][-1]['model']
        assert uniformfull['features']==list(range(d)) and uniformfull['joint_feasible']
        for family in FAMILIES:
            part=oldblock['families'][family];ref=np.asarray(part['reference_weights']);rank=part['ranking']
            bycount,fallback=inherited(part,d);modes={}
            def provide(count):
                if count not in bycount:
                    # Normally all required prefixes occur in the inherited
                    # earliest-fraction trace. Preserve an explicit exception.
                    bycount[count]=_candidate(A,labels[ii],ref,sorted(rank[:count]),count)
                return bycount[count]
            for mode in MODES:
                recipes=[]
                for recipe_index,fraction in enumerate(FRACTIONS):
                    requested=max(int(np.ceil(d/2)),int(np.ceil(fraction*d)))
                    row=restored(A,labels[ii],ref,rank,requested,mode,provide,lambda:uniformfull,guard)
                    if mode=='legacy':
                        old=part['candidates'][recipe_index]['model']
                        for k in ('features','weights','requested_count','restored_count','uniform_fallback'):equivalent(row[k] if k not in ('features','weights') else row['model'][k],old[k])
                    for step in row['trace']:
                        c=step['candidate'];witness=uniform if step.get('explicit_uniform_fallback') else ref
                        key=(tuple(c['features']),tuple(c['weights']),tuple(witness))
                        if key not in freshchecks:check_candidate(A,labels[ii],witness,c['features'],c,counts);freshchecks.add(key)
                    c=row['model'];key=(tuple(c['features']),tuple(c['weights']))
                    if key not in scored:
                        scored[key]=query_scores(A,labels[ii],B,labels[jj],c)
                        check_score(A,labels[ii],B,labels[jj],c,scored[key],counts)
                    row['score']=scored[key];row['recipe_index']=recipe_index;row['fraction']=fraction
                    recipes.append(row)
                modes[mode]=recipes
            for i in range(4):
                assert len(modes['graph_signed'][i]['model']['features'])<=len(modes['signed'][i]['model']['features'])<=len(modes['legacy'][i]['model']['features'])
                assert len(modes['graph_positive'][i]['model']['features'])<=len(modes['legacy'][i]['model']['features'])
                # Limited independent target solves need not preserve an order
                # between graph-positive and graph-signed realized endpoints.
            families[family]={'reference_weights':ref.tolist(),'ranking':rank,'modes':modes}
        blocks.append({'inner_index':index,'fit_indices':ii.tolist(),'validation_indices':jj.tolist(),
            'families':families,'guard_counters':guard.counters,'independent_check_counts':dict(counts),
            'class_cache_artifact':statepath.relative_to(ROOT).as_posix()})
        assert freeze()==hashes
        result={'dataset':name,'d':d,'blocks':blocks,'source_hashes':hashes,'complete':False,
                'seconds':previous+time.perf_counter()-start,'new_outer_scoring':False}
        save(path,result);print(name,'inner',index,'complete',guard.counters,dict(counts),flush=True)
    summaries={mode:summarize(blocks,mode) for mode in MODES}
    result={'dataset':name,'d':raw.shape[1],'blocks':blocks,'summaries':summaries,
            'baseline_full_grids':full_baselines(basepanel['inner']),'source_hashes':hashes,'complete':True,
            'seconds':previous+time.perf_counter()-start,'new_outer_scoring':False,
            'scope':'cached_reference_first_panel_inner_development_not_full_method_refit_or_confirmation'}
    assert freeze()==hashes;save(path,result);return result


def aggregate(rows):
    out=[]
    for r in rows:
        policies={}
        for policy in ('clean_first','compression_policy'):
            parts={mode:r['summaries'][mode][policy] for mode in MODES}
            contrasts={}
            for newer,older in (('graph_positive','legacy'),('graph_signed','signed'),('signed','legacy'),('graph_signed','legacy')):
                a,b=parts[newer],parts[older]
                contrasts[newer+'_vs_'+older]={'clean_BA_delta':a['mean_BA']-b['mean_BA'],
                    'certified_BA_delta':a['mean_certified_BA']-b['mean_certified_BA'],
                    'extra_removal':(b['mean_count']-a['mean_count'])/r['d']}
            policies[policy]={'choices':parts,'contrasts':contrasts}
        out.append({'dataset':r['dataset'],'policies':policies})
    gates={}
    for policy in ('clean_first','compression_policy'):
        primary=[r['policies'][policy]['contrasts']['graph_positive_vs_legacy'] for r in out]
        removal=float(np.mean([r['extra_removal'] for r in primary]));clean=float(np.mean([r['clean_BA_delta'] for r in primary]));cert=float(np.mean([r['certified_BA_delta'] for r in primary]));improved=sum(r['extra_removal']>1e-12 for r in primary)
        gates[policy]={'mean_extra_removal':removal,'mean_clean_BA_delta':clean,'mean_certified_BA_delta':cert,
            'datasets_more_compact':improved,'resource_gate':len(rows)==6 and removal>=.02 and improved>=2 and clean>=-.005 and cert>=-.005}
    return {'complete':len(rows)==6,'datasets':len(rows),'rows':out,'primary_mean_extra_removal':removal,
        'primary_policy':'clean_first','policy_gate_results':gates,
        'predeclared_resource_gate':gates['clean_first']['resource_gate'],
        'new_outer_scoring':False,'predictive_superiority_established':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--only',choices=NAMES);args=p.parse_args()
    threadpool_limits(1);hashes=freeze();rows=[run(n,hashes) for n in ((args.only,) if args.only else NAMES)]
    if len(rows)==6:save(FOLDER/'summary.json',aggregate(rows))


if __name__=='__main__':main()
