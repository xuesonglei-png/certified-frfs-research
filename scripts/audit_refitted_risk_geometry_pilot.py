"""Independent actual-fold refits, all feasible risk probes and path replay."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import mutual_info_score
from threadpoolctl import threadpool_limits
from audit_nested_geometry_forest import (ROOT,membership,score,BA,check_model,equivalent,load_dataset,
    preprocess_fit,interaction_tree,weight_certificate,transfer_weights,capped_sparse_projection,paired)
from certified_frfs.nested_teacher_regularized import inner_splits
from certified_frfs.majority_metric import MajorityTerms

SOURCE = ROOT/'results/nested-geometry-forest'
FOLDER = ROOT/'results/refitted-risk-geometry-pilot'
METHODS = ('old','refitted','matched','full')


def edge_workspace(A,y):
    bins = membership(A).argmax(axis=2); d = A.shape[1]
    single = [mutual_info_score(bins[:,a],y) for a in range(d)]
    edges = []
    for u in range(d):
        for v in range(u+1,d):
            value = max(0.,float(mutual_info_score(3*bins[:,u]+bins[:,v],y))-max(single[u],single[v]))
            edges.append((value,u,v))
    return sorted(edges,key=lambda v:(-v[0],v[1],v[2]))


def subset_edges(workspace,fs):
    keep = set(fs); components = [{a} for a in fs]; result = []
    for _,u,v in workspace:
        if u not in keep or v not in keep: continue
        cu = next(c for c in components if u in c); cv = next(c for c in components if v in c)
        if cu is not cv:
            combined = cu|cv; components.remove(cu); components.remove(cv); components.append(combined); result.append([u,v])
        if len(components) == 1: break
    return result


def audit(name):
    threadpool_limits(1); counts = Counter(); artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text())
    old_path = SOURCE/(name+'.json'); source = json.loads(old_path.read_text()); assert saved['complete'] and source['complete']
    for rel,value in saved['fingerprints'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == value
    assert json.loads((SOURCE/'complete-audit.json').read_text())['passed']
    old_audit = json.loads((SOURCE/(name+'-audit.json')).read_text())
    assert old_audit['artifact_sha256'] == hashlib.sha256(old_path.read_bytes()).hexdigest()
    assert old_audit['auditor_sha256'] == hashlib.sha256((ROOT/'scripts/audit_nested_geometry_forest.py').read_bytes()).hexdigest()
    hashes = {'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    checkpoint = FOLDER/(name+'-audit.json')
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        if previous['fingerprints'] == hashes: return previous['result']
    X,y = load_dataset(name); expected = [(i,j,s,inner) for i,s in enumerate(source['splits']) for j,inner in enumerate(s['inner'])]
    assert len(saved['records']) == len(expected)
    for r,(i,j,s,inner) in zip(saved['records'],expected):
        assert r['outer_index'] == i and r['inner_index'] == j and r['outer_reference_indices'] == s['reference_indices']
        for k in ('reference_indices','validation_indices'): assert r[k] == inner[k]
        ri,a,b = (np.array(r[k]) for k in ('outer_reference_indices','reference_indices','validation_indices'))
        assert set(a).isdisjoint(b) and len(set(a)|set(b)) == len(ri)
        raw,labels = X[ri[a]],y[ri[a]]; A,B = preprocess_fit(raw,X[ri[b]]); d = A.shape[1]
        stage = r['stage']; path = stage['path']; risk = stage['risk']; old = inner['stages'][7]
        assert stage['C'] == stage['pair_penalty'] == 1. and stage['fraction'] == .8 and not stage['teacher_guards_enforced']
        assert path['tie_decimals'] == 12 and risk['seed'] == 20261014
        splits = inner_splits(labels,seed=20261014); assert len(splits) == len(risk['folds'])
        fold_data = []
        for fold,(c,e) in zip(risk['folds'],splits):
            assert fold['reference_indices'] == c.tolist() and fold['validation_indices'] == e.tolist()
            assert set(c).isdisjoint(e) and len(set(c)|set(e)) == len(labels)
            T,Q = preprocess_fit(raw[c],raw[e]); fold_data.append((T,Q,labels[c],labels[e],edge_workspace(T,labels[c])))
        fit_scores = []
        for record in risk['models']:
            T,Q,train,truth,workspace = fold_data[record['fold_index']]; fit = record['fit']
            assert fit['C'] == fit['pair_penalty'] == 1. and fit['edges'] == subset_edges(workspace,fit['features'])
            check_model(T,train,fit,Counter()); scores = score(Q,fit); classes = np.array(fit['classes'])
            own = np.argmax(truth[:,None] == classes[None],axis=1)
            balance = np.array([1/(len(np.unique(truth))*sum(truth == c)) for c in truth])
            value = float(balance @ (logsumexp(scores,axis=1)-scores[np.arange(len(truth)),own]))
            np.testing.assert_allclose(record['balanced_logloss'],value,atol=1e-12)
            fit_scores.append(value); counts['independent_fold_model_refits'] += 1; counts['risk_models'] += 1
            counts['risk_model_caps'] += not fit['optimizer_converged']
        used_models = []; support_scores = {}
        for sid,record in enumerate(risk['scores']):
            fs = tuple(record['features']); assert fs not in support_scores
            support_scores[fs] = sid
            assert len(record['fold_model_ids']) == len(fold_data)
            values = []
            for fj,mid in enumerate(record['fold_model_ids']):
                model = risk['models'][mid]; assert model['fold_index'] == fj and model['fit']['features'] == list(fs)
                used_models.append(mid); values.append(fit_scores[mid])
            np.testing.assert_allclose(record['mean_balanced_logloss'],np.mean(values),atol=1e-12); counts['support_risk_scores'] += 1
        assert sorted(used_models) == list(range(len(risk['models'])))
        original = np.ones(d)/np.sqrt(d); protected = MajorityTerms(A,labels,weights=original).bounds(range(d),allowed_enemies=2) > 1e-10
        verified = {tuple(g['features']):g for g in inner['geometry']}; geometries = {}
        def candidate(record,fs):
            assert record['features'] == fs; key = tuple(fs)
            if key not in geometries:
                if key in verified: geometries[key] = verified[key]; counts['reused_verified_geometries'] += 1
                else:
                    w = original if len(fs) == d else capped_sparse_projection(transfer_weights(A,fs,original)/np.sqrt(d),len(fs),2/d)*np.sqrt(d)
                    embedded = np.zeros(d); embedded[fs] = w
                    geometries[key] = {'features':fs,'weights':embedded.tolist(),'certificate':weight_certificate(A,labels,fs,original,w,protected=protected)}
                    counts['new_geometry_replays'] += 1
            expected = geometries[key]; np.testing.assert_allclose(record['weights'],expected['weights'],atol=1e-12); equivalent(record['certificate'],expected['certificate'])
            sid = record['risk_score_id']
            if sid is not None: assert risk['scores'][sid]['features'] == fs
            return expected['certificate']['feasible']
        active = list(range(d)); ai = ti = 0; assert candidate(path['trace'][0],active)
        assert path['trace'][0]['risk_score_id'] == support_scores[tuple(active)]
        scored_supports = {tuple(active)}; minimum = int(np.ceil(.8*d)); accepted = False
        for ranking in path['rankings']:
            assert ranking['features'] == active and len(active) > minimum
            feasible = []
            for removed in active:
                fs = [a for a in active if a != removed]; attempt = path['attempts'][ai]
                assert attempt['removed'] == removed
                valid = candidate(attempt,fs)
                if valid:
                    assert attempt['risk_score_id'] is not None
                    scored = risk['scores'][attempt['risk_score_id']]; feasible.append((scored['mean_balanced_logloss'],removed,ai))
                    scored_supports.add(tuple(fs))
                else: assert attempt['risk_score_id'] is None and not attempt['accepted']
                counts['attempts'] += 1; counts['geometry_rejections'] += not valid; ai += 1
            ordered = sorted(feasible,key=lambda v:(round(v[0],12),v[1]))
            assert [(v['removed'],v['attempt_id']) for v in ranking['choices']] == [(removed,attempt_id) for _,removed,attempt_id in ordered]
            np.testing.assert_allclose([v['loss'] for v in ranking['choices']],[value for value,_,_ in ordered],atol=1e-12)
            for _,_,attempt_id in feasible: assert path['attempts'][attempt_id]['accepted'] == (attempt_id == ordered[0][2])
            accepted = bool(ordered)
            if not accepted: assert ranking is path['rankings'][-1]; break
            attempt = path['attempts'][ordered[0][2]]; ti += 1
            equivalent(path['trace'][ti],{k:v for k,v in attempt.items() if k not in ('removed','accepted')}); active = attempt['features']
        assert ai == len(path['attempts']) and ti+1 == len(path['trace']) and (len(active) == minimum or not accepted)
        assert scored_supports == set(support_scores)
        equivalent(stage['candidate'],path['trace'][-1]); assert stage['fit']['features'] == active
        assert r['fits']['refitted'] == stage['fit']
        assert r['fits']['old'] == inner['models'][old['policies']['0.8']['model_id']]
        assert r['fits']['full'] == inner['models'][old['policies']['1.0']['model_id']]
        matched = next(v for v in reversed(path['trace']) if len(v['features']) >= len(r['fits']['old']['features']))
        assert r['fits']['matched']['features'] == matched['features']
        assert r['matched_count_exact'] == (len(matched['features']) == len(r['fits']['old']['features']))
        counts['matched_count_mismatches'] += not r['matched_count_exact']; counts['unchanged_supports'] += r['fits']['old']['features'] == active
        rng = np.random.default_rng(144+i); assert r['query_seed'] == 144+i
        queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        for m,fit in r['fits'].items():
            assert fit['C'] == fit['pair_penalty'] == 1.
            fs = fit['features']; assert fit['edges'] == [sorted([fs[u],fs[v]]) for u,v in interaction_tree(A[:,fs],labels)['edges']]
            check_model(A,labels,fit,Counter()); values = [BA(score(Q,fit),fit['classes'],y[ri[b]]) for Q in queries]
            equivalent(r['metrics'][m],{'clean':values[0],'noise':float(np.mean(values[1:])),'removed':1-len(fs)/d})
            counts['query_predictions'] += 4; counts['final_fit_caps'] += not fit['optimizer_converged']
        counts['records'] += 1; print(name,i,j,'actual refit audit passed',flush=True)
    result = {'dataset':name,'counts':dict(counts),'methods':{m:{metric:float(np.mean([r['metrics'][m][metric] for r in saved['records']])) for metric in ('clean','noise','removed')} for m in METHODS}}
    checkpoint.write_text(json.dumps({'fingerprints':hashes,'result':result},indent=2),encoding='utf-8'); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args()
    summary = None if args.only else json.loads((FOLDER/'complete.json').read_text()); rows = []
    names = [args.only] if args.only else [r['dataset'] for r in summary['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(audit,name) for name in names]): rows.append(f.result())
    if args.only: print(rows,flush=True); return
    lookup = {r['dataset']:r for r in rows}; rows = [lookup[n] for n in names]
    equivalent(summary['rows'],[{'dataset':r['dataset'],'methods':r['methods']} for r in rows]); counts = sum((Counter(r['counts']) for r in rows),Counter())
    for k,v in summary['counts'].items(): assert counts[k] == v
    assert counts['records'] == 128
    for metric in ('clean','noise'):
        for m in METHODS[1:]: equivalent(summary['contrasts'][metric][m],paired([r['methods'][m][metric]-r['methods']['old'][metric] for r in rows]))
    for m in METHODS:
        for metric in ('clean','noise','removed'): equivalent(summary['means'][m][metric],float(np.mean([r['methods'][m][metric] for r in rows])))
    gate = summary['contrasts']['clean']['refitted']['mean_pp'] >= .5 and summary['means']['refitted']['removed'] >= summary['means']['old']['removed'] and counts['risk_model_caps'] == counts['final_fit_caps'] == 0
    assert gate == summary['gate_passed']
    result = {'passed':True,'counts':dict(counts),'scope':'independent_actual_fold_refits_risk_scores_every_feasible_candidate_geometry_selection_final_queries',
        'limits':'Adaptive_search_scores_not_unbiased_risk; full_reference_labels_in_FRS_screen; old geometry exact-audit-verified reuse; exposed_development',
        'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)


if __name__ == '__main__': main()
