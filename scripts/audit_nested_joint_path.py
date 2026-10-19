"""Independent joint/original phase, geometry, scores and choice replay."""
import argparse
import hashlib
import json
from pathlib import Path
from itertools import combinations
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from audit_nested_geometry_forest import (ROOT,score,BA,check_model,equivalent,load_dataset,
    preprocess_fit,interaction_tree,weight_certificate,transfer_weights,capped_sparse_projection,paired)
from certified_frfs.majority_metric import MajorityTerms

SOURCE = ROOT/'results/nested-geometry-forest'
FOLDER = ROOT/'results/nested-joint-path'
CONFIGS = [(c,g) for c in (.1,1.,10.) for g in (None,10.,3.,1.)]
FRACTIONS = (.5,.65,.8,1.)
METHODS = ('proposed','original','full','fgs2023','sfrip2025')


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def block_audit(block,old,raw,labels,counts,*,outer_family=None):
    A,_ = preprocess_fit(raw,raw); d = A.shape[1]
    original = np.ones(d)/np.sqrt(d); protected = MajorityTerms(A,labels,weights=original).bounds(range(d),allowed_enemies=2) > 1e-10
    old_models = {identity(v) for v in old['models']}; old_trees = {identity(v) for v in old['trees']}
    for fit in block['models']:
        if identity(fit) in old_models: counts['reused_verified_models'] += 1
        else: check_model(A,labels,fit,Counter()); counts['independent_new_classifier_refits'] += 1
        counts['models'] += 1; counts['optimizer_caps'] += not fit['optimizer_converged']
    for tree in block['trees']:
        if identity(tree) in old_trees: counts['reused_verified_trees'] += 1
        else:
            fs = tree['features']; local = interaction_tree(A[:,fs],labels); equivalent(local,tree['local_tree'])
            assert tree['edges'] == [sorted([fs[u],fs[v]]) for u,v in local['edges']]
            counts['new_tree_replays'] += 1
    tree_map = {tuple(t['features']):t['edges'] for t in block['trees']}
    for fit in block['models']:
        if fit['edges']: assert fit['edges'] == tree_map[tuple(fit['features'])]
    old_geometry = {tuple(g['features']):g for g in old['geometry']}
    for geometry in block['geometry']:
        fs = geometry['features']
        if tuple(fs) in old_geometry:
            equivalent(geometry,old_geometry[tuple(fs)]); counts['reused_verified_geometries'] += 1
        else:
            w = original if len(fs) == d else capped_sparse_projection(transfer_weights(A,fs,original)/np.sqrt(d),len(fs),2/d)*np.sqrt(d)
            embedded = np.zeros(d); embedded[fs] = w
            np.testing.assert_allclose(geometry['weights'],embedded,atol=1e-12)
            equivalent(geometry['certificate'],weight_certificate(A,labels,fs,original,w,protected=protected))
            counts['new_geometry_replays'] += 1
    stages = block['stages']; stages = list(stages.values()) if isinstance(stages,dict) else stages
    for stage in stages:
        C,gamma = stage['C'],stage['pair_penalty']; path = stage['path']; family = 'joint' if outer_family is None else outer_family
        assert not stage['teacher_guards_enforced'] and path['tie_decimals'] == 12
        assert stage['original_majority_protected_counts'] == {str(c):int(sum(protected & (labels == c))) for c in np.unique(labels)}
        teacher = block['models'][stage['policies']['1.0']['model_id']]
        assert teacher['features'] == list(range(d)) and teacher['C'] == C
        assert teacher['pair_penalty'] == (1. if gamma is None else gamma)
        assert teacher['edges'] == ([] if gamma is None else tree_map[tuple(range(d))])
        classes = np.asarray(teacher['classes']); balance = np.array([1/(len(classes)*sum(labels == c)) for c in labels])
        center = balance @ A; np.testing.assert_allclose(path['center'],center,atol=1e-12)
        target = score(A,teacher); target -= target.mean(axis=1,keepdims=True); probes = {}
        def loss(fs):
            key = tuple(fs)
            if key not in probes:
                replaced = A.copy(); omitted = [a for a in range(d) if a not in fs]; replaced[:,omitted] = center[omitted]
                values = score(replaced,teacher); values -= values.mean(axis=1,keepdims=True)
                probes[key] = float(balance @ np.mean((values-target)**2,axis=1)); counts['direct_raw_projection_probes'] += 1
            return probes[key]
        def candidate(record,fs,value):
            g = block['geometry'][record['geometry_id']]; assert g['features'] == fs
            np.testing.assert_allclose(record['projection_loss'],value,atol=1e-11,rtol=1e-9)
            return g['certificate']['feasible']
        active = list(range(d)); minimum = int(np.ceil(d/2)); ai = ti = gi = jumps = 0
        assert candidate(path['trace'][0],active,0.)
        while len(active) > minimum:
            singles = sorted([(loss([a for a in active if a != z]),(z,)) for z in active],key=lambda v:(round(v[0],12),v[1]))
            phases = [(1,singles)]; accepted = None
            for size,ordered in phases:
                ranking = path['rankings'][gi]; gi += 1
                assert ranking['features'] == active
                if family == 'joint':
                    assert ranking['deletion_size'] == size
                    assert [v['removed'] for v in ranking['choices']] == [list(r) for _,r in ordered]
                else: assert [v['removed'] for v in ranking['choices']] == [r[0] for _,r in ordered]
                np.testing.assert_allclose([v['loss'] for v in ranking['choices']],[value for value,_ in ordered],atol=1e-11,rtol=1e-9)
                for value,removed in ordered:
                    attempted = path['attempts'][ai]; ai += 1; fs = [a for a in active if a not in removed]
                    assert attempted['removed'] == (list(removed) if family == 'joint' else removed[0])
                    feasible = candidate(attempted,fs,value); assert attempted['accepted'] == feasible
                    counts['attempts'] += 1; counts['pair_attempts'] += size == 2; counts['geometry_rejections'] += not feasible
                    if feasible:
                        ti += 1; assert candidate(path['trace'][ti],fs,value)
                        accepted = fs; jumps += size == 2; break
                if accepted is not None: break
                if family == 'joint' and size == 1 and len(active)-2 >= minimum:
                    assert path['pair_pool_size'] == 8
                    pool = sorted(r[0] for _,r in singles[:8])
                    pairs = sorted([(loss([a for a in active if a not in r]),r) for r in combinations(pool,2)],key=lambda v:(round(v[0],12),v[1]))
                    phases.append((2,pairs))
            if accepted is None: break
            active = accepted
        assert ai == len(path['attempts']) and gi == len(path['rankings']) and ti+1 == len(path['trace'])
        if family == 'joint': assert jumps == path['accepted_pair_jumps']
        if outer_family is None: counts['inner_pair_jumps'] += jumps
        else: counts['selected_outer_pair_jumps'] += jumps
        for f in FRACTIONS:
            selected = next(v for v in reversed(path['trace']) if len(block['geometry'][v['geometry_id']]['features']) >= int(np.ceil(f*d)))
            policy = stage['policies'][str(f)]; assert policy['geometry_id'] == selected['geometry_id']
            fit = block['models'][policy['model_id']]; fs = block['geometry'][selected['geometry_id']]['features']
            assert fit['features'] == fs and fit['C'] == C and fit['pair_penalty'] == (1. if gamma is None else gamma)
            assert fit['edges'] == ([] if gamma is None else tree_map[tuple(fs)])
        counts['joint_stages' if family == 'joint' else 'original_stages'] += 1


def audit(name):
    threadpool_limits(1); counts = Counter()
    path = FOLDER/(name+'.json'); saved = json.loads(path.read_text()); assert saved['complete']
    old_path = SOURCE/(name+'.json'); source = json.loads(old_path.read_text()); assert source['complete']
    for rel,value in saved['fingerprints'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == value
    assert json.loads((SOURCE/'complete-audit.json').read_text())['passed']
    per = json.loads((SOURCE/(name+'-audit.json')).read_text())
    assert per['artifact_sha256'] == hashlib.sha256(old_path.read_bytes()).hexdigest()
    assert per['auditor_sha256'] == hashlib.sha256((ROOT/'scripts/audit_nested_geometry_forest.py').read_bytes()).hexdigest()
    hashes = {'artifact_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    checkpoint = FOLDER/(name+'-audit.json')
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        if previous['fingerprints'] == hashes: return previous['result']
    X,y = load_dataset(name); assert len(saved['splits']) == len(source['splits'])
    for i,(s,old) in enumerate(zip(saved['splits'],source['splits'])):
        for k in ('reference_indices','validation_indices'): assert s[k] == old[k]
        ri,vi = np.array(s['reference_indices']),np.array(s['validation_indices']); assert set(ri).isdisjoint(vi)
        assert len(s['inner']) == len(old['inner'])
        expected_choices = [(family,fi,mi,f,fraction_index) for fi,family in enumerate(('original','joint')) for mi in range(12) for fraction_index,f in enumerate(FRACTIONS)]
        assert len(s['trace']) == 96
        for row,(family,fi,mi,f,fraction_index) in zip(s['trace'],expected_choices):
            assert (row['family'],row['family_index'],row['model_index'],row['fraction'],row['fraction_index']) == (family,fi,mi,f,fraction_index)
            assert len(row['folds']) == len(s['inner'])
        for j,(block,previous) in enumerate(zip(s['inner'],old['inner'])):
            for k in ('reference_indices','validation_indices'): assert block[k] == previous[k]
            a,b = np.array(block['reference_indices']),np.array(block['validation_indices'])
            assert set(a).isdisjoint(b) and len(set(a)|set(b)) == len(ri)
            block_audit(block,previous,X[ri[a]],y[ri[a]],counts)
            _,B = preprocess_fit(X[ri[a]],X[ri[b]])
            assert len(block['stages']) == 12
            for row in s['trace']:
                fold = row['folds'][j]; fit = block['models'][fold['model_id']]; C,gamma = CONFIGS[row['model_index']]
                assert fit['features'] == fold['features'] and fit['C'] == C and fit['pair_penalty'] == (1. if gamma is None else gamma)
                if row['family'] == 'original':
                    policy = previous['stages'][row['model_index']]['policies'][str(row['fraction'])]
                    assert fit == previous['models'][policy['model_id']]
                else:
                    policy = block['stages'][row['model_index']]['policies'][str(row['fraction'])]
                    assert fold['model_id'] == policy['model_id']
                np.testing.assert_allclose(fold['clean'],BA(score(B,fit),fit['classes'],y[ri[b]]),atol=1e-12)
                counts['inner_scores'] += 1
            counts['inner_splits'] += 1
        for row in s['trace']:
            np.testing.assert_allclose(row['mean_clean'],np.mean([v['clean'] for v in row['folds']]),atol=1e-12)
            np.testing.assert_allclose(row['mean_size'],np.mean([len(v['features']) for v in row['folds']]),atol=1e-12)
        rows = s['trace']; selected = max(range(96),key=lambda j:(round(rows[j]['mean_clean'],12),-rows[j]['mean_size'],-rows[j]['model_index'],-rows[j]['fraction_index'],-rows[j]['family_index']))
        chosen = rows[selected]; proposed = s['results']['proposed']
        assert proposed['chosen_index'] == selected and proposed['family'] == chosen['family'] and proposed['fraction'] == chosen['fraction'] and proposed['model_index'] == chosen['model_index']
        block_audit(s,old,X[ri],y[ri],counts,outer_family=chosen['family'])
        policy = s['stages']['proposed']['policies'][str(chosen['fraction'])]
        assert proposed['model_id'] == policy['model_id'] and proposed['geometry_id'] == policy['geometry_id']
        fit = s['models'][proposed['model_id']]; assert fit['features'] == proposed['features']
        assert s['geometry'][proposed['geometry_id']]['certificate']['feasible']
        counts['selected_joint'] += chosen['family'] == 'joint'; counts['selected_full'] += len(fit['features']) == saved['d']
        _,B = preprocess_fit(X[ri],X[vi]); rng = np.random.default_rng(144+i); assert s['query_seed'] == 144+i
        queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        for method,r in s['results'].items():
            fit = s['models'][r['model_id']]; assert r['features'] == fit['features']
            if method != 'proposed':
                source_method = 'proposed' if method == 'original' else method
                expected = old['results'][source_method]
                assert r['source_method'] == source_method and r['source_model_id'] == expected['model_id']
                assert fit == old['models'][expected['model_id']]
            values = [BA(score(Q,fit),fit['classes'],y[vi]) for Q in queries]
            equivalent(r['metrics'],{'clean':values[0],'noise':float(np.mean(values[1:]))})
            if method != 'proposed': equivalent(r['metrics'],old['results'][r['source_method']]['metrics'])
            counts['outer_predictions'] += 4
        counts['outer_splits'] += 1; print(name,i,'nested joint audit passed',flush=True)
    methods = {m:{metric:float(np.mean([s['results'][m]['metrics'][metric] for s in saved['splits']])) for metric in ('clean','noise')} for m in METHODS}
    for m in METHODS: methods[m]['removed'] = float(np.mean([1-len(s['results'][m]['features'])/saved['d'] for s in saved['splits']]))
    result = {'dataset':name,'methods':methods,'counts':dict(counts)}
    checkpoint.write_text(json.dumps({'fingerprints':hashes,'result':result},indent=2),encoding='utf-8'); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args()
    summary = None if args.only else json.loads((FOLDER/'complete.json').read_text())
    names = [args.only] if args.only else [r['dataset'] for r in summary['rows']]; rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(audit,name) for name in names]): rows.append(f.result())
    if args.only: print(rows,flush=True); return
    lookup = {r['dataset']:r for r in rows}; rows = [lookup[n] for n in names]
    equivalent(summary['rows'],[{'dataset':r['dataset'],'methods':r['methods']} for r in rows])
    counts = sum((Counter(r['counts']) for r in rows),Counter())
    for k,v in summary['counts'].items(): assert counts[k] == v
    assert counts['outer_splits'] == 44 and counts['inner_splits'] == 128
    for metric in ('clean','noise'):
        for m in METHODS[1:]: equivalent(summary['contrasts'][metric][m],paired([r['methods']['proposed'][metric]-r['methods'][m][metric] for r in rows]))
    for m in METHODS:
        for metric in ('clean','noise','removed'): equivalent(summary['means'][m][metric],float(np.mean([r['methods'][m][metric] for r in rows])))
    p = {m:summary['contrasts']['clean'][m]['sign_flip_two_sided_p'] for m in ('full','fgs2023','sfrip2025')}
    previous = 0.; adjusted = {}
    for j,m in enumerate(sorted(p,key=p.get)):
        previous = max(previous,min(1.,(3-j)*p[m])); adjusted[m] = previous
    equivalent(summary['descriptive_holm_p'],adjusted)
    gate = all(summary['contrasts']['clean'][m]['mean_pp'] >= .5 for m in ('fgs2023','sfrip2025')) and summary['contrasts']['clean']['full']['mean_pp'] >= 0 and summary['means']['proposed']['removed'] >= .1 and counts['optimizer_caps'] == 0
    assert gate == summary['gate_passed']
    result = {'passed':True,'counts':dict(counts),'scope':'raw_substitution_phases_new_geometry_model_replay_all_inner_choices_outer_predictions',
        'limits':'Old fits/trees/geometries reused only with source independent audit and exact hashes; exposed development; native convention gaps; no population or teacher-margin guarantee',
        'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)


if __name__ == '__main__': main()
