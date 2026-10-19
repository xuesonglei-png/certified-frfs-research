"""Independent all-inner/outer exact margins, selection, supports, fits audit."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from audit_nested_geometry_forest import (ROOT,score,BA,check_model,equivalent,load_dataset,
    preprocess_fit,interaction_tree,weight_certificate,transfer_weights,capped_sparse_projection)
from certified_frfs.majority_metric import MajorityTerms
from audit_forest_robustness_curve import independent_boxes,lp_first_query
from run_nested_joint_path import source_verified
from run_nested_geometry_forest import MODELS,GRIDS,PATHS,verified_native,holm
from run_forest_robustness_curve import RADII,coverage,area
from review_box_iterations import paired

FOLDER = ROOT/'results/nested-robust-policy'
METHODS = ('proposed','full','fgs2023','sfrip2025')
ALL_METHODS = METHODS+tuple('old_'+m for m in METHODS)


def independent_choice(rows):
    clean = [float(np.mean([r['clean'] for r in row['folds']])) for row in rows]
    robust = [float(np.mean([r['robust'] for r in row['folds']])) for row in rows]
    sizes = [float(np.mean([len(r['features']) for r in row['folds']])) for row in rows]
    for i,row in enumerate(rows):
        equivalent(row['mean_clean'],clean[i]); equivalent(row['mean_robust'],robust[i]); equivalent(row['mean_size'],sizes[i])
    best = max(round(v,12) for v in clean)
    eligible = [i for i,v in enumerate(clean) if round(v,12) >= best-.005-1e-12]
    chosen = max(eligible,key=lambda i:(round(robust[i],12),round(clean[i],12),-sizes[i],
        -rows[i]['model_index'],-rows[i]['selector_index'],-rows[i]['fraction_index']))
    return chosen,eligible,best


def audit_new_stage(block,source,A,y,counts):
    """Cached source geometries verified by equality; new ones recalculated."""
    d = A.shape[1]; original = np.ones(d)/np.sqrt(d)
    protected = MajorityTerms(A,y,weights=original).bounds(range(d),allowed_enemies=2) > 1e-10
    cached = {tuple(g['features']):g for g in source['geometry']}; checked = set()
    def geometry(gid):
        point = block['geometry'][gid]; fs = point['features']; key = tuple(fs)
        if key not in checked:
            if key in cached:
                equivalent(point,cached[key]); counts['verified_source_geometries'] += 1
            else:
                w = original if len(fs) == d else capped_sparse_projection(transfer_weights(A,fs,original)/np.sqrt(d),len(fs),2/d)*np.sqrt(d)
                embedded = np.zeros(d); embedded[fs] = w
                equivalent(point,{'features':fs,'weights':embedded.tolist(),
                    'certificate':weight_certificate(A,y,fs,original,w,protected=protected)})
                counts['new_geometry_replays'] += 1
            checked.add(key)
        return point
    stage = block['stages']['proposed']; path = stage['path']; C,gamma = stage['C'],stage['pair_penalty']
    teacher = block['models'][stage['policies']['1.0']['model_id']]
    assert teacher['features'] == list(range(d)) and teacher['C'] == C
    classes = np.unique(y); balance = np.array([1/(len(classes)*sum(y == c)) for c in y])
    center = balance @ A; np.testing.assert_allclose(path['center'],center,atol=1e-12)
    assert stage['original_majority_protected_counts'] == {str(c):int(np.sum(protected & (y == c))) for c in classes}
    assert not stage['teacher_guards_enforced'] and path['tie_decimals'] == 12
    target = score(A,teacher); target -= target.mean(axis=1,keepdims=True); probes = {}
    def loss(fs):
        key = tuple(fs)
        if key not in probes:
            query = A.copy(); omitted = [a for a in range(d) if a not in fs]; query[:,omitted] = center[omitted]
            values = score(query,teacher); values -= values.mean(axis=1,keepdims=True)
            probes[key] = float(balance @ np.mean((values-target)**2,axis=1)); counts['raw_projection_probes'] += 1
        return probes[key]
    def point(record,fs,value):
        g = geometry(record['geometry_id']); assert g['features'] == fs
        np.testing.assert_allclose(record['projection_loss'],value,atol=1e-11,rtol=1e-9)
        return g['certificate']['feasible']
    active = list(range(d)); ai = ti = 0; accepted = False; assert point(path['trace'][0],active,0.)
    for ranking in path['rankings']:
        assert ranking['features'] == active and len(active) > int(np.ceil(d/2))
        ordered = sorted([(loss([a for a in active if a != removed]),removed) for removed in active],key=lambda r:(round(r[0],12),r[1]))
        assert [r['removed'] for r in ranking['choices']] == [a for _,a in ordered]
        np.testing.assert_allclose([r['loss'] for r in ranking['choices']],[v for v,_ in ordered],atol=1e-11,rtol=1e-9)
        accepted = False
        for value,removed in ordered:
            attempt = path['attempts'][ai]; ai += 1; fs = [a for a in active if a != removed]
            assert attempt['removed'] == removed; feasible = point(attempt,fs,value); assert attempt['accepted'] == feasible
            counts['attempts'] += 1
            if feasible:
                ti += 1; assert point(path['trace'][ti],fs,value); active = fs; accepted = True; break
        if not accepted: assert ranking is path['rankings'][-1]
    assert ai == len(path['attempts']) and ti+1 == len(path['trace']) and (len(active) == int(np.ceil(d/2)) or not accepted)
    for fraction,policy in stage['policies'].items():
        record = next(r for r in reversed(path['trace']) if len(geometry(r['geometry_id'])['features']) >= int(np.ceil(float(fraction)*d)))
        assert policy['geometry_id'] == record['geometry_id']
        fit = block['models'][policy['model_id']]; assert fit['features'] == geometry(policy['geometry_id'])['features']
        assert fit['C'] == C and fit['pair_penalty'] == (1. if gamma is None else gamma)
    return stage


def audit(name):
    threadpool_limits(1); counts = Counter(); artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text())
    source = source_verified(name); native = verified_native(name,source)
    assert saved['complete'] and saved['selection_radius'] == .03 and saved['radii'] == list(RADII)
    for rel,value in saved['fingerprints'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == value
    fingerprints = {'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    checkpoint = FOLDER/(name+'-audit.json')
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        if previous['fingerprints'] == fingerprints: return previous['result']
    X,y = load_dataset(name); assert len(saved['splits']) == len(source['splits'])
    for i,(split,original) in enumerate(zip(saved['splits'],source['splits'])):
        for key in ('reference_indices','validation_indices'): assert split[key] == original[key]
        ri,vi = np.array(split['reference_indices']),np.array(split['validation_indices'])
        assert set(ri).isdisjoint(vi) and len(split['inner']) == len(original['inner'])
        for j,(inner,old_inner) in enumerate(zip(split['inner'],original['inner'])):
            for key in ('reference_indices','validation_indices'): assert inner[key] == old_inner[key]
            a,b = np.array(inner['reference_indices']),np.array(inner['validation_indices'])
            assert set(a).isdisjoint(b) and len(set(a)|set(b)) == len(ri)
            _,B = preprocess_fit(X[ri[a]],X[ri[b]]); truth = y[ri[b]]
            mids = sorted({row['folds'][j]['model_id'] for rows in original['trace'].values() for row in rows})
            assert [r['model_id'] for r in inner['scores']] == mids
            cache = {}
            for row in inner['scores']:
                fit = old_inner['models'][row['model_id']]; exact,_,_,_ = independent_boxes(B,truth,fit,.03)
                margins = exact.min(axis=1); np.testing.assert_allclose(row['margins'],margins,atol=1e-9,rtol=1e-9)
                equivalent(row['robust'],coverage(margins > 1e-10,truth)); equivalent(row['clean'],BA(score(B,fit),fit['classes'],truth))
                cache[row['model_id']] = row; counts['inner_score_models'] += 1; counts['inner_leaf_replays'] += 1
            for m,rows in split['trace'].items():
                assert len(rows) == len(original['trace'][m])
                for row,oldrow in zip(rows,original['trace'][m]):
                    for key in ('model_index','selector_index','fraction_index'): assert row[key] == oldrow[key]
                    if m == 'proposed': assert row['fraction'] == oldrow['fraction']
                    fold = row['folds'][j]; oldfold = oldrow['folds'][j]
                    for key in ('model_id','features','clean'): equivalent(fold[key],oldfold[key])
                    equivalent(fold['robust'],cache[fold['model_id']]['robust'])
            counts['inner_splits'] += 1
        for m,rows in split['trace'].items():
            chosen,eligible,best = independent_choice(rows); choice = split['choices'][m]
            assert choice['chosen_index'] == chosen and choice['eligible_indices'] == eligible and choice['best_clean'] == best
            assert choice['clean_tolerance'] == .005 and split['results'][m]['chosen_index'] == chosen
        A,B = preprocess_fit(X[ri],X[vi]); labels = y[ri]; truth = y[vi]; results = split['results']
        assert split['proposed_stage_reused'] == results['proposed']['outer_fit_reused']
        new_stage = None if split['proposed_stage_reused'] else audit_new_stage(split,original,A,labels,counts)
        for fit in split['models']:
            if any(fit == old for old in original['models']): counts['verified_source_models'] += 1
            else:
                fs = fit['features']; expected_edges = [] if not fit['edges'] else [sorted([fs[u],fs[v]]) for u,v in interaction_tree(A[:,fs],labels)['edges']]
                assert fit['edges'] == expected_edges; check_model(A,labels,fit,counts)
            counts['optimizer_caps'] += 0  # check_model already counts new caps
        for m in METHODS:
            row = results[m]; choice = split['trace'][m][row['chosen_index']]; fit = split['models'][row['model_id']]
            assert row['model_index'] == choice['model_index'] and row['selector_index'] == choice['selector_index']
            oldresult = original['results'][m]; reused = row['chosen_index'] == oldresult['chosen_index']
            assert row['outer_fit_reused'] == reused
            cfg = MODELS[choice['model_index']]
            assert fit['C'] == cfg['C'] and fit['pair_penalty'] == (1. if cfg['pair_penalty'] is None else cfg['pair_penalty'])
            if reused: assert fit == original['models'][oldresult['model_id']]
            if m == 'proposed':
                if reused:
                    certificate = original['geometry'][oldresult['geometry_id']]['certificate']
                else:
                    policy = new_stage['policies'][str(choice['fraction'])]
                    assert row['geometry_id'] == policy['geometry_id'] and row['model_id'] == policy['model_id']
                    certificate = split['geometry'][policy['geometry_id']]['certificate']
                assert certificate['feasible']; counts['proposed_certificates'] += 1
            elif m == 'full': assert fit['features'] == list(range(X.shape[1]))
            else:
                config = GRIDS[m][choice['selector_index']]; path = split['native_paths'][m]
                assert path['config'] == config
                key = json.dumps(config,sort_keys=True); archived = native['splits'][i]['outer_paths'][m]
                actual = archived[key] if key in archived else PATHS[m](A,labels,**config)
                equivalent(path['path'],actual); assert fit['features'] == sorted(actual['native']['features'])
            counts['all_selected_changed'] += not reused
            if m == 'proposed': counts['selected_proposed_changed'] += not reused
        assert split['query_seed'] == 144+i; rng = np.random.default_rng(144+i)
        queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        for m,row in results.items():
            fit = split['models'][row['model_id']]; assert row['features'] == fit['features']
            if m.startswith('old_'):
                oldresult = original['results'][m[4:]]; assert fit == original['models'][oldresult['model_id']]
            noise = [BA(score(Q,fit),fit['classes'],truth) for Q in queries]
            curves = []
            for eps,margins_saved in zip(RADII,row['margins']):
                exact,_,points,Z = independent_boxes(B,truth,fit,eps)
                margins = exact.min(axis=1); np.testing.assert_allclose(margins_saved,margins,atol=1e-9,rtol=1e-9)
                for rival,value in lp_first_query(fit,int(truth[0]),points[0],Z[0]).items():
                    np.testing.assert_allclose(value,exact[0,rival],atol=1e-8,rtol=1e-8); counts['outer_lp_solves'] += 1
                curves.append(coverage(margins > 1e-10,truth)); counts['outer_leaf_replays'] += 1
            equivalent(row['metrics'],{'clean':noise[0],'noise':float(np.mean(noise[1:])),
                'robust':curves[2],'area':area(curves),'removed':1-len(fit['features'])/X.shape[1],'curve':curves})
            counts['outer_score_replays'] += 4
        # Summary cap count uses unique registry models, including cache reuse.
        counts['summary_optimizer_caps'] += sum(not f['optimizer_converged'] for f in split['models'])
        counts['outer_splits'] += 1; print(name,i,'nested robust audit passed',flush=True)
    result = {'dataset':name,'counts':dict(counts),'methods':{m:{metric:float(np.mean([s['results'][m]['metrics'][metric] for s in saved['splits']]))
        for metric in ('clean','noise','robust','area','removed')} for m in ALL_METHODS}}
    checkpoint.write_text(json.dumps({'fingerprints':fingerprints,'result':result},indent=2),encoding='utf-8'); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args()
    summary = None if args.only else json.loads((FOLDER/'complete.json').read_text())
    names = [args.only] if args.only else [r['dataset'] for r in summary['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = [f.result() for f in as_completed([pool.submit(audit,n) for n in names])]
    if args.only: print(rows,flush=True); return
    lookup = {r['dataset']:r for r in rows}; rows = [lookup[n] for n in names]
    equivalent(summary['rows'],[{'dataset':r['dataset'],'methods':r['methods']} for r in rows])
    counts = sum((Counter(r['counts']) for r in rows),Counter())
    for k,v in summary['counts'].items(): assert counts['summary_optimizer_caps' if k == 'optimizer_caps' else k] == v
    assert counts['outer_splits'] == 44 and counts['inner_splits'] == 128 and counts['proposed_certificates'] == 44
    for m in ALL_METHODS:
        for metric in ('clean','noise','robust','area','removed'): equivalent(summary['means'][m][metric],float(np.mean([r['methods'][m][metric] for r in rows])))
    for m in METHODS:
        for metric in ('clean','noise','robust','area'): equivalent(summary['selection_effects'][m][metric],paired([r['methods'][m][metric]-r['methods']['old_'+m][metric] for r in rows]))
    for metric in ('clean','robust','area'):
        for m in METHODS[1:]: equivalent(summary['new_method_contrasts'][metric][m],paired([r['methods']['proposed'][metric]-r['methods'][m][metric] for r in rows]))
        equivalent(summary['descriptive_holm'][metric],holm({m:v['sign_flip_two_sided_p'] for m,v in summary['new_method_contrasts'][metric].items()}))
    effect = summary['selection_effects']['proposed']
    gate = effect['robust']['mean_pp'] >= 1 and effect['robust']['sign_flip_two_sided_p'] < .05 and effect['clean']['mean_pp'] >= -.5 and counts['summary_optimizer_caps'] == 0
    assert summary['gate_passed'] == gate and summary['goal_achieved'] is False
    result = {'passed':True,'counts':dict(counts),'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'all_inner_and_outer_independent_leaf_margins_choices_new_certified_paths_models_native_supports_LP_queries',
        'limits':'Adaptive_training_validation_clean_band_not_test_noninferiority; native_budgets_differ; exposed_development'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)


if __name__ == '__main__': main()
