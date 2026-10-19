"""Full child-fold selector/metric isolation, refits, CV choices and queries."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified
from run_nested_geometry_forest import GRIDS,verified_native,holm
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.nested_teacher_regularized import inner_splits
from certified_frfs.geometry_only_forest_stage import GeometryStageContext
from certified_frfs.recent_budget import fgs_budget_path,sfrip_budget_path
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.capped_sparse import capped_sparse_projection
from certified_frfs.functional_pipeline_cv import CONFIGS
from audit_nested_geometry_forest import BA,score as score3
from audit_forest_compact_transfer import equivalent
from audit_spline_capacity_pilot import independent_tree
from functional_spline_audit_oracle import refit_functional
from spline_forest_audit_oracle import score,leaf_margins,lp_margins_first
from run_forest_robustness_curve import coverage
from review_box_iterations import paired

FOLDER = ROOT/'results/functional-normalization-pilot'
METHODS = ('certified','full','fgs_native','sfrip_native','fgs_matched','sfrip_matched')
ALL_METHODS = METHODS+('old3','full3')
SEED = 20261017


def replay_child_supports(A,y,saved,counts):
    """Raw substitution greedy search with general-weight certificates.

    No production Gram ranking or fit_stage selector is called. Teacher
    fitting is the previously audited common classifier; source of label
    information is restricted to this child TRAINING reference.
    """
    context = GeometryStageContext(A,y); context.A = A
    d = A.shape[1]; teacher = context.classifier(list(range(d)),1.,1.)
    q = len(teacher['classes']); balance = np.array([1/(q*sum(y == c)) for c in y]); center = balance@A
    target = score3(A,teacher); target -= target.mean(axis=1,keepdims=True)
    original = np.ones(d)/np.sqrt(d)
    protected = MajorityTerms(A,y,weights=original).bounds(range(d),allowed_enemies=2) > 1e-10
    cache = {}
    def candidate(fs):
        key = tuple(fs)
        if key not in cache:
            weights = original if len(fs) == d else capped_sparse_projection(transfer_weights(A,fs,original)/np.sqrt(d),len(fs),2/d)*np.sqrt(d)
            embedded = np.zeros(d); embedded[fs] = weights
            cache[key] = {'features':fs,'weights':embedded.tolist(),
                'certificate':weight_certificate(A,y,fs,original,weights,protected=protected)}
            counts['child_geometry_replays'] += 1
        return cache[key]
    active = list(range(d))
    while len(active) > int(np.ceil(.8*d)):
        losses = []
        for removed in active:
            fs = [a for a in active if a != removed]; Q = A.copy(); omitted = [a for a in range(d) if a not in fs]
            Q[:,omitted] = center[omitted]; values = score3(Q,teacher); values -= values.mean(axis=1,keepdims=True)
            losses.append((float(balance @ np.mean((values-target)**2,axis=1)),removed,fs)); counts['child_raw_projection_probes'] += 1
        losses.sort(key=lambda row:(round(row[0],12),row[1])); accepted = None
        for _,_,fs in losses:
            if candidate(fs)['certificate']['feasible']: accepted = fs; break
        if accepted is None: break
        active = accepted
    final = candidate(active); assert final['certificate']['feasible']
    # Production candidate contains projection_loss in addition to geometry.
    for key in ('features','weights','certificate'): equivalent(saved['candidate'][key],final[key])
    Q = A.copy(); omitted = [a for a in range(d) if a not in active]; Q[:,omitted] = center[omitted]
    values = score3(Q,teacher); values -= values.mean(axis=1,keepdims=True)
    equivalent(saved['candidate']['projection_loss'],float(balance @ np.mean((values-target)**2,axis=1)))
    fgs = fgs_budget_path(A,y,beta=.8,xi=1.); sfrip = sfrip_budget_path(A,y,beta=3.,confidence=.8)
    equivalent(saved['fgs_path'],fgs); equivalent(saved['sfrip_path'],sfrip); counts['child_native_path_replays'] += 2
    expected = {'certified':active,'full':list(range(d)),
        'fgs_native':sorted(fgs['native']['features']),'sfrip_native':sorted(sfrip['native']['features']),
        'fgs_matched':sorted(fgs['ranking'][:len(active)]),'sfrip_matched':sorted(sfrip['ranking'][:len(active)])}
    equivalent(saved['supports'],expected); assert saved['target_count'] == len(active)
    return expected


def audit(name):
    threadpool_limits(1); counts = Counter(); precision = {k:0. for k in ('coef_error','bias_error','score_error','gradient_inf','objective_gap')}
    artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text()); assert saved['complete']
    source = source_verified(name); native = verified_native(name,source)
    for rel,value in saved['fingerprints'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == value
    fingerprints = {'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'functional_oracle_sha256':hashlib.sha256((ROOT/'scripts/functional_spline_audit_oracle.py').read_bytes()).hexdigest(),
        'spline_oracle_sha256':hashlib.sha256((ROOT/'scripts/spline_forest_audit_oracle.py').read_bytes()).hexdigest()}
    checkpoint = FOLDER/(name+'-audit.json')
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        if previous['fingerprints'] == fingerprints: return previous['result']
    X,y = load_dataset(name); expected = [(i,j,s,inner) for i,s in enumerate(source['splits']) for j,inner in enumerate(s['inner'])]
    assert len(saved['records']) == len(expected)
    for r,(i,j,s,inner) in zip(saved['records'],expected):
        assert r['outer_index'] == i and r['inner_index'] == j and r['outer_reference_indices'] == s['reference_indices']
        for key in ('reference_indices','validation_indices'): assert r[key] == inner[key]
        ri,a,b = (np.array(r[k]) for k in ('outer_reference_indices','reference_indices','validation_indices'))
        assert set(a).isdisjoint(b) and len(set(a)|set(b)) == len(ri)
        raw,labels = X[ri[a]],y[ri[a]]; A,B = preprocess_fit(raw,X[ri[b]])
        tune = r['tuning']; assert tune['seed'] == SEED
        splits = inner_splits(labels,seed=SEED); assert len(tune['folds']) == len(splits)
        for fi,(fold,(ca,cb)) in enumerate(zip(tune['folds'],splits)):
            assert fold['reference_indices'] == ca.tolist() and fold['validation_indices'] == cb.tolist()
            assert set(ca).isdisjoint(cb) and len(set(ca)|set(cb)) == len(raw)
            AA,BB = preprocess_fit(raw[ca],raw[cb]); train,truth = labels[ca],labels[cb]
            supports = replay_child_supports(AA,train,fold['support_block'],counts); tree_cache = {}; used = set()
            for method in METHODS:
                assert len(tune['trace'][method]) == len(CONFIGS)
                for ci,row in enumerate(tune['trace'][method]):
                    assert row['config_index'] == ci and len(row['fold_clean']) == len(splits) and len(row['fold_model_ids']) == len(splits)
                    mid = row['fold_model_ids'][fi]; used.add(mid); model = fold['models'][mid]; fit = model['fit']; cfg = CONFIGS[ci]
                    assert fit['features'] == supports[method] and fit['C'] == cfg['C']
                    np.testing.assert_allclose(fit['knots'],np.tile(cfg['knots'],(A.shape[1],1)),atol=1e-12)
                    assert fit['pair_penalty'] == (1. if cfg['pair_penalty'] is None else cfg['pair_penalty'])
                    fs = fit['features']; key = tuple(fs)
                    if key not in tree_cache: tree_cache[key] = independent_tree(AA,train,fs,np.tile([0,.5,1],(A.shape[1],1)))
                    assert fit['edges'] == ([] if cfg['pair_penalty'] is None else tree_cache[key])
                    clean = BA(score(BB,fit),fit['classes'],truth); equivalent(model['clean'],clean); equivalent(row['fold_clean'][fi],clean)
            assert used == set(range(len(fold['models'])))
            for model in fold['models']:
                fit = model['fit']; diagnostics = refit_functional(AA,train,fit)
                for key,value in diagnostics.items(): precision[key] = max(precision[key],value)
                counts['child_models'] += 1; counts['child_optimizer_caps'] += not fit['optimizer_converged']
                counts['child_independent_refits'] += 1
            counts['child_folds'] += 1
        for method in METHODS:
            rows = tune['trace'][method]; scores = [float(np.mean(row['fold_clean'])) for row in rows]
            best = max(range(len(rows)),key=lambda ci:(round(scores[ci],12),-CONFIGS[ci]['k_index'],-CONFIGS[ci]['C_index'],-CONFIGS[ci]['pair_index']))
            choice = tune['choices'][method]; assert choice['config_index'] == best
            equivalent(choice['mean_clean'],scores[best]); equivalent(choice['all_mean_clean'],scores)
            assert r['selected_configs'][method] == CONFIGS[best]
        oldstage = inner['stages'][7]
        old = inner['models'][oldstage['policies']['0.8']['model_id']]; full = inner['models'][oldstage['policies']['1.0']['model_id']]
        assert {k:v for k,v in r['fits']['old3'].items() if k != 'knots'} == old
        assert {k:v for k,v in r['fits']['full3'].items() if k != 'knots'} == full
        equivalent(r['candidate'],inner['geometry'][oldstage['policies']['0.8']['geometry_id']])
        assert r['candidate']['certificate']['feasible'] and r['target_count'] == len(old['features'])
        supports = {'certified':old['features'],'full':list(range(A.shape[1]))}
        for method,ci,prefix in [('fgs2023',4,'fgs'),('sfrip2025',3,'sfrip')]:
            path = native['splits'][i]['inner'][j]['paths'][method][ci]; actual = r['selector_paths'][method]
            assert actual['selector_index'] == ci and actual['config'] == GRIDS[method][ci]; equivalent(actual['path'],path)
            supports[prefix+'_native'] = sorted(path['native']['features']); supports[prefix+'_matched'] = sorted(path['ranking'][:r['target_count']])
        equivalent(r['supports'],supports)
        exact = all(len(r['fits'][m]['features']) == r['target_count'] for m in ('fgs_matched','sfrip_matched'))
        assert r['counts_exact'] == exact; counts['count_mismatches'] += not exact
        assert r['query_seed'] == 144+i; rng = np.random.default_rng(144+i)
        queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]; truth = y[ri[b]]
        for method,fit in r['fits'].items():
            if method in METHODS:
                cfg = r['selected_configs'][method]; assert fit['features'] == supports[method] and fit['C'] == cfg['C']
                np.testing.assert_allclose(fit['knots'],np.tile(cfg['knots'],(A.shape[1],1)),atol=1e-12)
                assert fit['pair_penalty'] == (1. if cfg['pair_penalty'] is None else cfg['pair_penalty'])
                fs = fit['features']; edges = independent_tree(A,labels,fs,np.tile([0,.5,1],(A.shape[1],1)))
                assert fit['edges'] == ([] if cfg['pair_penalty'] is None else edges)
                diagnostics = refit_functional(A,labels,fit)
                for key,value in diagnostics.items(): precision[key] = max(precision[key],value)
                counts['parent_independent_refits'] += 1
            else: np.testing.assert_allclose(fit['knots'],np.tile([0,.5,1],(A.shape[1],1)),atol=1e-12)
            values = [BA(score(Q,fit),fit['classes'],truth) for Q in queries]
            pair,_,_ = leaf_margins(B,truth,fit,.01); margins = pair.min(axis=1); box = r['boxes'][method]
            np.testing.assert_allclose(box['margins'],margins,atol=1e-9,rtol=1e-9)
            rivals = np.array(box['rivals']); assert rivals.tolist() == pair.argmin(axis=1).tolist()
            witness = np.array(box['witness_raw']); lo,hi = np.maximum(0,B-.01),np.minimum(1,B+.01)
            assert np.all(witness >= lo-1e-12) and np.all(witness <= hi+1e-12)
            wscores = score(witness,fit); own = np.argmax(truth[:,None] == np.array(fit['classes'])[None],axis=1)
            np.testing.assert_allclose(wscores[np.arange(len(B)),own]-wscores[np.arange(len(B)),rivals],margins,atol=1e-9,rtol=1e-9)
            for rival,value in lp_margins_first(B,truth,fit,.01).items():
                np.testing.assert_allclose(value,pair[0,rival],atol=1e-8,rtol=1e-8); counts['parent_lp_solves'] += 1
            equivalent(r['metrics'][method],{'clean':values[0],'noise':float(np.mean(values[1:])),
                'robust':coverage(margins > 1e-10,truth),'removed':1-len(fit['features'])/A.shape[1]})
            counts['parent_optimizer_caps'] += not fit['optimizer_converged']; counts['parent_leaf_replays'] += 1
            counts['parent_witness_replays'] += len(B); counts['query_predictions'] += 4
        counts['records'] += 1; print(name,i,j,'functional pilot audit passed',flush=True)
    result = {'dataset':name,'counts':dict(counts),'precision_maxima':precision,'methods':{m:{metric:float(np.mean([r['metrics'][m][metric] for r in saved['records']]))
        for metric in ('clean','noise','robust','removed')} for m in ALL_METHODS}}
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
    for k,v in summary['counts'].items(): assert counts[k] == v
    assert counts['records'] == 128
    for m in ALL_METHODS:
        for metric in ('clean','noise','robust','removed'): equivalent(summary['means'][m][metric],float(np.mean([r['methods'][m][metric] for r in rows])))
    for metric in ('clean','noise','robust'):
        for m in ALL_METHODS[1:]: equivalent(summary['contrasts'][metric][m],paired([r['methods']['certified'][metric]-r['methods'][m][metric] for r in rows]))
    clean = summary['contrasts']['clean']; corrected = holm({m:clean[m]['sign_flip_two_sided_p'] for m in ('full','fgs_matched','sfrip_matched')})
    equivalent(summary['matched_holm'],corrected)
    gate = all(clean[m]['mean_pp'] >= .5 and corrected[m] < .05 for m in ('fgs_matched','sfrip_matched')) and clean['full']['mean_pp'] >= 0 and counts['count_mismatches'] == counts['parent_optimizer_caps'] == counts['child_optimizer_caps'] == 0
    assert summary['method_gate_passed'] == gate and summary['goal_achieved'] is False
    result = {'passed':True,'counts':dict(counts),'precision_maxima':{k:max(r['precision_maxima'][k] for r in rows) for k in rows[0]['precision_maxima']},
        'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'functional_oracle_sha256':hashlib.sha256((ROOT/'scripts/functional_spline_audit_oracle.py').read_bytes()).hexdigest(),
        'scope':'child_train_only_support_reconstruction_independent_functional_refits_choices_parent_geometry_LP_witnesses_queries',
        'limits':'Fixed_recent_selector_settings; exposed_development; numerical_optimization_caps_reported_not_ignored'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)


if __name__ == '__main__': main()
