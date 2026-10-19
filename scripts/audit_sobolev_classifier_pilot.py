"""Source-isolated CV, independent Sobolev refits, paths and raw-box audit."""
import argparse,json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from run_sobolev_classifier_pilot import INPUT,FOLDER,METHODS,ALL_METHODS,CONFIGS
from run_nested_geometry_forest import holm
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.capped_sparse import capped_sparse_projection
from audit_nested_geometry_forest import BA,check_model
from audit_forest_compact_transfer import equivalent
from audit_spline_capacity_pilot import independent_tree
from audit_functional_normalization_pilot import replay_child_supports
from audit_empirical_product_fixed_count import marginal_integrals,replay_path
from spline_forest_audit_oracle import score,leaf_margins,lp_margins_first
from sobolev_audit_oracle import refit
from run_forest_robustness_curve import coverage
from review_box_iterations import paired

def audit(name):
    threadpool_limits(1); source = source_verified(name); saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete']
    parent = json.loads((INPUT/(name+'.json')).read_text()); upstream = json.loads((INPUT/(name+'-audit.json')).read_text())
    assert upstream['fingerprints'][name+'.json'] == digest(INPUT/(name+'.json'))
    for rel,value in saved['fingerprints'].items(): assert digest(ROOT/rel) == value
    fingerprints = {p.name:digest(p) for p in [FOLDER/(name+'.json'),Path(__file__),ROOT/'scripts/sobolev_audit_oracle.py',
        ROOT/'scripts/audit_empirical_product_fixed_count.py',ROOT/'scripts/audit_functional_normalization_pilot.py']}
    checkpoint = FOLDER/(name+'-audit.json')
    if checkpoint.exists():
        prior = json.loads(checkpoint.read_text())
        if prior['fingerprints'] == fingerprints: return prior['result']
    counts = Counter(); precision = {k:0. for k in ('coef_error','bias_error','score_error','gradient_inf','objective_gap')}
    X,y = load_dataset(name); assert len(saved['records']) == len(source['splits'])
    for i,(r,s) in enumerate(zip(saved['records'],source['splits'])):
        assert r['outer_index'] == i
        for key in ('reference_indices','validation_indices'): assert r[key] == s[key]
        ri,qi = np.array(r['reference_indices']),np.array(r['validation_indices']); assert set(ri).isdisjoint(qi)
        raw,labels = X[ri],y[ri]; A,B = preprocess_fit(raw,X[qi]); d = A.shape[1]
        children = [c for c in parent['records'] if c['outer_index'] == i]; assert len(children) == len(r['folds']) == len(s['inner'])
        for j,(fold,child) in enumerate(zip(r['folds'],children)):
            assert fold['inner_index'] == j
            for key in ('reference_indices','validation_indices'): assert fold[key] == child[key]
            a,b = np.array(fold['reference_indices']),np.array(fold['validation_indices']); assert set(a).isdisjoint(b)
            AA,BB = preprocess_fit(raw[a],raw[b]); mapping = {'refined':'empirical_refit','old_selector':'old','full':'full','fgs_matched':'fgs_matched','sfrip_matched':'sfrip_matched'}
            used = set()
            for method in METHODS:
                rows = r['trace'][method]; assert len(rows) == len(CONFIGS)
                for ci,row in enumerate(rows):
                    assert row['config_index'] == ci and len(row['scores']) == len(children) and len(row['model_ids']) == len(children)
                    mid = row['model_ids'][j]; used.add(mid); model = fold['models'][mid]; fit = model['fit']; cfg = CONFIGS[ci]; prior = child['fits'][mapping[method]]
                    assert fit['features'] == prior['features'] and fit['edges'] == ([] if cfg['pair'] is None else prior['edges'])
                    assert fit['C'] == cfg['C'] and fit['alpha'] == cfg['alpha']
                    equivalent(model['clean'],BA(score(BB,fit),fit['classes'],labels[b])); equivalent(row['scores'][j],model['clean'])
            assert used == set(range(len(fold['models'])))
            for model in fold['models']:
                diagnostic = refit(AA,labels[a],model['fit'])
                for k,v in diagnostic.items(): precision[k] = max(precision[k],v)
                counts['child_models'] += 1; counts['child_caps'] += not model['fit']['optimizer_converged']; counts['child_refits'] += 1
        for method in ALL_METHODS:
            base = 'refined' if method == 'refined_alpha0' else 'old_selector' if method == 'old_alpha0' else method
            allowed = [ci for ci,c in enumerate(CONFIGS) if not method.endswith('alpha0') or c['alpha'] == 0]
            best = max(allowed,key=lambda ci:(round(float(np.mean(r['trace'][base][ci]['scores'])),12),-CONFIGS[ci]['alpha_index'],-CONFIGS[ci]['C_index'],-CONFIGS[ci]['pair_index']))
            assert r['choices'][method] == best
        old = r['stage']['policies']['0.8']['candidate']; target = len(old['features']); assert r['target_count'] == target
        fake = {'candidate':old,'target_count':target,'fgs_path':r['fgs_path'],'sfrip_path':r['sfrip_path'],
            'supports':{'certified':old['features'],'full':list(range(d)),
                'fgs_native':sorted(r['fgs_path']['native']['features']),'sfrip_native':sorted(r['sfrip_path']['native']['features']),
                'fgs_matched':sorted(r['fgs_path']['ranking'][:target]),'sfrip_matched':sorted(r['sfrip_path']['ranking'][:target])}}
        replay_child_supports(A,labels,fake,counts)
        check_model(A,labels,{k:v for k,v in r['teacher'].items() if k != 'knots'},Counter()); counts['teacher_refits'] += 1
        probes = np.r_[A,np.random.default_rng(283).uniform(size=(100,d))]
        np.testing.assert_allclose(score(probes,r['teacher']),score(probes,r['centered_teacher']),atol=1e-10,rtol=1e-10)
        energy,law = marginal_integrals(A,labels,r['centered_teacher']); equivalent(law,r['marginal_law'])
        for k,v in energy.items(): equivalent(r['energy'][k],v)
        counts['component_integrals'] += d+len(r['teacher']['edges'])
        original = np.ones(d)/np.sqrt(d); protected = MajorityTerms(A,labels,weights=original).bounds(range(d),allowed_enemies=2) > 1e-10; cache = {}
        def geometry(fs):
            key = tuple(fs)
            if key not in cache:
                weights = original if len(fs) == d else capped_sparse_projection(transfer_weights(A,fs,original)/np.sqrt(d),len(fs),2/d)*np.sqrt(d)
                embedded = np.zeros(d); embedded[fs] = weights
                cache[key] = {'features':fs,'weights':embedded.tolist(),'certificate':weight_certificate(A,labels,fs,original,weights,protected=protected)}
                counts['geometry_replays'] += 1
            return cache[key]
        replay_path(r['path'],r['energy'],geometry,counts)
        supports = {**{k:v for k,v in fake['supports'].items() if k in ('full','fgs_matched','sfrip_matched')},
            'old_selector':old['features'],'old_alpha0':old['features'],'refined':r['path']['features'],'refined_alpha0':r['path']['features']}
        equivalent(r['supports'],supports); exact = all(len(supports[m]) == target for m in ('refined','old_selector','fgs_matched','sfrip_matched'))
        assert exact == r['counts_exact']; counts['count_mismatches'] += not exact; counts['refined_changed'] += supports['refined'] != supports['old_selector']
        assert r['query_seed'] == 144+i; rng = np.random.default_rng(144+i); queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        for method,fit in r['fits'].items():
            cfg = CONFIGS[r['choices'][method]]; assert fit['features'] == supports[method] and fit['C'] == cfg['C'] and fit['alpha'] == cfg['alpha']
            assert fit['edges'] == ([] if cfg['pair'] is None else independent_tree(A,labels,fit['features'],fit['knots']))
            np.testing.assert_allclose(fit['knots'],np.tile([0.,.5,1.],(d,1)),atol=1e-12)
            diagnostic = refit(A,labels,fit)
            for k,v in diagnostic.items(): precision[k] = max(precision[k],v)
            counts['parent_refits'] += 1; counts['parent_caps'] += not fit['optimizer_converged']
            values = [BA(score(Q,fit),fit['classes'],y[qi]) for Q in queries]
            pair,_,_ = leaf_margins(B,y[qi],fit,.01); margins = pair.min(axis=1); box = r['boxes'][method]
            np.testing.assert_allclose(box['margins'],margins,atol=1e-9,rtol=1e-9); rivals = np.array(box['rivals']); assert rivals.tolist() == pair.argmin(axis=1).tolist()
            witness = np.array(box['witness_raw']); assert np.all(witness >= np.maximum(0,B-.01)-1e-12) and np.all(witness <= np.minimum(1,B+.01)+1e-12)
            ws = score(witness,fit); own = np.argmax(y[qi,None] == np.array(fit['classes'])[None],axis=1)
            np.testing.assert_allclose(ws[np.arange(len(B)),own]-ws[np.arange(len(B)),rivals],margins,atol=1e-9,rtol=1e-9)
            for rival,value in lp_margins_first(B,y[qi],fit,.01).items():
                np.testing.assert_allclose(value,pair[0,rival],atol=1e-8,rtol=1e-8); counts['box_LP'] += 1
            equivalent(r['metrics'][method],{'clean':values[0],'noise':float(np.mean(values[1:])),
                'robust':coverage(margins > 1e-10,y[qi]),'removed':1-len(fit['features'])/d})
            counts['leaf_replays'] += 1; counts['witness_replays'] += len(B); counts['query_predictions'] += 4
        counts['records'] += 1; print(name,i,'Sobolev audit passed',flush=True)
    result = {'dataset':name,'counts':dict(counts),'precision_maxima':precision,'methods':{m:{metric:float(np.mean([r['metrics'][m][metric] for r in saved['records']])) for metric in ('clean','noise','robust','removed')} for m in ALL_METHODS}}
    checkpoint.write_text(json.dumps({'fingerprints':fingerprints,'result':result},indent=2),encoding='utf-8'); return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args()
    summary = None if args.only else json.loads((FOLDER/'complete.json').read_text()); names = [args.only] if args.only else [r['dataset'] for r in summary['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = [f.result() for f in as_completed([pool.submit(audit,n) for n in names])]
    if args.only: print(rows,flush=True)
    else:
        lookup = {r['dataset']:r for r in rows}; rows = [lookup[n] for n in names]; counts = sum((Counter(r['counts']) for r in rows),Counter())
        equivalent(summary['rows'],[{'dataset':r['dataset'],'methods':r['methods']} for r in rows])
        for k,v in summary['counts'].items(): assert counts[k] == v
        for m in ALL_METHODS:
            for metric in ('clean','noise','robust','removed'): equivalent(summary['means'][m][metric],float(np.mean([r['methods'][m][metric] for r in rows])))
        for metric in ('clean','noise','robust'):
            for m in ALL_METHODS[1:]: equivalent(summary['contrasts'][metric][m],paired([r['methods']['refined'][metric]-r['methods'][m][metric] for r in rows]))
        clean = summary['contrasts']['clean']; corrected = holm({m:clean[m]['sign_flip_two_sided_p'] for m in ('full','fgs_matched','sfrip_matched')}); equivalent(summary['matched_holm'],corrected)
        gate = all(clean[m]['mean_pp'] >= .5 and corrected[m] < .05 for m in ('fgs_matched','sfrip_matched')) and clean['full']['mean_pp'] >= 0 and summary['means']['refined']['removed'] >= .1 and counts['count_mismatches'] == counts['parent_caps'] == counts['child_caps'] == 0
        assert gate == summary['development_numeric_gate'] and not summary['goal_achieved']
        result = {'passed':True,'counts':dict(counts),'precision_maxima':{k:max(r['precision_maxima'][k] for r in rows) for k in rows[0]['precision_maxima']},
            'auditor_sha256':digest(Path(__file__)),'scope':'independent_quadrature_stationarity_refits_CVchoices_sourceisolation_outercertified_paths_allboxes',
            'limits':'Exposed_development_fixed_recent_selector_settings_generic_Sobolev_regularization_not_novel_FRS_theory'}
        (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)
