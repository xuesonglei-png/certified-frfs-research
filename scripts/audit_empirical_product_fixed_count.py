"""Independent marginal integrals, MILP, feasible paths, refits and box checks."""
import argparse,json,hashlib,itertools
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from run_nested_geometry_forest import GRIDS,verified_native,holm
from run_empirical_product_fixed_count import FOLDER,METHODS
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.capped_sparse import capped_sparse_projection
from audit_nested_geometry_forest import BA,check_model
from audit_forest_compact_transfer import equivalent
from audit_spline_capacity_pilot import independent_tree
from audit_forest_anova_dp_pilot import integrated_energies,independent_optimum
from spline_forest_audit_oracle import score,leaf_margins,lp_margins_first
from run_forest_robustness_curve import coverage
from review_box_iterations import paired


def marginal_integrals(A,labels,fit):
    classes = np.unique(labels); mass = np.array([1/(len(classes)*sum(labels == c)) for c in labels])
    W = np.array(fit['coef']); q,d,k = W.shape
    V = np.array(fit['pair_coef']).reshape(q,len(fit['edges']),k,k)
    Z = [np.column_stack([np.interp(A[:,a],fit['knots'][a],np.eye(k)[j]) for j in range(k)]) for a in range(d)]
    means = [mass@z for z in Z]; grams = [z.T@(mass[:,None]*z) for z in Z]
    for a in range(d): np.testing.assert_allclose(W[:,a]@means[a],0,atol=1e-9)
    for e,(u,v) in enumerate(fit['edges']):
        np.testing.assert_allclose(V[:,e]@means[v],0,atol=1e-9)
        np.testing.assert_allclose(np.einsum('k,qkl->ql',means[u],V[:,e]),0,atol=1e-9)
    CW,CV = W-W.mean(axis=0,keepdims=True),V-V.mean(axis=0,keepdims=True)
    unaries = [float(mass@np.mean((z@CW[:,a].T)**2,axis=1)) for a,z in enumerate(Z)]
    # Independent second-moment contraction. Avoid an n-by-n product grid
    # while integrating exactly over both empirical coordinate marginals.
    pairs = []
    for e,(u,v) in enumerate(fit['edges']):
        value = 0.
        for c in range(q):
            block = CV[c,e]; value += float(np.sum((grams[u]@block@grams[v])*block))/q
        pairs.append(value)
    energy = {'unary_energy':unaries,'pair_energy':pairs,'edges':fit['edges'],'q':q,'total_energy':float(sum(unaries)+sum(pairs))}
    return energy,{'marginal_means':np.array(means).tolist(),'marginal_grams':np.array(grams).tolist(),'weights':mass.tolist()}


def loss(energy,fs):
    keep = set(fs)
    return float(sum(v for a,v in enumerate(energy['unary_energy']) if a not in keep)
        +sum(v for (u,w),v in zip(energy['edges'],energy['pair_energy']) if u not in keep or w not in keep))


def replay_path(path,energy,geometry,counts):
    fs = path['initial_features']; initial = loss(energy,fs); dp = path['dp']; count = len(fs)
    best,_,method = independent_optimum(energy,count); counts['independent_'+method+'_optima'] += 1
    np.testing.assert_allclose(energy['total_energy']-loss(energy,dp['features']),best,atol=1e-9,rtol=1e-9)
    assert len(dp['features']) == dp['count'] == count
    equivalent(dp['retained_energy'],energy['total_energy']-loss(energy,dp['features']))
    equivalent(dp['omission_energy'],max(0.,energy['total_energy']-dp['retained_energy']))
    assert path['swap_pool'] == 8 and path['rounds'] == 2 and geometry(fs)['certificate']['feasible']
    attempts = iter(path['attempts']); current = initial; expected_stages = []
    def check_attempt(expected,support,value):
        actual = next(attempts); candidate = geometry(support)
        for key,want in expected.items(): assert actual[key] == want
        equivalent(actual['candidate'],candidate); equivalent(actual['loss'],value)
        assert actual['accepted'] == candidate['certificate']['feasible']
        counts['attempts_replayed'] += 1
        return actual['accepted']
    if dp['omission_energy'] < current-1e-12 and dp['features'] != fs:
        if check_attempt({'source':'global_surrogate_DP'},dp['features'],loss(energy,dp['features'])):
            fs = dp['features']; current = loss(energy,fs)
    for step in range(2):
        candidates = []
        for removed in fs:
            for added in range(len(energy['unary_energy'])):
                if added in fs: continue
                support = sorted([a for a in fs if a != removed]+[added]); value = loss(energy,support)
                if value < current-1e-12: candidates.append((value,tuple(support),removed,added))
        candidates.sort(key=lambda row:(row[0],row[1]))
        expected_stages.append({'round':step,'features':fs.copy(),'loss':current,'ranked_improvements':len(candidates),'tested_limit':8})
        accepted = None
        for value,support,removed,added in candidates[:8]:
            if check_attempt({'source':'swap','round':step,'remove':removed,'add':added},list(support),value):
                accepted = list(support); current = value; break
        if accepted is None: break
        fs = accepted
    assert next(attempts,None) is None
    equivalent(path['stages'],expected_stages); assert path['features'] == fs and len(fs) == count
    equivalent(path['candidate'],geometry(fs)); equivalent(path['initial_loss'],initial); equivalent(path['final_loss'],current)
    assert current <= initial+1e-10 and path['candidate']['certificate']['feasible']


def audit(name):
    threadpool_limits(1); counts = Counter(); artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text())
    source = source_verified(name); native = verified_native(name,source); assert saved['complete']
    for rel,value in saved['fingerprints'].items(): assert digest(ROOT/rel) == value
    fingerprints = {p.name:digest(p) for p in [artifact,Path(__file__),ROOT/'scripts/audit_forest_anova_dp_pilot.py',
        ROOT/'scripts/spline_forest_audit_oracle.py',ROOT/'scripts/audit_nested_geometry_forest.py']}
    checkpoint = FOLDER/(name+'-audit.json')
    if checkpoint.exists():
        prior = json.loads(checkpoint.read_text())
        if prior['fingerprints'] == fingerprints: return prior['result']
    X,y = load_dataset(name); expected = [(i,j,s,inner) for i,s in enumerate(source['splits']) for j,inner in enumerate(s['inner'])]
    assert len(expected) == len(saved['records'])
    for r,(i,j,s,inner) in zip(saved['records'],expected):
        assert r['outer_index'] == i and r['inner_index'] == j and r['outer_reference_indices'] == s['reference_indices']
        for key in ('reference_indices','validation_indices'): assert r[key] == inner[key]
        ri,a,b = (np.array(r[k]) for k in ('outer_reference_indices','reference_indices','validation_indices'))
        assert set(a).isdisjoint(b) and len(set(a)|set(b)) == len(ri)
        A,B = preprocess_fit(X[ri[a]],X[ri[b]]); labels,truth = y[ri[a]],y[ri[b]]; d = A.shape[1]; fits = r['fits']; stage = inner['stages'][7]
        strip = lambda fit:{k:v for k,v in fit.items() if k != 'knots'}
        assert strip(fits['old']) == inner['models'][stage['policies']['0.8']['model_id']]
        assert strip(fits['full']) == inner['models'][stage['policies']['1.0']['model_id']]
        probes = np.r_[A,np.random.default_rng(278).uniform(size=(100,d))]
        for key in ('empirical','uniform'):
            teacher = r[key+'_teacher']; assert teacher['edges'] == fits['full']['edges'] and teacher['knots'] == fits['full']['knots']
            np.testing.assert_allclose(score(probes,teacher),score(probes,fits['full']),atol=1e-10,rtol=1e-10)
        empirical,law = marginal_integrals(A,labels,r['empirical_teacher']); equivalent(law,r['marginal_law'])
        uniform = integrated_energies(r['uniform_teacher'])
        for key,energy in [('empirical',empirical),('uniform',uniform)]:
            for field,value in energy.items(): equivalent(r[key+'_energy'][field],value)
            counts['independent_component_integrals'] += d+len(fits['full']['edges'])
        original = np.ones(d)/np.sqrt(d)
        protected = MajorityTerms(A,labels,weights=original).bounds(range(d),allowed_enemies=2) > 1e-10
        cache = {}
        def geometry(fs):
            key = tuple(fs)
            if key not in cache:
                weights = original if len(fs) == d else capped_sparse_projection(transfer_weights(A,fs,original)/np.sqrt(d),len(fs),2/d)*np.sqrt(d)
                embedded = np.zeros(d); embedded[fs] = weights
                cache[key] = {'features':fs,'weights':embedded.tolist(),
                    'certificate':weight_certificate(A,labels,fs,original,weights,protected=protected)}
                counts['independent_geometry_replays'] += 1
            return cache[key]
        target = len(fits['old']['features']); assert r['target_count'] == target
        for key in ('empirical','uniform'):
            path = r['paths'][key]; assert path['initial_features'] == fits['old']['features']
            # Use independently validated stored energies for deterministic float
            # tie replay; integration arithmetic need not be bit-identical.
            replay_path(path,r[key+'_energy'],geometry,counts)
            assert fits[key+'_refit']['features'] == path['features']
            counts[key+'_changed'] += path['features'] != fits['old']['features']
            counts[key+'_attempts'] += len(path['attempts']); counts[key+'_rejections'] += sum(not a['accepted'] for a in path['attempts'])
        for method,ci,prefix in [('fgs2023',4,'fgs'),('sfrip2025',3,'sfrip')]:
            sourcepath = native['splits'][i]['inner'][j]['paths'][method][ci]; path = r['selector_paths'][method]
            assert path['selector_index'] == ci and path['config'] == GRIDS[method][ci] and path['ranking'] == sourcepath['ranking']
            assert path['native_features'] == sorted(sourcepath['native']['features']) and path['matched_features'] == sorted(sourcepath['ranking'][:target])
            for suffix in ('native','matched'): assert fits[prefix+'_'+suffix]['features'] == path[suffix+'_features']
        exact = all(len(fits[m]['features']) == target for m in ('empirical_refit','uniform_refit','fgs_matched','sfrip_matched'))
        assert r['counts_exact'] == exact; counts['count_mismatches'] += not exact
        assert r['query_seed'] == 144+i; rng = np.random.default_rng(144+i)
        queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        for method,fit in fits.items():
            np.testing.assert_allclose(fit['knots'],np.tile([0.,.5,1.],(d,1)),atol=1e-12)
            assert fit['C'] == fit['pair_penalty'] == 1. and fit['edges'] == independent_tree(A,labels,fit['features'],fit['knots'])
            check_model(A,labels,strip(fit),Counter()); counts['independent_classifier_refits'] += 1
            counts['optimizer_caps'] += not fit['optimizer_converged']
            values = [BA(score(Q,fit),fit['classes'],truth) for Q in queries]
            pair,_,_ = leaf_margins(B,truth,fit,.01); margins = pair.min(axis=1); box = r['boxes'][method]
            np.testing.assert_allclose(box['margins'],margins,atol=1e-9,rtol=1e-9)
            rivals = np.array(box['rivals']); assert rivals.tolist() == pair.argmin(axis=1).tolist()
            witness = np.array(box['witness_raw']); assert np.all(witness >= np.maximum(0,B-.01)-1e-12) and np.all(witness <= np.minimum(1,B+.01)+1e-12)
            ws = score(witness,fit); own = np.argmax(truth[:,None] == np.array(fit['classes'])[None],axis=1)
            np.testing.assert_allclose(ws[np.arange(len(B)),own]-ws[np.arange(len(B)),rivals],margins,atol=1e-9,rtol=1e-9)
            for rival,value in lp_margins_first(B,truth,fit,.01).items():
                np.testing.assert_allclose(value,pair[0,rival],atol=1e-8,rtol=1e-8); counts['independent_box_LP'] += 1
            equivalent(r['metrics'][method],{'clean':values[0],'noise':float(np.mean(values[1:])),
                'robust':coverage(margins > 1e-10,truth),'removed':1-len(fit['features'])/d})
            counts['independent_leaf_replays'] += 1; counts['raw_witness_replays'] += len(B); counts['query_predictions'] += 4
        counts['records'] += 1; print(name,i,j,'empirical fixed-count audit passed',flush=True)
    result = {'dataset':name,'counts':dict(counts),'methods':{m:{metric:float(np.mean([r['metrics'][m][metric] for r in saved['records']]))
        for metric in ('clean','noise','robust','removed')} for m in METHODS}}
    checkpoint.write_text(json.dumps({'fingerprints':fingerprints,'result':result},indent=2),encoding='utf-8'); return result

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args()
    summary = None if args.only else json.loads((FOLDER/'complete.json').read_text())
    names = [args.only] if args.only else [r['dataset'] for r in summary['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = [f.result() for f in as_completed([pool.submit(audit,n) for n in names])]
    if args.only: print(rows,flush=True); return
    lookup = {r['dataset']:r for r in rows}; rows = [lookup[n] for n in names]
    equivalent(summary['rows'],[{'dataset':r['dataset'],'methods':r['methods']} for r in rows]); counts = sum((Counter(r['counts']) for r in rows),Counter())
    for k,v in summary['counts'].items(): assert counts[k] == v
    assert counts['records'] == 128
    for m in METHODS:
        for metric in ('clean','noise','robust','removed'): equivalent(summary['means'][m][metric],float(np.mean([r['methods'][m][metric] for r in rows])))
    for metric in ('clean','noise','robust'):
        for m in METHODS:
            if m != 'empirical_refit': equivalent(summary['contrasts'][metric][m],paired([r['methods']['empirical_refit'][metric]-r['methods'][m][metric] for r in rows]))
    clean = summary['contrasts']['clean']; corrected = holm({m:clean[m]['sign_flip_two_sided_p'] for m in ('full','fgs_matched','sfrip_matched')}); equivalent(summary['matched_holm'],corrected)
    gate = all(clean[m]['mean_pp'] >= .5 and corrected[m] < .05 for m in ('fgs_matched','sfrip_matched')) and clean['full']['mean_pp'] >= 0 and summary['means']['empirical_refit']['removed'] >= .1 and counts['count_mismatches'] == counts['optimizer_caps'] == 0
    assert summary['development_numeric_gate'] == gate and summary['goal_achieved'] is False
    result = {'passed':True,'counts':dict(counts),'auditor_sha256':digest(Path(__file__)),
        'scope':'independent_marginal_integrals_global_optima_all_attempt_certificates_deterministic_paths_all_refits_query_boxes',
        'limits':'Empirical_product_surrogate_not_joint_risk; fixed_baseline_settings; exposed_development'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)

if __name__ == '__main__': main()
