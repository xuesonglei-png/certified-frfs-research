"""Independent integration/MILP FRS screening and fixed/refitted scores."""
import argparse
import itertools
import hashlib
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified
from run_nested_geometry_forest import GRIDS,verified_native,holm
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.capped_sparse import capped_sparse_projection
from audit_nested_geometry_forest import BA,check_model
from audit_forest_compact_transfer import equivalent
from audit_spline_capacity_pilot import independent_tree
from spline_forest_audit_oracle import membership,score,leaf_margins,lp_margins_first
from run_forest_robustness_curve import coverage
from review_box_iterations import paired

FOLDER = ROOT/'results/forest-anova-dp-pilot'
METHODS = ('old','full','dp_refit','dp_projected','fgs_native','sfrip_native','fgs_matched','sfrip_matched')


def quadrature(knots):
    nodes,weights = np.polynomial.legendre.leggauss(2); points = []; masses = []
    for lo,hi in zip(knots[:-1],knots[1:]):
        points.extend(((lo+hi)/2+nodes*(hi-lo)/2).tolist()); masses.extend((weights*(hi-lo)/2).tolist())
    Z = np.column_stack([np.interp(points,knots,np.eye(len(knots))[j]) for j in range(len(knots))])
    return Z,np.array(masses)


def integrated_energies(fit):
    W = np.array(fit['coef']); q,d,k = W.shape; V = np.array(fit['pair_coef']).reshape(q,len(fit['edges']),k,k)
    W -= W.mean(axis=0,keepdims=True); V -= V.mean(axis=0,keepdims=True)
    quads = [quadrature(t) for t in fit['knots']]; unaries = []; pairs = []
    for a,(Z,mass) in enumerate(quads):
        values = Z@W[:,a].T; np.testing.assert_allclose(mass@values,0,atol=1e-9)
        unaries.append(float(mass@np.mean(values**2,axis=1)))
    for e,(u,v) in enumerate(fit['edges']):
        Zu,mu = quads[u]; Zv,mv = quads[v]; values = np.einsum('sk,qkl,tl->qst',Zu,V[:,e],Zv)
        np.testing.assert_allclose(np.einsum('s,qst->qt',mu,values),0,atol=1e-9)
        np.testing.assert_allclose(values@mv,0,atol=1e-9)
        pairs.append(float(mu@np.mean(values**2,axis=0)@mv))
    return {'unary_energy':unaries,'pair_energy':pairs,'edges':fit['edges'],'q':q,'total_energy':float(sum(unaries)+sum(pairs))}


def independent_optimum(energies,count):
    unary,pair,edges = np.array(energies['unary_energy']),np.array(energies['pair_energy']),energies['edges']; d = len(unary)
    if d <= 13:
        values = [(float(sum(unary[a] for a in fs)+sum(pair[e] for e,(u,v) in enumerate(edges) if u in fs and v in fs)),fs)
            for fs in itertools.combinations(range(d),count)]
        best = max(v for v,_ in values); supports = [fs for value,fs in values if value == best]
        return best,list(min(supports)),'exhaustive'
    total = d+len(edges); rows,cols,values,lo,hi = [],[],[],[],[]
    def add(entries,left,right):
        row = len(lo); lo.append(left); hi.append(right)
        for c,v in entries: rows.append(row); cols.append(c); values.append(v)
    add([(a,1.) for a in range(d)],count,count)
    for e,(u,v) in enumerate(edges):
        z = d+e
        add([(z,1.),(u,-1.)],-np.inf,0.)
        add([(z,1.),(v,-1.)],-np.inf,0.)
        add([(z,1.),(u,-1.),(v,-1.)],-1.,np.inf)
    matrix = coo_matrix((values,(rows,cols)),shape=(len(lo),total)).tocsc()
    result = milp(-np.r_[unary,pair],integrality=np.ones(total),bounds=Bounds(0,1),
        constraints=LinearConstraint(matrix,np.array(lo),np.array(hi)),options={'mip_rel_gap':0.,'time_limit':30.})
    assert result.success and abs(result.mip_gap) < 1e-12
    support = np.flatnonzero(result.x[:d] > .5).tolist(); assert len(support) == count
    return float(-result.fun),support,'MILP'


def conditional_raw_score(X,source,features):
    """Integrate omitted coordinates directly in source unary/pair terms."""
    fs = set(features); Z = membership(X,source['knots']); q,d,k = np.array(source['coef']).shape
    W,V = np.array(source['coef']),np.array(source['pair_coef']).reshape(q,len(source['edges']),k,k)
    means = [mass@M for M,mass in (quadrature(t) for t in source['knots'])]
    conditional = np.stack([Z[:,a] if a in fs else np.tile(means[a],(len(X),1)) for a in range(d)],axis=1)
    result = np.einsum('ndk,qdk->nq',conditional,W)+np.array(source['intercept'])
    for e,(u,v) in enumerate(source['edges']): result += np.einsum('nk,qkl,nl->nq',conditional[:,u],V[:,e],conditional[:,v])
    return result


def audit(name):
    threadpool_limits(1); counts = Counter(); artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text())
    source = source_verified(name); native = verified_native(name,source); assert saved['complete']
    for rel,value in saved['fingerprints'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == value
    fingerprints = {'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'oracle_sha256':hashlib.sha256((ROOT/'scripts/spline_forest_audit_oracle.py').read_bytes()).hexdigest()}
    checkpoint = FOLDER/(name+'-audit.json')
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        if previous['fingerprints'] == fingerprints: return previous['result']
    X,y = load_dataset(name); expected = [(i,j,s,inner) for i,s in enumerate(source['splits']) for j,inner in enumerate(s['inner'])]
    assert len(expected) == len(saved['records'])
    for r,(i,j,s,inner) in zip(saved['records'],expected):
        assert r['outer_index'] == i and r['inner_index'] == j and r['outer_reference_indices'] == s['reference_indices']
        for key in ('reference_indices','validation_indices'): assert r[key] == inner[key]
        ri,a,b = (np.array(r[k]) for k in ('outer_reference_indices','reference_indices','validation_indices'))
        assert set(a).isdisjoint(b) and len(set(a)|set(b)) == len(ri)
        A,B = preprocess_fit(X[ri[a]],X[ri[b]]); labels,truth = y[ri[a]],y[ri[b]]; d = A.shape[1]; fits = r['fits']
        stage = inner['stages'][7]
        strip = lambda fit:{k:v for k,v in fit.items() if k != 'knots'}
        assert strip(fits['old']) == inner['models'][stage['policies']['0.8']['model_id']]
        assert strip(fits['full']) == inner['models'][stage['policies']['1.0']['model_id']]
        teacher = r['teacher']; assert teacher['edges'] == fits['full']['edges'] and teacher['knots'] == fits['full']['knots']
        rng = np.random.default_rng(273); probes = np.r_[A,rng.uniform(size=(100,d))]
        np.testing.assert_allclose(score(probes,teacher),score(probes,fits['full']),atol=1e-10,rtol=1e-10)
        energies = integrated_energies(teacher)
        for key,value in energies.items(): equivalent(r['energies'][key],value)
        counts['independent_energy_integrals'] += d+len(teacher['edges'])
        original = np.ones(d)/np.sqrt(d); protected = MajorityTerms(A,labels,weights=original).bounds(range(d),allowed_enemies=2) > 1e-10
        for index,candidate in enumerate(r['candidates']):
            dp,geometry = candidate['dp'],candidate['candidate']; count = int(np.ceil(.8*d))+index
            assert dp['count'] == count and len(dp['features']) == count and geometry['features'] == dp['features']
            optimum,optfs,method = independent_optimum(energies,count)
            keep = set(dp['features'])
            retained = sum(v for a,v in enumerate(energies['unary_energy']) if a in keep)+sum(v for (u,w),v in zip(energies['edges'],energies['pair_energy']) if u in keep and w in keep)
            np.testing.assert_allclose(retained,optimum,atol=1e-9,rtol=1e-9)
            equivalent(dp['retained_energy'],retained); equivalent(dp['omission_energy'],max(0.,energies['total_energy']-retained))
            counts['independent_'+method+'_optima'] += 1; counts['total_dp_candidates'] += 1
            fs = dp['features']; weights = original if count == d else capped_sparse_projection(transfer_weights(A,fs,original)/np.sqrt(d),count,2/d)*np.sqrt(d)
            embedded = np.zeros(d); embedded[fs] = weights
            equivalent(geometry,{'features':fs,'weights':embedded.tolist(),'certificate':weight_certificate(A,labels,fs,original,weights,protected=protected)})
            assert geometry['certificate']['feasible'] == (index == len(r['candidates'])-1)
            counts['independent_geometry_replays'] += 1
        final = r['candidates'][-1]['candidate']; fs = final['features']; assert r['target_count'] == len(fs)
        assert fits['dp_refit']['features'] == fits['dp_projected']['features'] == fs
        np.testing.assert_allclose(score(probes,fits['dp_projected']),conditional_raw_score(probes,fits['full'],fs),atol=1e-10,rtol=1e-10)
        for key,support in [('old',fits['old']['features']),('selected_dp',fs)]:
            keep = set(support); value = sum(v for a,v in enumerate(energies['unary_energy']) if a not in keep)+sum(v for (u,w),v in zip(energies['edges'],energies['pair_energy']) if u not in keep or w not in keep)
            equivalent(r['uniform_omission_losses'][key],value)
        for method,ci,prefix in [('fgs2023',4,'fgs'),('sfrip2025',3,'sfrip')]:
            sourcepath = native['splits'][i]['inner'][j]['paths'][method][ci]; path = r['selector_paths'][method]
            assert path['selector_index'] == ci and path['config'] == GRIDS[method][ci] and path['ranking'] == sourcepath['ranking']
            assert path['native_features'] == sorted(sourcepath['native']['features']) and path['matched_features'] == sorted(sourcepath['ranking'][:len(fs)])
            assert fits[prefix+'_native']['features'] == path['native_features'] and fits[prefix+'_matched']['features'] == path['matched_features']
        exact = all(len(fits[m]['features']) == len(fs) for m in ('fgs_matched','sfrip_matched'))
        assert r['counts_exact'] == exact; counts['count_mismatches'] += not exact
        counts['same_old_supports'] += fits['old']['features'] == fs; counts['full_fallbacks'] += len(fs) == d
        counts['restored_count_references'] += len(r['candidates']) > 1
        assert r['query_seed'] == 144+i; rng = np.random.default_rng(144+i)
        queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        for method,fit in fits.items():
            np.testing.assert_allclose(fit['knots'],np.tile([0.,.5,1.],(d,1)),atol=1e-12)
            assert fit['C'] == fit['pair_penalty'] == 1.
            if method != 'dp_projected':
                ffs = fit['features']; assert fit['edges'] == independent_tree(A,labels,ffs,fit['knots'])
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
        counts['records'] += 1; print(name,i,j,'anova DP audit passed',flush=True)
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
            if m != 'dp_refit': equivalent(summary['contrasts'][metric][m],paired([r['methods']['dp_refit'][metric]-r['methods'][m][metric] for r in rows]))
    clean = summary['contrasts']['clean']; corrected = holm({m:clean[m]['sign_flip_two_sided_p'] for m in ('full','fgs_matched','sfrip_matched')}); equivalent(summary['matched_holm'],corrected)
    gate = all(clean[m]['mean_pp'] >= .5 and corrected[m] < .05 for m in ('fgs_matched','sfrip_matched')) and clean['full']['mean_pp'] >= 0 and summary['means']['dp_refit']['removed'] >= .1 and counts['count_mismatches'] == counts['optimizer_caps'] == 0
    assert summary['gate_passed'] == gate and summary['goal_achieved'] is False
    result = {'passed':True,'counts':dict(counts),'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'independent_quadrature_energy_MILP_exhaustive_optimum_conditional_scores_FRS_allrefits_boxLP_witnesses',
        'limits':'Uniform_model_reconstruction_surrogate_not_global_FRS_optimum_or_UCI_population_prediction_guarantee'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)


if __name__ == '__main__': main()
