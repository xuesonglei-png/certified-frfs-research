"""Independent leaf elimination, sparse marginal LP, witnesses and bounds."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from threadpoolctl import threadpool_limits
from audit_nested_geometry_forest import ROOT,score,BA,membership,equivalent,load_dataset,preprocess_fit
from audit_task_reference_geometry_pilot import raw_probe_sensitivity
from run_forest_robustness_curve import METHODS,CERTIFICATES,RADII,area,coverage,source_verified,matched_verified
from run_nested_geometry_forest import holm
from review_box_iterations import paired

FOLDER = ROOT/'results/forest-robustness-curve'


def independent_boxes(X,y,fit,eps):
    """Different elimination order, direct raw memberships, all-query replay."""
    q,d = len(fit['classes']),X.shape[1]; W,V = np.array(fit['coef']),np.array(fit['pair_coef']).reshape(q,len(fit['edges']),3,3)
    b = np.array(fit['intercept']); own = np.argmax(y[:,None] == np.array(fit['classes'])[None],axis=1)
    lo,hi = np.maximum(0,X-eps),np.minimum(1,X+eps)
    points = np.stack([lo,hi,np.clip(.5,lo,hi)],axis=-1)
    Z = np.stack([np.maximum(1-2*points,0),1-abs(2*points-1),np.maximum(2*points-1,0)],axis=-1)
    unary = np.einsum('nqdk,ndsk->nqds',W[own,None]-W[None],Z)
    base = b[own,None]-b[None]; conservative = base+unary.min(axis=-1).sum(axis=-1)
    edges = {}; neighbors = {a:set() for a in range(d)}
    for e,(u,v) in enumerate(fit['edges']):
        energy = np.einsum('nqkl,nsk,ntl->nqst',V[own,e][:,None]-V[None,:,e],Z[:,u],Z[:,v])
        edges[(u,v)] = energy; neighbors[u].add(v); neighbors[v].add(u)
        conservative += energy.min(axis=(-1,-2))
    cost = {a:unary[:,:,a,:].copy() for a in range(d)}; exact = base.copy()
    active = set(range(d))
    while active:
        leaves = sorted(a for a in active if len(neighbors[a]) <= 1)
        assert leaves, 'cycle in supposedly forest interaction graph'
        for a in leaves:
            if neighbors[a]:
                other = next(iter(neighbors[a])); key = tuple(sorted((a,other)))
                energy = edges[key] if key[0] == a else edges[key].swapaxes(-1,-2)
                cost[other] += (cost[a][:,:,:,None]+energy).min(axis=-2)
                neighbors[other].remove(a); neighbors[a].clear()
            else: exact += cost[a].min(axis=-1)
            active.remove(a)
    exact[np.arange(len(X)),own] = np.inf; conservative[np.arange(len(X)),own] = np.inf
    return exact,conservative,points,Z


def lp_first_query(fit,label,points,Z):
    """Independent integral local-polytope LP for all rivals of first query."""
    d = len(points); edges = fit['edges']; count = 3*d+9*len(edges)
    rows,cols,values,rhs = [],[],[],[]
    def add(entries,value):
        row = len(rhs); rhs.append(value)
        for col,coef in entries: rows.append(row); cols.append(col); values.append(coef)
    for a in range(d): add([(3*a+k,1.) for k in range(3)],1.)
    for e,(u,v) in enumerate(edges):
        start = 3*d+9*e
        for k in range(3):
            add([(start+3*k+s,1.) for s in range(3)]+[(3*u+k,-1.)],0.)
            add([(start+3*s+k,1.) for s in range(3)]+[(3*v+k,-1.)],0.)
    matrix = coo_matrix((values,(rows,cols)),shape=(len(rhs),count)).tocsr()
    W,V,b = np.array(fit['coef']),np.array(fit['pair_coef']),np.array(fit['intercept'])
    own = fit['classes'].index(int(label)); results = {}
    for rival in range(len(fit['classes'])):
        if rival == own: continue
        objective = np.zeros(count)
        objective[:3*d] = np.einsum('dsk,dk->ds',Z,W[own]-W[rival]).ravel()
        for e,(u,v) in enumerate(edges):
            objective[3*d+9*e:3*d+9*(e+1)] = np.einsum('sk,kl,tl->st',Z[u],V[own,e]-V[rival,e],Z[v]).ravel()
        solved = linprog(objective,A_eq=matrix,b_eq=np.array(rhs),bounds=(0,None),method='highs',
            options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'ipm_optimality_tolerance':1e-10})
        assert solved.success
        results[rival] = float(solved.fun+b[own]-b[rival])
    return results


def audit(name):
    threadpool_limits(1); counts = Counter(); artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text())
    source,matched = source_verified(name),matched_verified(name)
    assert saved['complete'] and saved['radii'] == list(RADII)
    for rel,value in saved['fingerprints'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == value
    fingerprints = {'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    checkpoint = FOLDER/(name+'-audit.json')
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        if previous['fingerprints'] == fingerprints: return previous['result']
    X,y = load_dataset(name)
    assert len(saved['splits']) == len(source['splits']) == len(matched['splits'])
    for split,original,control in zip(saved['splits'],source['splits'],matched['splits']):
        for key in ('reference_indices','validation_indices'): assert split[key] == original[key] == control[key]
        ri,vi = np.array(split['reference_indices']),np.array(split['validation_indices'])
        # Archived benchmark uses a train/query subsample, not necessarily
        # every row of the downloaded dataset. Source audit fixes its identity.
        assert set(ri).isdisjoint(vi) and all(0 <= k < len(X) for k in np.r_[ri,vi])
        assert len(set(ri)) == len(ri) and len(set(vi)) == len(vi)
        _,B = preprocess_fit(X[ri],X[vi]); truth = y[vi]; d = X.shape[1]
        expected = {m:original['models'][original['results'][m]['model_id']] for m in METHODS[:4]}
        expected.update({m+'_matched':control['results'][m]['fit'] for m in ('fgs2023','sfrip2025')})
        assert split['results'].keys() == expected.keys()
        for method,record in split['results'].items():
            fit = record['fit']; assert fit == expected[method] and record['features'] == fit['features']
            counts['verified_source_models'] += 1; counts['optimizer_caps'] += not fit['optimizer_converged']
            if method.endswith('_matched'): assert len(fit['features']) == len(expected['proposed']['features'])
            clean_scores = score(B,fit); clean = BA(clean_scores,fit['classes'],truth); equivalent(record['clean'],clean)
            sensitivity = raw_probe_sensitivity(fit); q = len(fit['classes']); ell = np.zeros((q,q,d))
            for pair,constants in zip(sensitivity['class_index_pairs'],sensitivity['contrast_constants']):
                c,r = pair; ell[c,r] = ell[r,c] = constants
            own = np.argmax(truth[:,None] == np.array(fit['classes'])[None],axis=1)
            assert len(record['curves']) == len(RADII)
            for row,eps in zip(record['curves'],RADII):
                assert row['radius'] == eps
                exact,independent,points,Z = independent_boxes(B,truth,fit,eps)
                lo,hi = np.maximum(0,B-eps),np.minimum(1,B+eps)
                lip = clean_scores[np.arange(len(B)),own,None]-clean_scores-np.einsum('nqd,nd->nq',ell[own],np.maximum(B-lo,hi-B))
                lip[np.arange(len(B)),own] = np.inf
                vectors = {'exact':exact.min(axis=1),'independent':independent.min(axis=1),'lipschitz':lip.min(axis=1)}
                for k,v in vectors.items():
                    np.testing.assert_allclose(row['margins'][k],v,atol=1e-9,rtol=1e-9)
                    equivalent(row['coverage'][k],coverage(v > 1e-10,truth))
                pair_saved = np.array([[np.inf if v is None else v for v in row] for row in row['pair_margins']])
                np.testing.assert_allclose(pair_saved,exact,atol=1e-9,rtol=1e-9)
                rivals = np.array(row['rivals']); assert rivals.tolist() == exact.argmin(axis=1).tolist()
                witness = np.array(row['witness_raw']); assert witness.shape == B.shape
                assert np.all(witness >= lo-1e-12) and np.all(witness <= hi+1e-12)
                witness_scores = score(witness,fit); attained = witness_scores[np.arange(len(B)),own]-witness_scores[np.arange(len(B)),rivals]
                np.testing.assert_allclose(attained,vectors['exact'],atol=1e-9,rtol=1e-9)
                equivalent(row['negative_witness_rate'],coverage(attained < -1e-10,truth))
                for rival,value in lp_first_query(fit,int(truth[0]),points[0],Z[0]).items():
                    np.testing.assert_allclose(value,exact[0,rival],atol=1e-8,rtol=1e-8); counts['independent_lp_solves'] += 1
                assert np.all(vectors['exact'] >= vectors['independent']-1e-9)
                assert np.all(vectors['exact'] >= vectors['lipschitz']-1e-9)
                counts['independent_leaf_elimination_replays'] += 1; counts['witness_score_replays'] += len(B)
                counts['radius_evaluations'] += 1
            for k in CERTIFICATES:
                equivalent(record['area'][k],area([r['coverage'][k] for r in record['curves']]))
                values = np.array([r['coverage'][k] for r in record['curves']]); assert np.all(np.diff(values) <= 1e-12)
            tie_gap = clean-record['curves'][0]['coverage']['exact']; equivalent(record['radius_zero_tie_gap'],tie_gap)
            counts['zero_radius_nonzero_tie_gaps'] += abs(tie_gap) > 1e-12
            counts['models'] += 1
        counts['outer_splits'] += 1; print(name,counts['outer_splits'],'robustness audit passed',flush=True)
    methods = {m:{'clean':float(np.mean([s['results'][m]['clean'] for s in saved['splits']])),
        'removed':float(np.mean([1-len(s['results'][m]['features'])/saved['d'] for s in saved['splits']])),
        'area':{k:float(np.mean([s['results'][m]['area'][k] for s in saved['splits']])) for k in CERTIFICATES},
        'curves':{k:[float(np.mean([s['results'][m]['curves'][i]['coverage'][k] for s in saved['splits']])) for i in range(len(RADII))] for k in CERTIFICATES}} for m in METHODS}
    result = {'dataset':name,'methods':methods,'counts':dict(counts)}
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
    assert counts['outer_splits'] == 44 and counts['models'] == 264 and counts['radius_evaluations'] == 1320
    for m in METHODS:
        for metric in ('clean','removed'): equivalent(summary['means'][m][metric],float(np.mean([r['methods'][m][metric] for r in rows])))
        for k in CERTIFICATES:
            equivalent(summary['means'][m]['area'][k],float(np.mean([r['methods'][m]['area'][k] for r in rows])))
            equivalent(summary['means'][m]['curves'][k],np.mean([r['methods'][m]['curves'][k] for r in rows],axis=0).tolist())
        for k in CERTIFICATES[1:]: equivalent(summary['same_classifier_certificate_gain'][m][k],paired([r['methods'][m]['area']['exact']-r['methods'][m]['area'][k] for r in rows]))
    for m in METHODS[1:]: equivalent(summary['exact_method_gain'][m],paired([r['methods']['proposed']['area']['exact']-r['methods'][m]['area']['exact'] for r in rows]))
    equivalent(summary['exact_method_gain_holm'],holm({m:r['sign_flip_two_sided_p'] for m,r in summary['exact_method_gain'].items()}))
    primary = summary['same_classifier_certificate_gain']['proposed']['independent']
    assert summary['diagnostic_gate_passed'] == (primary['mean_pp'] >= 1 and primary['sign_flip_two_sided_p'] < .05)
    assert summary['goal_achieved'] is False
    result = {'passed':True,'counts':dict(counts),'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'all_query_independent_leaf_elimination_all_witness_scores_conservative_bounds_first_query_integral_LP',
        'limits':'Unchanged_fixed_classifiers; exposed_development_numeric_boxes_not_population_or_clean_SOTA_claim'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)


if __name__ == '__main__': main()
