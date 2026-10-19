"""Fresh raw-basis logistic fits, bootstrap isolation, mean logits and LP probes."""
import argparse,json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from screen_bagged_fixed_forest import FOLDER
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.nested_teacher_regularized import inner_splits
from certified_frfs.triangular_classifier import triangular_memberships
from certified_frfs.fuzzy_interaction_tree import interaction_tree
from certified_frfs.forest_fuzzy_classifier import forest_scores,forest_box_margins
from audit_fuzzy_forest_capacity import lp_margin
from audit_forest_compact_transfer import equivalent

CONTRAST = np.array([[1/np.sqrt(2),1/np.sqrt(6)],[-1/np.sqrt(2),1/np.sqrt(6)],[0.,-2/np.sqrt(6)]])


def raw_design(X,edges):
    Z = triangular_memberships(X)@CONTRAST
    return np.column_stack([Z.reshape(len(X),-1)]+[(Z[:,u,:,None]*Z[:,v,None,:]).reshape(len(X),4) for u,v in edges])


def audit(name):
    threadpool_limits(1); source = source_verified(name); artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text())
    assert saved['complete'] and not saved['archived_query_labels_used']
    for rel,sha in saved['fingerprints'].items(): assert digest(ROOT/rel) == sha
    fp = {'artifact_sha256':digest(artifact),'auditor_sha256':digest(Path(__file__))}; checkpoint = FOLDER/(name+'-audit.json')
    if checkpoint.exists():
        prior = json.loads(checkpoint.read_text())
        if prior['fingerprints'] == fp: return prior['result']
    X,y = load_dataset(name); counts = Counter(); maxima = {'score_error':0.,'LP_error':0.}; reference_metrics = []
    expected = [(i,j,s,inner) for i,s in enumerate(source['splits']) for j,inner in enumerate(s['inner'])]; assert len(saved['rows']) == len(expected)
    for r,(i,j,s,inner) in zip(saved['rows'],expected):
        assert r['outer'] == i and r['inner'] == j and r['outer_reference_indices'] == s['reference_indices'] and r['reference_indices'] == inner['reference_indices']
        ri,a = np.array(s['reference_indices']),np.array(inner['reference_indices']); raw,labels = X[ri[a]],y[ri[a]]; d = raw.shape[1]
        splits = inner_splits(labels,seed=20261027); assert len(splits) == len(r['folds']); evaluated = []
        for fi,((ca,cb),fold) in enumerate(zip(splits,r['folds'])):
            assert fold['reference_indices'] == ca.tolist() and fold['validation_indices'] == cb.tolist() and set(ca).isdisjoint(cb)
            A,B = preprocess_fit(raw[ca],raw[cb]); train,truth = labels[ca],labels[cb]; classes = np.unique(train); q = len(classes)
            tree = interaction_tree(A,train); equivalent(fold['tree'],tree); edges = tree['edges']; design,query = raw_design(A,edges),raw_design(B,edges)
            bag = fold['bag']; assert bag['seed'] == 20261027 and len(bag['members']) == 12 and len(bag['bootstrap_indices']) == 12
            rng = np.random.default_rng(20261027); groups = [np.flatnonzero(train == c) for c in classes]
            indices = [np.concatenate([rng.choice(ids,len(ids),replace=True) for ids in groups]).tolist() for _ in range(12)]
            assert bag['bootstrap_indices'] == indices
            for ids in indices:
                assert set(ids) <= set(range(len(A))) and all(sum(train[ids] == c) == sum(train == c) for c in classes)
            def refit(ids,fit,kind):
                assert fit['classes'] == classes.tolist() and fit['features'] == list(range(d)) and fit['edges'] == edges
                assert fit['C'] == fit['pair_penalty'] == 1. and fit['class_balanced'] and not fit['teacher_guards_enforced']
                model = LogisticRegression(C=1.,class_weight='balanced',max_iter=4000,tol=1e-8,random_state=0).fit(design[ids],train[ids])
                if q == 2:
                    beta = np.vstack([np.zeros(design.shape[1]),model.coef_[0]]); bias = np.array([0.,model.intercept_[0]])
                else: beta,bias = model.coef_.copy(),model.intercept_.copy()
                W = (beta[:,:2*d].reshape(q,d,2)@CONTRAST.T)
                V = np.array([np.einsum('ki,qij,lj->qkl',CONTRAST,beta[:,2*d+4*e:2*d+4*(e+1)].reshape(q,2,2),CONTRAST) for e in range(len(edges))])
                V = np.moveaxis(V,0,1) if len(edges) else np.zeros((q,0,3,3))
                W -= W[:1].copy(); V -= V[:1].copy(); bias -= bias[:1].copy()
                np.testing.assert_allclose(fit['coef'],W,atol=1e-10); np.testing.assert_allclose(np.array(fit['pair_coef']).reshape(q,len(edges),3,3),V,atol=1e-10)
                np.testing.assert_allclose(fit['intercept'],bias,atol=1e-10)
                iterations = int(model.n_iter_.max()); assert fit['optimizer_iterations'] == iterations and fit['optimizer_converged'] == (iterations < 4000)
                counts[kind+'_fits'] += 1; counts[kind+'_caps'] += iterations >= 4000
                raw_scores = query@beta.T
                if q == 2: raw_scores += [0.,model.intercept_[0]]
                else: raw_scores += model.intercept_
                raw_scores -= raw_scores[:,:1].copy()
                stored = forest_scores(B,fit['coef'],fit['intercept'],edges,np.array(fit['pair_coef']).reshape(q,len(edges),3,3))
                maxima['score_error'] = max(maxima['score_error'],float(np.max(abs(raw_scores-stored))))
                np.testing.assert_allclose(stored,raw_scores,atol=1e-10); return raw_scores
            single = refit(list(range(len(A))),fold['single'],'single')
            member_scores = [refit(ids,fit,'member') for ids,fit in zip(indices,bag['members'])]
            merged = bag['fit']
            for key in ('features','edges','classes','C','pair_penalty'): assert merged[key] == fold['single'][key]
            for key in ('coef','pair_coef','intercept'):
                np.testing.assert_allclose(merged[key],np.mean([np.array(f[key]) for f in bag['members']],axis=0),atol=1e-12)
            assert merged['optimizer_converged'] == all(f['optimizer_converged'] for f in bag['members'])
            avg = np.mean(member_scores,axis=0); deployed = forest_scores(B,merged['coef'],merged['intercept'],edges,np.array(merged['pair_coef']).reshape(q,len(edges),3,3))
            np.testing.assert_allclose(avg,deployed,atol=1e-10); counts['mean_score_replays'] += 1
            evaluations = {}
            for method,score in [('single',single),('bagged',avg)]:
                pred = classes[score.argmax(axis=1)]; own = np.searchsorted(classes,truth); losses = logsumexp(score,axis=1)-score[np.arange(len(B)),own]
                report = fold['evaluations'][method]; assert report['predictions'] == pred.tolist(); np.testing.assert_allclose(report['scores'],score,atol=1e-10)
                values = {'balanced_accuracy':float(np.mean([np.mean(pred[truth == c] == c) for c in np.unique(truth)])),
                    'balanced_logloss':float(np.mean([np.mean(losses[truth == c]) for c in np.unique(truth)]))}
                for key,value in values.items(): equivalent(report[key],value)
                evaluations[method] = values
            evaluated.append(evaluations); counts['internal_folds'] += 1
            if i == j == fi == 0:
                own = np.searchsorted(classes,truth[:1])[0]; rival = 1 if own == 0 else 0
                oracle = forest_box_margins(B[:1],truth[:1],merged['coef'],merged['intercept'],classes,edges,np.array(merged['pair_coef']).reshape(q,len(edges),3,3),eps=.01,witnesses=True)
                bound = lp_margin(B[0],own,rival,np.array(merged['coef']),np.array(merged['intercept']),edges,np.array(merged['pair_coef']).reshape(q,len(edges),3,3))
                error = abs(bound-oracle['pair_margins'][0,rival]); np.testing.assert_allclose(bound,oracle['pair_margins'][0,rival],atol=1e-7)
                maxima['LP_error'] = max(maxima['LP_error'],float(error)); counts['merged_LP_probes'] += 1
        rebuilt = {m:{k:float(np.mean([f[m][k] for f in evaluated])) for k in ('balanced_accuracy','balanced_logloss')} for m in ('single','bagged')}
        equivalent(r['metrics'],rebuilt); reference_metrics.append(rebuilt); counts['references'] += 1
        counts['positive_BA_references'] += rebuilt['bagged']['balanced_accuracy']-rebuilt['single']['balanced_accuracy'] > 1e-12
        print(name,i,j,'independent bagged classifier audit passed',flush=True)
    result = {'dataset':name,'counts':dict(counts),'precision_maxima':maxima,'metrics':{m:{k:float(np.mean([r[m][k] for r in reference_metrics])) for k in ('balanced_accuracy','balanced_logloss')} for m in ('single','bagged')}}
    checkpoint.write_text(json.dumps({'fingerprints':fp,'result':result},indent=2),encoding='utf-8'); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args()
    if args.only:
        print(audit(args.only),flush=True)
        raise SystemExit(0)
    saved = json.loads((FOLDER/'complete.json').read_text()); names = [r['dataset'] for r in saved['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = [f.result() for f in as_completed([pool.submit(audit,n) for n in names])]
    lookup = {r['dataset']:r for r in rows}; counts = sum((Counter(r['counts']) for r in rows),Counter())
    for k,v in saved['counts'].items(): assert counts[k] == v
    for row in saved['rows']: equivalent(row['metrics'],lookup[row['dataset']]['metrics'])
    means = {m:{k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in ('balanced_accuracy','balanced_logloss')} for m in ('single','bagged')}; equivalent(saved['means'],means)
    gain = 100*(means['bagged']['balanced_accuracy']-means['single']['balanced_accuracy']); loggain = means['single']['balanced_logloss']-means['bagged']['balanced_logloss']
    equivalent(saved['BA_gain_pp'],gain); equivalent(saved['logloss_gain'],loggain)
    gate = gain >= .5 and loggain >= 0 and counts['positive_BA_references'] >= 20 and counts['single_caps'] == counts['member_caps'] == 0
    assert saved['numeric_resource_gate'] == gate and not saved['goal_achieved'] and not saved['archived_query_labels_used']
    result = {'passed':True,'counts':dict(counts),'resource_gate':bool(gate),'BA_gain_pp':gain,'logloss_gain':loggain,
        'precision_maxima':{k:max(r['precision_maxima'][k] for r in rows) for k in ('score_error','LP_error')},'auditor_sha256':digest(Path(__file__)),
        'scope':'fresh_raw_design_member_single_refits_fixedgraph_mean_score_LP_probes_complete_validation_isolation'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)
