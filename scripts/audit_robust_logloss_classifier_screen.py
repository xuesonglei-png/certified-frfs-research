"""Fresh training solver replay, saved-control provenance, score/LP validation."""
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from screen_robust_logloss_classifier import FOLDER,INPUT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.nested_teacher_regularized import inner_splits
from certified_frfs.forest_robust_logloss import fit_robust_forest_logloss
from certified_frfs.forest_fuzzy_classifier import forest_scores,forest_box_margins
from audit_fuzzy_forest_capacity import lp_margin
from audit_forest_compact_transfer import equivalent


def audit(name):
    threadpool_limits(1); source_verified(name); artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text())
    assert saved['complete'] and not saved['archived_query_labels_used']
    for rel,sha in saved['fingerprints'].items(): assert digest(ROOT/rel) == sha
    upstream = json.loads((INPUT/(name+'.json')).read_text()); audit = json.loads((INPUT/(name+'-audit.json')).read_text())
    assert audit['fingerprints']['artifact_sha256'] == digest(INPUT/(name+'.json'))
    fp = {'artifact_sha256':digest(artifact),'auditor_sha256':digest(Path(__file__))}; checkpoint = FOLDER/(name+'-audit.json')
    if checkpoint.exists():
        prior = json.loads(checkpoint.read_text())
        if prior['fingerprints'] == fp: return prior['result']
    r = upstream['rows'][0]; assert r['outer'] == r['inner'] == 0
    assert saved['outer_reference_indices'] == r['outer_reference_indices'] and saved['reference_indices'] == r['reference_indices']
    X,y = load_dataset(name); ri,a = np.array(saved['outer_reference_indices']),np.array(saved['reference_indices']); raw,labels = X[ri[a]],y[ri[a]]
    splits = inner_splits(labels,seed=20261027); assert len(splits) == len(saved['folds']) == len(r['folds'])
    counts = Counter(); maxima = {'fresh_score_error':0.,'LP_error':0.}; evaluations = []
    for fi,((ca,cb),f,old) in enumerate(zip(splits,saved['folds'],r['folds'])):
        assert f['reference_indices'] == old['reference_indices'] == ca.tolist() and f['validation_indices'] == old['validation_indices'] == cb.tolist()
        assert set(ca).isdisjoint(cb) and f['tree'] == old['tree'] and f['ordinary'] == old['single']
        A,B = preprocess_fit(raw[ca],raw[cb]); train,truth = labels[ca],labels[cb]
        fresh = fit_robust_forest_logloss(A,train,list(range(A.shape[1])),f['tree']['edges'],C=1.,eps=.01)
        fit = f['fit']; assert fresh['usable'] == fit['usable'] and fresh['status'] == fit['status']
        assert f['ordinary_fallback'] == (not fit['usable']); counts['fresh_training_solves'] += 1
        counts['fits'] += 1; counts['new_failures'] += f['ordinary_fallback']; counts['ordinary_caps'] += not f['ordinary']['optimizer_converged']
        if fit['usable']:
            assert fit['C'] == fit['pair_penalty'] == 1. and fit['eps'] == .01 and fit['separation_tol'] == 2e-7 and not fit['teacher_guards_enforced']
            for key in ('coef','pair_coef','intercept','beta'):
                np.testing.assert_allclose(fit[key],fresh[key],atol=1e-6,rtol=1e-6)
            np.testing.assert_allclose(fit['exact_envelope_objective'],fresh['exact_envelope_objective'],atol=1e-7,rtol=1e-7)
            classes = np.array(fit['classes']); q = len(classes); own = np.searchsorted(classes,train)
            oracle = forest_box_margins(A,train,fit['coef'],fit['intercept'],classes,fit['edges'],np.array(fit['pair_coef']).reshape(q,len(fit['edges']),3,3),eps=.01,witnesses=True)
            pair = oracle['pair_margins'].copy(); pair[np.arange(len(A)),own] = 0.
            np.testing.assert_allclose(fit['envelope_pair_margins'],pair,atol=1e-10)
            assert np.max(-pair-np.array(fit['upper'])) <= fit['separation_tol']+1e-10
        models = {'ordinary':f['ordinary'],'robust':f['ordinary'] if f['ordinary_fallback'] else fit}; values = {}
        for m,model in models.items():
            classes = np.array(model['classes']); q = len(classes); own = np.searchsorted(classes,truth)
            W,V,b = np.array(model['coef']),np.array(model['pair_coef']).reshape(q,len(model['edges']),3,3),np.array(model['intercept'])
            scores = forest_scores(B,W,b,model['edges'],V); pred = classes[scores.argmax(axis=1)]
            oracle = forest_box_margins(B,truth,W,b,classes,model['edges'],V,eps=.01,witnesses=True)
            pair = oracle['pair_margins'].copy(); pair[np.arange(len(B)),own] = 0.
            nominal = logsumexp(scores,axis=1)-scores[np.arange(len(B)),own]; envelope = logsumexp(-pair,axis=1)
            def balanced(array): return float(np.mean([np.mean(array[truth == c]) for c in np.unique(truth)]))
            metric = {'clean_BA':balanced(pred == truth),'certified_BA':balanced(oracle['margins'] > 1e-10),
                'nominal_logloss':balanced(nominal),'envelope_logloss':balanced(envelope)}
            report = f['evaluations'][m]; assert report['predictions'] == pred.tolist()
            np.testing.assert_allclose(report['scores'],scores,atol=1e-10); np.testing.assert_allclose(report['pair_margins'],pair,atol=1e-10)
            equivalent(report['metrics'],metric); values[m] = metric; counts['validation_score_replays'] += 1
            if fi == 0:
                rival = 1 if own[0] == 0 else 0
                independent = lp_margin(B[0],own[0],rival,W,b,model['edges'],V)
                np.testing.assert_allclose(independent,pair[0,rival],atol=1e-7)
                maxima['LP_error'] = max(maxima['LP_error'],float(abs(independent-pair[0,rival]))); counts['validation_LP_probes'] += 1
            if m == 'robust' and fit['usable']:
                freshscores = forest_scores(B,fresh['coef'],fresh['intercept'],fresh['edges'],np.array(fresh['pair_coef']).reshape(q,len(fresh['edges']),3,3))
                maxima['fresh_score_error'] = max(maxima['fresh_score_error'],float(np.max(abs(freshscores-scores))))
                np.testing.assert_allclose(freshscores,scores,atol=1e-6,rtol=1e-6)
        evaluations.append(values); print(name,fi,'fresh robust train/validation audit passed',flush=True)
    metrics = {m:{k:float(np.mean([r[m][k] for r in evaluations])) for k in ('clean_BA','certified_BA','nominal_logloss','envelope_logloss')} for m in ('ordinary','robust')}
    counts['datasets'] += 1; counts['positive_clean_datasets'] += metrics['robust']['clean_BA'] > metrics['ordinary']['clean_BA']+1e-12
    result = {'dataset':name,'metrics':metrics,'counts':dict(counts),'precision_maxima':maxima}
    checkpoint.write_text(json.dumps({'fingerprints':fp,'result':result},indent=2),encoding='utf-8'); return result


if __name__ == '__main__':
    saved = json.loads((FOLDER/'complete.json').read_text()); names = [r['dataset'] for r in saved['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = [f.result() for f in as_completed([pool.submit(audit,n) for n in names])]
    lookup = {r['dataset']:r for r in rows}; counts = sum((Counter(r['counts']) for r in rows),Counter())
    for k,v in saved['counts'].items(): assert counts[k] == v
    for row in saved['rows']: equivalent(row['metrics'],lookup[row['dataset']]['metrics'])
    means = {m:{k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in ('clean_BA','certified_BA','nominal_logloss','envelope_logloss')} for m in ('ordinary','robust')}; equivalent(saved['means'],means)
    gain = 100*(means['robust']['clean_BA']-means['ordinary']['clean_BA']); loss_gain = means['ordinary']['envelope_logloss']-means['robust']['envelope_logloss']
    equivalent(saved['clean_BA_gain_pp'],gain); equivalent(saved['validation_envelope_loss_gain'],loss_gain)
    gate = gain >= .5 and loss_gain >= 0 and counts['positive_clean_datasets'] >= 8 and counts['new_failures'] == counts['ordinary_caps'] == 0
    assert saved['numeric_resource_gate'] == gate and not saved['goal_achieved']
    result = {'passed':True,'counts':dict(counts),'resource_gate':bool(gate),'clean_BA_gain_pp':gain,'validation_envelope_loss_gain':loss_gain,
        'precision_maxima':{k:max(r['precision_maxima'][k] for r in rows) for k in ('fresh_score_error','LP_error')},
        'auditor_sha256':digest(Path(__file__)),'scope':'allfresh_training_solves_validation_scores_LP_provenance_clean_primary_full_classifier_only'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)
