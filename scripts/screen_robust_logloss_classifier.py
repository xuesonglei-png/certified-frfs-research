"""All15 first references, fold-isolated robust-risk classifier prediction."""
import json,time
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.forest_fuzzy_classifier import forest_scores,forest_box_margins
from certified_frfs.forest_robust_logloss import fit_robust_forest_logloss

INPUT = ROOT/'results/bagged-forest-mechanism-screen'; FOLDER = ROOT/'results/robust-logloss-classifier-screen'


def evaluate(Q,truth,fit):
    classes = np.array(fit['classes']); own = np.searchsorted(classes,truth); q = len(classes)
    W,V,b = np.array(fit['coef']),np.array(fit['pair_coef']).reshape(q,len(fit['edges']),3,3),np.array(fit['intercept'])
    scores = forest_scores(Q,W,b,fit['edges'],V); pred = classes[scores.argmax(axis=1)]
    oracle = forest_box_margins(Q,truth,W,b,classes,fit['edges'],V,eps=.01,witnesses=True)
    pair = oracle['pair_margins'].copy(); pair[np.arange(len(Q)),own] = 0.
    nominal = logsumexp(scores,axis=1)-scores[np.arange(len(Q)),own]; envelope = logsumexp(-pair,axis=1)
    def balanced(values): return float(np.mean([np.mean(values[truth == c]) for c in np.unique(truth)]))
    return {'scores':scores.tolist(),'predictions':pred.tolist(),'pair_margins':pair.tolist(),
        'metrics':{'clean_BA':balanced(pred == truth),'certified_BA':balanced(oracle['margins'] > 1e-10),
            'nominal_logloss':balanced(nominal),'envelope_logloss':balanced(envelope)}}


def run(name):
    threadpool_limits(1); source = source_verified(name); X,y = load_dataset(name)
    upstream = json.loads((INPUT/(name+'.json')).read_text()); audit = json.loads((INPUT/(name+'-audit.json')).read_text())
    assert audit['fingerprints']['artifact_sha256'] == digest(INPUT/(name+'.json'))
    assert json.loads((INPUT/'complete-audit.json').read_text())['passed'] and json.loads((ROOT/'results/forest-robust-logloss-pilot/complete-audit.json').read_text())['resource_gate']
    hashes = dict(upstream['fingerprints'])
    for p in (Path(__file__),ROOT/'src/certified_frfs/forest_robust_logloss.py',ROOT/'research/robust-logloss-classifier-screen-protocol.md',
        INPUT/(name+'.json'),INPUT/(name+'-audit.json')): hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    for rel,sha in hashes.items(): assert digest(ROOT/rel) == sha
    artifact = FOLDER/(name+'.json'); folds = []
    if artifact.exists():
        prior = json.loads(artifact.read_text()); assert prior['fingerprints'] == hashes
        if prior['complete']: return
        folds = prior['folds']
    r = upstream['rows'][0]; assert r['outer'] == r['inner'] == 0
    ri,a = np.array(r['outer_reference_indices']),np.array(r['reference_indices']); raw,labels = X[ri[a]],y[ri[a]]
    for index,fold in enumerate(r['folds']):
        if index < len(folds): continue
        ca,cb = np.array(fold['reference_indices']),np.array(fold['validation_indices']); A,B = preprocess_fit(raw[ca],raw[cb]); start = time.perf_counter()
        control = fold['single']; fit = fit_robust_forest_logloss(A,labels[ca],list(range(A.shape[1])),fold['tree']['edges'],C=1.,eps=.01)
        failed = not fit['usable']; deployed = control if failed else fit
        evaluations = {'ordinary':evaluate(B,labels[cb],control),'robust':evaluate(B,labels[cb],deployed)}
        record = {'reference_indices':ca.tolist(),'validation_indices':cb.tolist(),'tree':fold['tree'],'ordinary':control,'fit':fit,
            'ordinary_fallback':failed,'evaluations':evaluations,'seconds':time.perf_counter()-start}
        folds.append(record)
        artifact.write_text(json.dumps({'dataset':name,'fingerprints':hashes,'outer_reference_indices':ri.tolist(),'reference_indices':a.tolist(),
            'folds':folds,'complete':len(folds) == len(r['folds']),'archived_query_labels_used':False,
            'scope':'firstref_internal_validation_full_classifier_only_no_reduct_SOTA'},separators=(',',':')),encoding='utf-8')
        print(name,index,'usable',not failed,'BAgainpp',round(100*(evaluations['robust']['metrics']['clean_BA']-evaluations['ordinary']['metrics']['clean_BA']),6),
            'seconds',round(record['seconds'],2),flush=True)


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True); names = [r['dataset'] for r in json.loads((INPUT/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]): f.result()
    rows = []; counts = Counter()
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete']
        metrics = {m:{k:float(np.mean([f['evaluations'][m]['metrics'][k] for f in saved['folds']])) for k in ('clean_BA','certified_BA','nominal_logloss','envelope_logloss')} for m in ('ordinary','robust')}
        rows.append({'dataset':name,'metrics':metrics}); counts['datasets'] += 1; counts['positive_clean_datasets'] += metrics['robust']['clean_BA'] > metrics['ordinary']['clean_BA']+1e-12
        for f in saved['folds']:
            counts['fits'] += 1; counts['new_failures'] += f['ordinary_fallback']; counts['ordinary_caps'] += not f['ordinary']['optimizer_converged']
    means = {m:{k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in ('clean_BA','certified_BA','nominal_logloss','envelope_logloss')} for m in ('ordinary','robust')}
    gain = 100*(means['robust']['clean_BA']-means['ordinary']['clean_BA']); envelope_gain = means['ordinary']['envelope_logloss']-means['robust']['envelope_logloss']
    gate = gain >= .5 and envelope_gain >= 0 and counts['positive_clean_datasets'] >= 8 and counts['new_failures'] == counts['ordinary_caps'] == 0
    result = {'rows':rows,'means':means,'counts':dict(counts),'clean_BA_gain_pp':gain,'validation_envelope_loss_gain':envelope_gain,
        'numeric_resource_gate':bool(gate),'independent_audit_required':True,'goal_achieved':False,'archived_query_labels_used':False,
        'scope':'all15_firstref_clean_primary_isolated_internal_full_classifier_mechanism_no_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)
