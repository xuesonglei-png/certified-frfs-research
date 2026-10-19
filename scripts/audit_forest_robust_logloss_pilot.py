"""Legacy full-class DP, independent LP, cut geometry and objective audit."""
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from pilot_forest_robust_logloss import FOLDER,NAMES
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.fuzzy_interaction_tree import interaction_tree
from certified_frfs.forest_fuzzy_classifier import forest_box_margins,forest_scores
from certified_frfs.orthogonal_forest_logistic import CONTRAST
from audit_fuzzy_forest_capacity import lp_margin
from audit_forest_compact_transfer import equivalent


if __name__ == '__main__':
    threadpool_limits(1); summary = json.loads((FOLDER/'complete.json').read_text()); rows = []; probes = 0; allpairs = 0; cuts = 0; maxlp = 0.
    for name in NAMES:
        source = source_verified(name); artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text())
        for rel,sha in saved['fingerprints'].items(): assert digest(ROOT/rel) == sha
        assert not saved['archived_query_labels_used']; s = source['splits'][0]; inner = s['inner'][0]
        assert saved['outer_reference_indices'] == s['reference_indices'] and saved['reference_indices'] == inner['reference_indices']
        X,y = load_dataset(name); ri,a = np.array(saved['outer_reference_indices']),np.array(saved['reference_indices']); raw,labels = X[ri[a]],y[ri[a]]; A,_ = preprocess_fit(raw,raw)
        equivalent(saved['tree'],interaction_tree(A,labels)); fit = saved['fit']; assert fit['usable'] and fit['status'] == 'separated'
        classes = np.array(fit['classes']); own = np.searchsorted(classes,labels); q,n = len(classes),len(A)
        assert fit['features'] == list(range(A.shape[1])) and fit['edges'] == saved['tree']['edges']
        assert fit['C'] == fit['pair_penalty'] == 1. and fit['eps'] == .01 and fit['separation_tol'] == 2e-7 and not fit['teacher_guards_enforced']
        W,V,b = np.array(fit['coef']),np.array(fit['pair_coef']).reshape(q,len(fit['edges']),3,3),np.array(fit['intercept'])
        oracle = forest_box_margins(A,labels,W,b,classes,fit['edges'],V,eps=.01,witnesses=True); margins = oracle['pair_margins'].copy(); margins[np.arange(n),own] = 0.
        np.testing.assert_allclose(fit['envelope_pair_margins'],margins,atol=1e-10)
        losses = logsumexp(-margins,axis=1); np.testing.assert_allclose(fit['envelope_losses'],losses,atol=1e-10)
        upper = np.array(fit['upper']); assert np.max(-margins-upper) <= fit['separation_tol']+1e-10
        np.testing.assert_allclose(upper[np.arange(n),own],0.,atol=1e-7)
        alpha = np.array([1/(q*sum(labels == c)) for c in labels])
        beta = np.array(fit['beta'])
        main = (W[:,fit['features']]@CONTRAST).reshape(q,-1)
        pair = np.einsum('ki,qekl,lj->qeij',CONTRAST,V,CONTRAST).reshape(q,-1)
        np.testing.assert_allclose(beta,np.column_stack([main,pair]),atol=1e-10)
        np.testing.assert_allclose(beta[0],0.,atol=1e-7); np.testing.assert_allclose(b[0],0.,atol=1e-7)
        penalized = beta[1] if q == 2 else beta-beta.mean(axis=0,keepdims=True)
        penalty = np.sum(penalized**2)/(2*n); objective = float(alpha@losses+penalty)
        equivalent(fit['exact_envelope_objective'],objective); equivalent(saved['new_envelope_objective'],objective)
        relaxation = float(alpha@logsumexp(upper,axis=1)+penalty); equivalent(fit['relaxation_objective'],relaxation)
        assert objective-relaxation <= fit['separation_tol']+1e-8 and fit['trace'][-1]['max_separation_violation'] <= fit['separation_tol']
        for cut in fit['cuts']:
            i,r = cut['object'],cut['rival']; point = np.array(cut['raw']); assert r != own[i]
            lo,hi = np.maximum(0,A[i]-.01),np.minimum(1,A[i]+.01); assert np.all(point >= lo-1e-12) and np.all(point <= hi+1e-12)
            vertices = np.stack([lo,hi,np.clip(.5,lo,hi)],axis=1)
            assert np.all(np.min(abs(vertices-point[:,None]),axis=1) < 1e-12)
            scores = forest_scores(point[None],W,b,fit['edges'],V)[0]
            assert scores[r]-scores[own[i]] <= upper[i,r]+fit['separation_tol']+1e-9; cuts += 1
        for i in sorted(set([0,n//2,n-1])):
            for rival in range(q):
                if rival == own[i]: continue
                independent = lp_margin(A[i],own[i],rival,W,b,fit['edges'],V)
                np.testing.assert_allclose(independent,margins[i,rival],atol=1e-7); maxlp = max(maxlp,float(abs(independent-margins[i,rival]))); probes += 1
        old = saved['ordinary']; ow,ov,ob = np.array(old['coef']),np.array(old['pair_coef']).reshape(q,len(fit['edges']),3,3),np.array(old['intercept'])
        oldoracle = forest_box_margins(A,labels,ow,ob,classes,old['edges'],ov,eps=.01,witnesses=True); oldm = oldoracle['pair_margins'].copy(); oldm[np.arange(n),own] = 0.
        if q == 2: ow,ov = ow[1],ov[1]
        else: ow,ov = ow-ow.mean(axis=0,keepdims=True),ov-ov.mean(axis=0,keepdims=True)
        oldvalue = float(alpha@logsumexp(-oldm,axis=1)+(np.sum(ow*ow)+np.sum(ov*ov))/(2*n)); equivalent(saved['old_envelope_objective'],oldvalue)
        assert objective <= oldvalue+1e-6; allpairs += n*(q-1)
        row = {'dataset':name,'old_objective':oldvalue,'new_objective':objective,'relative_improvement':(oldvalue-objective)/oldvalue}
        expected = next(r for r in summary['rows'] if r['dataset'] == name)
        for key,value in row.items(): equivalent(expected[key],value)
        rows.append(row); print(name,'legacyDP/LP/cut/objective audit passed',flush=True)
    gate = all(r['new_objective'] <= r['old_objective']+1e-6 for r in rows) and float(np.mean([r['relative_improvement'] for r in rows])) >= .001
    assert summary['numeric_resource_gate'] == gate and not summary['goal_achieved']
    result = {'passed':True,'resource_gate':bool(gate),'rows':rows,'legacy_all_pair_margins':allpairs,'cut_replays':cuts,'independent_LP_probes':probes,'max_LP_error':maxlp,
        'artifact_hashes':{name:digest(FOLDER/(name+'.json')) for name in NAMES},'auditor_sha256':digest(Path(__file__)),
        'scope':'numerical_epigraph_feasibility_and_exact_forest_losses_not_global_optimum_or_test_accuracy'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)
