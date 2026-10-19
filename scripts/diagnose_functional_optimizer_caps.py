"""Record stationarity and selection use of every archived capped candidate."""
import json
import numpy as np
from scipy.special import logsumexp
from run_nested_joint_path import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from functional_spline_audit_oracle import bases
from spline_forest_audit_oracle import membership


def main():
    rows = []
    summary = json.loads((ROOT/'results/functional-normalization-pilot/complete.json').read_text())
    for dataset in summary['rows']:
        name = dataset['dataset']; saved = json.loads((ROOT/'results/functional-normalization-pilot'/(name+'.json')).read_text())
        X,y = load_dataset(name)
        for r in saved['records']:
            ri,a = np.array(r['outer_reference_indices']),np.array(r['reference_indices'])
            raw,labels = X[ri[a]],y[ri[a]]
            for fi,fold in enumerate(r['tuning']['folds']):
                capped = [(mi,m['fit']) for mi,m in enumerate(fold['models']) if not m['fit']['optimizer_converged']]
                if not capped: continue
                ca,cb = np.array(fold['reference_indices']),np.array(fold['validation_indices'])
                A,_ = preprocess_fit(raw[ca],raw[cb]); train = labels[ca]
                for mi,fit in capped:
                    q,k = len(fit['classes']),len(fit['knots'][0]); fs = fit['features']
                    basis,H,mu = bases(fit['knots']); Z = np.einsum('ndk,dki->ndi',membership(A,fit['knots']),basis)
                    blocks = [Z[:,fs].reshape(len(A),-1)]
                    blocks += [(Z[:,u,:,None]*Z[:,v,None,:]).reshape(len(A),(k-1)**2)/np.sqrt(fit['pair_penalty']) for u,v in fit['edges']]
                    design = np.column_stack(blocks); W,V = np.array(fit['coef']),np.array(fit['pair_coef'])
                    dual = np.einsum('dkl,dli->dki',H,basis)
                    decoded = [np.einsum('qdk,dki->qdi',W[:,fs],dual[fs]).reshape(q,-1)]
                    decoded += [(np.einsum('ki,qkl,lj->qij',dual[u],V[:,e],dual[v])*np.sqrt(fit['pair_penalty'])).reshape(q,-1)
                        for e,(u,v) in enumerate(fit['edges'])]
                    theta = np.column_stack(decoded); theta -= theta.mean(axis=0,keepdims=True)
                    bias = np.array(fit['intercept']); bias -= bias.mean(); logits = design@theta.T+bias
                    own = np.argmax(train[:,None] == np.array(fit['classes'])[None],axis=1)
                    balance = np.array([1/(q*sum(train == c)) for c in train])
                    residual = balance[:,None]*(np.exp(logits-logsumexp(logits,axis=1,keepdims=True))-np.eye(q)[own])
                    gradient = residual.T@design+theta/(fit['C']*len(A))
                    error = max(float(abs(gradient).max()),float(abs(residual.sum(axis=0)).max()))
                    selected = [method for method,choice in r['tuning']['choices'].items()
                        if r['tuning']['trace'][method][choice['config_index']]['fold_model_ids'][fi] == mi]
                    rows.append({'dataset':name,'outer_index':r['outer_index'],'inner_index':r['inner_index'],
                        'child_fold':fi,'model_id':mi,'C':fit['C'],'k':k,'q':q,'iterations':fit['optimizer_iterations'],
                        'gradient_inf':error,'selected_by':selected})
    assert len(rows) == summary['counts']['child_optimizer_caps']
    result = {'caps':rows,'scope':'independent_stationarity_and_selection_use_diagnostic_no_artifacts_or_optimizer_flags_changed'}
    (ROOT/'results/functional-normalization-optimizer-caps.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result,flush=True)


if __name__ == '__main__': main()
