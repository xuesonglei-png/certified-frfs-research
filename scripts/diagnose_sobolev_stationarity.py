"""Correct flagged training fits on the same objective; do not overwrite v1."""
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,digest
from run_sobolev_classifier_pilot import FOLDER
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.spline_forest import spline_memberships,spline_scores
from certified_frfs.sobolev_spline_forest import sobolev_basis,hat_stiffness
from certified_frfs.functional_spline_forest import hat_integral_gram
from certified_frfs.logistic_stationarity_refinement import refine_balanced_logistic

def corrected_fit(A,y,fit):
    q,d,k = np.array(fit['coef']).shape; width = k-1; fs,edges = fit['features'],fit['edges']
    B = np.stack([sobolev_basis(t,fit['alpha']) for t in fit['knots']]); R = np.stack([hat_integral_gram(t)+fit['alpha']*hat_stiffness(t) for t in fit['knots']])
    Z = np.einsum('ndk,dki->ndi',spline_memberships(A,fit['knots']),B)
    design = np.column_stack([Z[:,fs].reshape(len(A),-1)]+[(Z[:,u,:,None]*Z[:,v,None,:]).reshape(len(A),width**2) for u,v in edges])
    dual = np.einsum('dkl,dli->dki',R,B); W,V = np.array(fit['coef']),np.array(fit['pair_coef']).reshape(q,len(edges),k,k)
    decoded = np.column_stack([np.einsum('qdk,dki->qdi',W[:,fs],dual[fs]).reshape(q,-1)]+
        [np.einsum('ki,qkl,lj->qij',dual[u],V[:,e],dual[v]).reshape(q,width**2) for e,(u,v) in enumerate(edges)])
    result = refine_balanced_logistic(design,y,decoded,fit['intercept'],fit['C'])
    theta = result['coef']; W = np.zeros((q,d,k)); V = np.zeros((q,len(edges),k,k))
    W[:,fs] = np.einsum('qdi,dki->qdk',theta[:,:width*len(fs)].reshape(q,len(fs),width),B[fs])
    for e,(u,v) in enumerate(edges):
        start = width*len(fs)+e*width**2; block = theta[:,start:start+width**2].reshape(q,width,width)
        V[:,e] = np.einsum('ki,qij,lj->qkl',B[u],block,B[v])
    new = {**fit,'coef':W.tolist(),'pair_coef':V.tolist(),'intercept':result['intercept'].tolist(),
        **{key:value for key,value in result.items() if key not in ('coef','intercept','classes','scope')}}
    return new,result

def diagnose(name):
    threadpool_limits(1); saved = json.loads((FOLDER/(name+'.json')).read_text()); X,y = load_dataset(name); rows = []
    for r in saved['records']:
        ri,qi = np.array(r['reference_indices']),np.array(r['validation_indices']); raw,labels = X[ri],y[ri]
        for fi,fold in enumerate(r['folds']):
            a,b = np.array(fold['reference_indices']),np.array(fold['validation_indices']); A,B = preprocess_fit(raw[a],raw[b])
            for mid,model in enumerate(fold['models']):
                fit = model['fit']
                if fit['optimizer_converged']: continue
                new,diagnostic = corrected_fit(A,labels[a],fit)
                oldscores,newscores = spline_scores(B,fit),spline_scores(B,new)
                selected = [m for m,ci in r['choices'].items() if r['trace']['refined' if m == 'refined_alpha0' else 'old_selector' if m == 'old_alpha0' else m][ci]['model_ids'][fi] == mid]
                rows.append({'outer':r['outer_index'],'child':fi,'model':mid,'scope':'child','selected_by':selected,
                    'old_gradient':fit['gradient_inf'],'new_gradient':diagnostic['gradient_inf'],'corrected_converged':new['optimizer_converged'],
                    'objective_change':diagnostic['objective']-fit['objective'],'initial_gradient':diagnostic['initial_gradient_inf'],
                    'validation_prediction_changes':int(np.sum(oldscores.argmax(axis=1) != newscores.argmax(axis=1))),
                    'score_max_change':float(abs(oldscores-newscores).max()),'refinement_iterations':diagnostic['refinement_iterations']})
        # Parent labels are not needed to assess prediction/score perturbations.
        A,B = preprocess_fit(raw,X[qi])
        for method,fit in r['fits'].items():
            if fit['optimizer_converged']: continue
            new,diagnostic = corrected_fit(A,labels,fit); oldscores,newscores = spline_scores(B,fit),spline_scores(B,new)
            rows.append({'outer':r['outer_index'],'scope':'parent','method':method,'old_gradient':fit['gradient_inf'],
                'new_gradient':diagnostic['gradient_inf'],'corrected_converged':new['optimizer_converged'],
                'objective_change':diagnostic['objective']-fit['objective'],'validation_prediction_changes':int(np.sum(oldscores.argmax(axis=1) != newscores.argmax(axis=1))),
                'score_max_change':float(abs(oldscores-newscores).max()),'refinement_iterations':diagnostic['refinement_iterations']})
    result = {'dataset':name,'artifact_sha256':digest(FOLDER/(name+'.json')),'rows':rows}
    print(name,'flagged',len(rows),'remaining',sum(not row['corrected_converged'] for row in rows),flush=True); return result

if __name__ == '__main__':
    names = [r['dataset'] for r in json.loads((FOLDER/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = [f.result() for f in as_completed([pool.submit(diagnose,n) for n in names])]
    flagged = [r for d in rows for r in d['rows']]
    result = {'frozen_outputs_unchanged':True,'rows':rows,'flagged_models':len(flagged),
        'remaining_flags':sum(not r['corrected_converged'] for r in flagged),
        'prediction_changes':sum(r['validation_prediction_changes'] for r in flagged),
        'maximum_score_change':max(r['score_max_change'] for r in flagged),
        'scope':'same_objective_stationarity_diagnosis_not_rerun_CV_or_claim_of_prediction_advantage',
        'code_sha256':digest(ROOT/'src/certified_frfs/logistic_stationarity_refinement.py')}
    (ROOT/'results/sobolev-stationarity-diagnosis.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print({k:v for k,v in result.items() if k != 'rows'},flush=True)
