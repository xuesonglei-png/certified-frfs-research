"""All15/full native controls: structural certificate obstacles, no training.

Native metrics were selected on already-exposed inner folds. This diagnostic
does NOT turn them into train-only initializations or select any new method.
Positive nominal class dependency loss > eta is a valid violation witness
because the nominal table belongs to the uncertainty box. A sufficient upper
bound above eta is merely unavailable evidence, never proof of infeasibility.
"""
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT,digest
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from dynamic_certificate_metric_v1 import original_protection,ranked_witnesses
from signed_classwise_dependency import signed_object_witness_bounds
from independent_canonical_majority import friend_enemy_bounds

BASE=ROOT/'results/frs-joint-zero-face-prediction'
FOLDER=ROOT/'results/native-certificate-obstacles-v1'
FREEZE=ROOT/'research/native-certificate-obstacles-v1-source-freeze.json'
STATUS=ROOT/'results/native-certificate-obstacles-v1-chain-status.json'
CONTROLS=('fgs_native120','fgs_weighted_native120','sfrip_native6','sfrip_weighted_native6')


def nominal_membership(A,y,fs,w):
    D=cdist(A[:,fs]*w,A[:,fs]*w,'cityblock')
    D[y[:,None]==y[None,:]]=np.inf
    return np.minimum(1.,D.min(axis=1))


def independent_nominal_membership(A,y,fs,w):
    out=[]
    for i in range(len(A)):
        enemies=np.flatnonzero(y!=y[i]);distance=np.zeros(len(enemies))
        for a,weight in zip(fs,w):distance+=abs(A[i,a]*weight-A[enemies,a]*weight)
        out.append(min(1.,float(distance.min())))
    return np.array(out)


def status(phase,**extra):
    STATUS.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,
        'classification_superiority_established':False,**extra},indent=2),encoding='utf-8')


def inspect_metric(A,y,metric,protected):
    d=A.shape[1];mass=np.sqrt(d);full=np.ones(d)/mass
    fs=metric['features'];raw=np.array(metric['weights']);w=raw*(mass/raw.sum())
    before=nominal_membership(A,y,list(range(d)),full)
    after=nominal_membership(A,y,fs,w)
    np.testing.assert_allclose(before,independent_nominal_membership(A,y,list(range(d)),full),atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(after,independent_nominal_membership(A,y,fs,w),atol=1e-12,rtol=1e-12)
    rows=ranked_witnesses(A[:,fs],y,w);mask=np.array([r is not None and r['bound']>1e-10 for r in rows])
    # Full independent majority mask, not just a few selected witnesses.
    independent=[]
    for i in range(len(A)):
        bounds=friend_enemy_bounds(A[:,fs],y,w,i)
        value=-np.inf if bounds is None else np.inf if len(bounds)<3 else sorted(bounds)[2]
        bound=value-1e-10*(1+abs(value)) if np.isfinite(value) else value
        independent.append(bound>1e-10)
    assert mask.tolist()==independent
    classes=[]
    for c in np.unique(y):
        own=y==c;g=own&protected;need=int(np.ceil(.9*g.sum()-1e-12));kept=int((g&mask).sum())
        nominal_loss=float(np.mean(before[own]-after[own]))
        classes.append({'class':c.item(),'count':int(own.sum()),'nominal_signed_loss':nominal_loss,
            'nominal_FRS_violation_witness':bool(nominal_loss>.1+1e-10),
            'original_protected':int(g.sum()),'required':need,'retained':kept,'majority_sufficient':kept>=need})
    # This existing numerical sufficient envelope is a diagnostic, not a new
    # theorem or independently full-reconstructed certificate implementation.
    bound=None
    if not any(r['nominal_FRS_violation_witness'] for r in classes):
        signed=signed_object_witness_bounds(A,y,fs,full,w)
        values=np.array(signed['signed_object_upper_bounds']);bound=[]
        for r in classes:
            rawbound=float(values[y==r['class']].mean())
            upper=float(np.clip(rawbound+1e-10*(1+abs(rawbound)),0,1))
            bound.append({'class':r['class'],'raw_signed_envelope':rawbound,'upper':upper,'sufficient_FRS':upper<=.1})
    return {'original_metric':metric,'normalized_metric':{'features':fs,'weights':w.tolist()},
        'normalization_multiplier':float(mass/raw.sum()),'original_mass':float(raw.sum()),'required_mass':float(mass),
        'classes':classes,'nominal_FRS_violation_witness':any(r['nominal_FRS_violation_witness'] for r in classes),
        'majority_sufficient_all_classes':all(r['majority_sufficient'] for r in classes),
        'signed_envelope':bound,'primitive_upper_bound_fully_independently_reconstructed':False,
        'prediction_evaluation_performed':False,
        'scaling_note':'Positive scaling preserves mathematical 5NN distance ranks but changes the FRS relation; no scaled predictions are assumed bitwise identical.'}


def main():
    threadpool_limits(1);assert not FREEZE.exists() and not FOLDER.exists();FOLDER.mkdir()
    parent=verify(BASE)
    paths=[p for n in ('src','scripts','tests') for p in (ROOT/n).rglob('*.py')]
    protocol=ROOT/'research/native-certificate-obstacles-v1-protocol.md';paths.append(protocol)
    FREEZE.write_text(json.dumps({'fingerprints':{p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        'source_parent_audit_sha256':digest(BASE/'complete-audit.json'),'source_parent':parent,
        'created_unix':time.time()},indent=2),encoding='utf-8')
    freeze=json.loads(FREEZE.read_text());rows=[]
    for name in sorted(parent['artifact_hashes']):
        for rel,sha in freeze['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
        p=BASE/(name+'.json');assert digest(p)==parent['artifact_hashes'][name]
        saved=json.loads(p.read_text());X,y=load_dataset(name)
        ri,ca=[np.array(saved[k]) for k in ('reference_indices','training_indices')]
        A,_=preprocess_fit(X[ri][ca],X[ri][ca]);labels=y[ri][ca]
        protected=original_protection(A,labels);arms={}
        for control in CONTROLS:
            status('diagnosing',dataset=name,control=control)
            t=time.perf_counter();arms[control]=inspect_metric(A,labels,saved['metrics'][control],protected)
            arms[control]['seconds']=time.perf_counter()-t
            print(name,control,'nominal_violation',arms[control]['nominal_FRS_violation_witness'],
                'majority_sufficient',arms[control]['majority_sufficient_all_classes'],
                'signed_envelope',arms[control]['signed_envelope'],flush=True)
        row={'dataset':name,'d':X.shape[1],'n_training':len(A),'arms':arms,'source_sha256':digest(p),
            'all_inner_validation_labels_not_used_here':True,'source_native_metrics_previously_validation_selected':True}
        (FOLDER/(name+'.json')).write_text(json.dumps(row,indent=2),encoding='utf-8');rows.append(row)
    summary={control:{'nominal_FRS_violation_datasets':sum(r['arms'][control]['nominal_FRS_violation_witness'] for r in rows),
        'majority_sufficient_datasets':sum(r['arms'][control]['majority_sufficient_all_classes'] for r in rows),
        'candidate_original_certificate_evidence_datasets':sum(
            r['arms'][control]['majority_sufficient_all_classes'] and r['arms'][control]['signed_envelope'] is not None and
            all(s['sufficient_FRS'] for s in r['arms'][control]['signed_envelope']) for r in rows)} for control in CONTROLS}
    out={'datasets':len(rows),'rows':rows,'summary':summary,'artifact_hashes':{r['dataset']:digest(FOLDER/(r['dataset']+'.json')) for r in rows},
        'freeze_sha256':digest(FREEZE),'goal_achieved':False,'classification_superiority_established':False,
        'limits':['No new learning or method selection; all15/all4 controls included.',
            'Native metrics already validation-selected; cannot become leakage-free train-only seeds.',
            'Nominal signed FRS violation is a necessary-condition rejection; sufficient majority failure or upper bound>.1 is not impossibility.',
            'No new classification scoring; normalization changes the relation while preserving mathematical ranks.',
            'Sufficient envelope primitive not independently fully reconstructed in this diagnostic; it does not certify a deployed new method.']}
    (FOLDER/'complete.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    (ROOT/'research/native-certificate-obstacles-v1-review.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    status('complete_diagnostic_not_confirmatory',summary=summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:status('failed',error=str(e));raise
