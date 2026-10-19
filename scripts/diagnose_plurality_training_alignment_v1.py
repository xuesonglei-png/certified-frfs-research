"""Train-only mechanism diagnosis; never changes or selects a fitted endpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_plurality_certificate_development_v1 import ROOT,FOLDER,FREEZE,frozen,inputs,digest,ARMS
from crossgroup_soft5_terms import crossgroup_terms
from density_calibrated_soft5 import balanced_membership,calibrate,objective as own_mass
from soft_plurality_objective_v1 import plurality_objective

OUT=ROOT/'research/plurality-training-alignment-v1'


def summarize(A,y,weights,terms,temperature):
    C,same,mask,labels=terms;w=np.asarray(weights,float)
    D=np.einsum('ijd,d->ij',C,w)
    classes=np.unique(y);own=np.searchsorted(classes,labels)
    Q=balanced_membership(D,mask,temperature)
    membership=(y[:,None]==classes[None,:]).astype(float)
    soft=Q@membership/5
    order=np.argsort(np.where(mask,D,np.inf),axis=1,kind='stable')[:,:5]
    hard=np.column_stack([(y[order]==c).sum(axis=1) for c in classes])/5
    rival_hard=hard.copy();rival_hard[np.arange(len(labels)),own]=-np.inf
    rival_soft=soft.copy();rival_soft[np.arange(len(labels)),own]=-np.inf
    hard_gap=hard[np.arange(len(labels)),own]-rival_hard.max(axis=1)
    soft_gap=soft[np.arange(len(labels)),own]-rival_soft.max(axis=1)
    def balanced(v):return float(np.mean([np.mean(np.asarray(v)[labels==c]) for c in np.unique(labels)]))
    # Ties do NOT count as errors here: independently report their frequency.
    values={'hard_strict_plurality_fraction':balanced(hard_gap>1e-12),
        'hard_plurality_tie_fraction':balanced(abs(hard_gap)<=1e-12),
        'hard_defeated_fraction':balanced(hard_gap< -1e-12),
        'hard_majority_fraction':balanced(hard[np.arange(len(labels)),own]>=.6-1e-12),
        'mean_soft_true_rival_gap':balanced(soft_gap),
        'mean_hard_true_rival_gap':balanced(hard_gap),
        'max_class_soft_hard_vote_error':float(abs(hard-soft).max()),
        'eligible_queries':len(labels),'temperature':temperature}
    # Entropy and data gradients separately reveal scale/saturation; no inference
    # from their norms to an optimizer constrained direction or test accuracy.
    for family,fn in (('own_mass',lambda reg:own_mass(w,terms,temperature,reg=reg)),
            ('plurality',lambda reg:plurality_objective(w,terms,y,temperature,reg=reg))):
        v,g=fn(0.);rv,rg=fn(.01)
        data=g-g.mean();entropy=(rg-g)-(rg-g).mean()
        values[family]={'data_objective':v,'entropy_objective':rv-v,
            'simplex_data_gradient_norm':float(np.linalg.norm(data)),
            'simplex_entropy_gradient_norm':float(np.linalg.norm(entropy))}
    return values


def diagnose(name):
    threadpool_limits(1);record=frozen();source,previous,A,y,_,_,hashes=inputs(name,record)
    assert (FOLDER/(name+'.audit.json')).exists(),'only independently audited scientific rows'
    audit=json.loads((FOLDER/(name+'.audit.json')).read_text());assert audit['passed']
    assert audit['sha256']==digest(FOLDER/(name+'.json')) and audit['freeze_sha256']==digest(FREEZE)
    reduced=A[:,source['original_support']];mass=np.sqrt(A.shape[1]);anchor=np.asarray(source['original_anchor'])/mass
    terms,groups=crossgroup_terms(reduced,y,eps=0.);tau,calibration=calibrate(terms,anchor)
    metrics={'anchor':anchor};fit_hashes={}
    for arm in ARMS:
        path=FOLDER/(name+'.'+arm+'.fit.json');fit_hashes[arm]=digest(path)
        assert fit_hashes[arm]==audit['fit_hashes'][arm]
        fit=json.loads(path.read_text())['fit'];metrics[arm]=np.asarray(fit['weights'])/mass
    rows={m:summarize(reduced,y,w,terms,tau) for m,w in metrics.items()}
    out={'dataset':name,'rows':rows,'calibration':calibration,'training_query_indices':groups['query_indices'],
        'source_hashes':hashes,'fit_hashes':fit_hashes,'freeze_sha256':digest(FREEZE),
        'audit_sha256':digest(FOLDER/(name+'.audit.json')),'diagnostic_script_sha256':digest(Path(__file__)),
        'heldout_labels_used_in_diagnostic':False,'endpoint_selected_or_modified':False,
        'classification_superiority_established':False,'goal_achieved':False,
        'scope':'postfit_training_only_diagnostic_not_new_fit_or_independent_predictive_evidence'}
    OUT.mkdir(exist_ok=True);path=OUT/(name+'.json');assert not path.exists()
    path.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'dataset':name,'rows':rows},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True)
    diagnose(parser.parse_args().dataset)
