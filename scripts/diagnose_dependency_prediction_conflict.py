"""Constructive incompatibility: duplicate attribute deletion preserves votes.

Exact redundant coordinates give a strict loss violation under frozen weights
despite identical nearest-neighbor ordering. This does not model independent
box perturbations as perfectly correlated or establish test superiority.
"""
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from run_decision_iteration import ROOT
from certified_frfs.core import gamma,certificate
from certified_frfs.shared_loss import shared_loss_bound
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.adaptive_witness import adaptive_bounds


def main():
    t=np.r_[np.linspace(.05,.15,8),np.linspace(.85,.95,8)]
    X=np.column_stack([t,t]);y=np.repeat([0,1],8);w=np.ones(2)/np.sqrt(2)
    reduced=[0];nominal=gamma(X,y,weights=w)-gamma(X,y,reduced,weights=w)
    fixed=MajorityTerms(X,y,weights=w);p=fixed.bounds([0,1],allowed_enemies=2)>1e-10
    b=np.maximum(fixed.bounds(reduced,allowed_enemies=2),adaptive_bounds(X,y,reduced,w))
    query=np.column_stack([np.linspace(0,1,201)]*2);predictions=[]
    for fs in ([0,1],reduced):
        model=KNeighborsClassifier(n_neighbors=5,metric='manhattan',algorithm='brute')
        model.fit(X[:,fs]*w[fs],y);predictions.append(model.predict(query[:,fs]*w[fs]))
    # Entire reduced distance matrix is exactly half of the full matrix.
    full=abs(X[:,None,:]-X[None,:,:])@w
    reduced_dist=abs(X[:,None,0]-X[None,:,0])*w[0]
    np.testing.assert_allclose(full,2*reduced_dist,rtol=0,atol=1e-15)
    assert np.array_equal(*predictions) and nominal>.1 and np.all(b[p]>1e-10)
    result={'scope':'constructive_frozen_weight_fuzzy_dependency_vs_prediction_conflict_not_generalization',
        'training_shape':list(X.shape),'removed_fraction':.5,'nominal_dependency_loss':nominal,
        'loss_tolerance':.1,'nominal_realization_proves_dependency_infeasible':True,
        'full_initial_majority_certified':int(p.sum()),'reduced_majority_retained':int(((b>1e-10)&p).sum()),
        'identical_predictions_on_queries':len(query),'distance_scale_factor':.5,
        'old_bound':certificate(X,y,reduced,weights=w,eps=.01,method='hybrid')['upper_bound'],
        'shared_bound':shared_loss_bound(X,y,reduced,w)['upper_bound']}
    directory=ROOT/'results/dependency-prediction-conflict';directory.mkdir(exist_ok=True)
    (directory/'duplicate-counterexample.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
