"""Independent screening of one discovery-selected subset; refit is uncertified."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from .crossfit_search import select_crossfit_queries


def paired_balanced_loss(reference_pred, candidate_pred, truth, alpha=.1):
    truth=np.asarray(truth)
    if not 0<alpha<1:
        raise ValueError('alpha must be in (0,1)')
    delta=(np.asarray(candidate_pred)!=truth).astype(float)-(np.asarray(reference_pred)!=truth).astype(float)
    classes=np.unique(truth)
    rows=[{'label':int(c),'count':int((truth==c).sum()),'mean_loss_delta':float(delta[truth==c].mean())}
          for c in classes]
    mean=float(np.mean([r['mean_loss_delta'] for r in rows]))
    # Each paired loss lies in [-1,1]. Class-balanced observation weights are 1/(C*n_c).
    weight_sq=sum(1/r['count'] for r in rows)/len(classes)**2
    radius=float(np.sqrt(2*weight_sq*np.log(1/alpha)))
    return {'mean_loss_delta':mean,'upper_loss_delta':min(1.,mean+radius),
            'hoeffding_radius':radius,'alpha':alpha,'class_diagnostics':rows,
            'scope':'independent_stratified_screen_fixed_models_conditional_on_discovery'}


def select_independent_screen(X,y,*,rank='relief',mode='robust',tolerance=.01,
                              seed=101,acceptance='empirical'):
    X,y=np.asarray(X,float),np.asarray(y)
    if acceptance not in ['empirical','hoeffding'] or not 0<=tolerance<=1:
        raise ValueError('invalid screening rule')
    indices=np.arange(len(X))
    discovery,screen=train_test_split(indices,test_size=.3,stratify=y,random_state=seed+41)
    search=select_crossfit_queries(X[discovery],y[discovery],rank=rank,mode=mode,
        constraint_tolerance=.01,penalty=.02,budget=24,seed=seed)
    proposed=search['features']
    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    scaler=MinMaxScaler(clip=True)
    A=scaler.fit_transform(imputer.fit_transform(X[discovery]))
    B=scaler.transform(imputer.transform(X[screen]))
    def predict(fs):
        return KNeighborsClassifier(n_neighbors=1,metric='manhattan',algorithm='brute').fit(A[:,fs],y[discovery]).predict(B[:,fs])
    assessment=paired_balanced_loss(predict(list(range(X.shape[1]))),predict(proposed),y[screen],alpha=.1/6)
    score=assessment['mean_loss_delta'] if acceptance=='empirical' else assessment['upper_loss_delta']
    accepted=score<=tolerance+1e-12
    return {'features':proposed if accepted else list(range(X.shape[1])),
            'candidate_features':proposed,'accepted':bool(accepted),'assessment':assessment,
            'discovery_indices':discovery.tolist(),'screen_indices':screen.tolist(),
            'discovery_search':search,'tolerance':tolerance,'acceptance':acceptance,
            'scope':'screened_feature_choice_final_full_training_refit_has_no_risk_bound'}
