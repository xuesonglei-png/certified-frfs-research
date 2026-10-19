"""Explicit fixed-budget adaptations, separate from published native stopping."""
import numpy as np
from .graph_recent import select_fgs
from .recent import sfrip_dependency_from_distances

def fgs_budget_path(X,y,**config):
    native=select_fgs(X,y,**config);chosen=set(native['features']);cd=native['cd']
    ranking=sorted(range(X.shape[1]),key=lambda a:(a not in chosen,-cd[a],a))
    return {'native':native,'ranking':ranking,'adaptation':'native_members_first_then_descending_cd_not_original_budget_algorithm'}

def sfrip_budget_path(X,y,*,beta=3.,confidence=.8,k=10,support=3,alpha=0.):
    X=np.asarray(X,float);y=np.asarray(y);n,d=X.shape
    current=np.zeros((n,n));remaining=list(range(d));ranking=[];history=[];value=0.;native=[];stopped=False
    while remaining:
        best=None;best_value=-np.inf;best_squared=None
        for a in remaining:
            candidate=current+(X[:,a,None]-X[None,:,a])**2
            score=sfrip_dependency_from_distances(np.sqrt(candidate),y,beta=beta,k=k,support=support,confidence=confidence)
            if score>best_value:best,best_value,best_squared=a,score,candidate
        gain=best_value-value;accepted=gain>alpha+1e-12
        if not stopped and accepted:native.append(best)
        else:stopped=True
        ranking.append(best);remaining.remove(best);current,value=best_squared,best_value
        history.append({'feature':int(best),'dependency':float(value),'gain':float(gain),'native_acceptance':accepted and not stopped})
    return {'native':{'features':native},'ranking':ranking,'history':history,
            'adaptation':'continued_greedy_after_native_stopping_not_original_fixed_budget_algorithm'}
