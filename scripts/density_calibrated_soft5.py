"""Anchor-gap calibrated crossgroup soft5; no query-label temperature tuning.

Gap theorem calibrates an anchor, not all subsequent learned weights. Clamps,
ties and drift are reported; original admission still governs deployment.
"""
import numpy as np
from scipy.special import expit,log_expit,logsumexp
from crossgroup_soft5_terms import crossgroup_terms
from soft_topk_metric_objective import soft_topk_membership


def balanced_membership(D,mask,temperature,*,return_slope=False):
    """Solve outside mass = inside deficit in log space, avoiding cancellation.

    A sum(q)-5 residual can be zero across a numerical plateau while tiny
    own-class probabilities still differ. This equation retains both tails.
    """
    D,mask=np.asarray(D,float),np.asarray(mask,bool)
    if D.shape!=mask.shape or not np.isfinite(D).all() or np.any(mask.sum(axis=1)<5):
        raise ValueError('finite distances with at least five allowed references required')
    Q=mask.astype(float);slope_out=np.zeros_like(D);active=mask.sum(axis=1)>5
    if not np.any(active):return (Q,slope_out) if return_slope else Q
    dist=D[active];allowed=mask[active]
    order=np.argsort(np.where(allowed,dist,np.inf),axis=1,kind='stable')
    inside=np.zeros_like(allowed);np.put_along_axis(inside,order[:,:5],True,axis=1)
    outside=allowed & ~inside
    center=np.take_along_axis(dist,order[:,4:5],axis=1)[:,0]
    Z=-(dist-center[:,None])/temperature
    low=-np.max(np.where(allowed,Z,-np.inf),axis=1)-40
    high=-np.min(np.where(allowed,Z,np.inf),axis=1)+40
    b=(np.take_along_axis(dist,order[:,5:6],axis=1)[:,0]-center)/(2*temperature)
    done=np.zeros(len(dist),bool)
    for _ in range(200):
        logits=Z+b[:,None]
        lout=np.where(outside,log_expit(logits),-np.inf)
        lin=np.where(inside,log_expit(-logits),-np.inf)
        norm_out=logsumexp(lout,axis=1);norm_in=logsumexp(lin,axis=1)
        residual=norm_out-norm_in
        done=done | (abs(residual)<1e-11)
        if done.all():
            Q[active]=np.where(allowed,expit(logits),0.)
            slope_out[active]=np.where(allowed,expit(logits)*expit(-logits),0.)
            assert np.max(abs(Q.sum(axis=1)-5))<1e-10
            return (Q,slope_out) if return_slope else Q
        low=np.where((residual<0)&~done,b,low);high=np.where((residual>0)&~done,b,high)
        slope=(np.exp(lout-norm_out[:,None])*expit(-logits)).sum(axis=1)
        slope+=(np.exp(lin-norm_in[:,None])*expit(logits)).sum(axis=1)
        newton=b-residual/np.maximum(slope,1e-14)
        inside_bracket=(newton>low)&(newton<high)&np.isfinite(newton)
        b=np.where(done,b,np.where(inside_bracket,newton,(low+high)/2))
    raise RuntimeError('log-balanced soft5 threshold did not converge')


def calibrate(terms,initial):
    C,same,mask,labels=terms
    if not len(labels):
        return .005,{'status':'no_crossgroup_queries','temperature':.005,'target_leakage_fraction':.1,
            'floor':.0005,'ceiling':.05,'eligible_queries':0,'zero_gap_queries':0,'raw_temperature':None,
            'anchor_queries_with_target_bound':0,'scope':'anchor_gap_calibration_not_learned_weight_or_risk_guarantee'}
    D=np.einsum('ijd,d->ij',C,initial);thresholds=[];gaps=[];zero=0;eligible=[]
    for row,allowed in zip(D,mask,strict=True):
        values=np.sort(row[allowed]);m=len(values)
        if m<=5:continue
        gap=values[5]-values[4]
        log_ratio=np.log(4.5*(m-5.5)/.25)
        assert log_ratio>0
        eligible.append((gap,log_ratio))
        if gap<=0:zero+=1
        else:thresholds.append(float(gap/log_ratio));gaps.append(float(gap))
    raw=float(np.median(thresholds)) if thresholds else .005
    temperature=float(np.clip(raw,.0005,.05))
    info={'status':'positive_gap_median' if thresholds else 'all_gaps_tied_fixed_fallback',
        'temperature':temperature,'target_leakage_fraction':.1,'floor':.0005,'ceiling':.05,
        'eligible_queries':len(eligible),'zero_gap_queries':zero,'raw_temperature':raw,
        'median_positive_gap':None if not gaps else float(np.median(gaps)),
        'anchor_queries_with_target_bound':int(sum(gap-temperature*log_ratio>1e-12 for gap,log_ratio in eligible)),
        'scope':'anchor_gap_calibration_not_learned_weight_or_risk_guarantee'}
    return temperature,info


def objective(w,terms,temperature,reg=.01,delta=1e-12):
    C,same,mask,labels=terms;w=np.asarray(w,float)
    if not len(labels):return 0.,np.zeros_like(w)
    classes=np.unique(labels);alpha=np.array([1/(len(classes)*sum(labels==c)) for c in labels])
    D=np.einsum('ijd,d->ij',C,w)
    Q,S=balanced_membership(D,mask,temperature,return_slope=True)
    own=(Q*same).sum(axis=1)/5
    mean_C=np.einsum('ij,ijd->id',S,C)/np.maximum(S.sum(axis=1)[:,None],np.finfo(float).tiny)
    same_S=S*same
    derivative=(same_S.sum(axis=1)[:,None]*mean_C-np.einsum('ij,ijd->id',same_S,C))/(5*temperature)
    log=np.log(len(w)*(w+delta))
    value=-alpha@np.log(own+delta)+reg*np.sum(w*log)
    gradient=-np.einsum('i,id->d',alpha/(own+delta),derivative)+reg*(log+w/(w+delta))
    return float(value),gradient
