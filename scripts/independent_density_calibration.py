"""Independent scalar anchor gaps and Brent loss for density calibration."""
import numpy as np
from independent_crossgroup_soft5 import independent_groups,independent_group_info


def calibration(A,y,w):
    A,y,w=np.asarray(A,float),np.asarray(y),np.asarray(w,float)
    info=independent_group_info(y);g=independent_groups(y)
    if not info['query_indices']:
        return {'status':'no_crossgroup_queries','temperature':.005,'target_leakage_fraction':.1,
            'floor':.0005,'ceiling':.05,'eligible_queries':0,'zero_gap_queries':0,'raw_temperature':None,
            'anchor_queries_with_target_bound':0,'scope':'anchor_gap_calibration_not_learned_weight_or_risk_guarantee'}
    lo,hi=np.maximum(0,A-.01),np.minimum(1,A+.01);thresholds=[];gaps=[];pairs=[];zero=0
    for i in info['query_indices']:
        ids=np.flatnonzero(g!=g[i]);same=y[ids]==y[i];distance=np.zeros(len(ids))
        if len(ids)==5:continue
        for a,weight in enumerate(w):
            lower=np.maximum(0,np.maximum(lo[ids,a]-hi[i,a],lo[i,a]-hi[ids,a]))
            upper=np.maximum(abs(lo[i,a]-hi[ids,a]),abs(hi[i,a]-lo[ids,a]))
            distance+=weight*np.where(same,upper,lower)
        ordered=sorted(distance.tolist());gap=ordered[5]-ordered[4]
        log_ratio=np.log((5-.5)*(len(ids)-5-.5)/(.5**2));pairs.append((gap,log_ratio))
        if gap==0:zero+=1
        else:thresholds.append(gap/log_ratio);gaps.append(gap)
    raw=float(np.median(thresholds)) if thresholds else .005
    temperature=min(.05,max(.0005,raw))
    return {'status':'positive_gap_median' if thresholds else 'all_gaps_tied_fixed_fallback',
        'temperature':temperature,'target_leakage_fraction':.1,'floor':.0005,'ceiling':.05,
        'eligible_queries':len(pairs),'zero_gap_queries':zero,'raw_temperature':raw,
        'median_positive_gap':None if not gaps else float(np.median(gaps)),
        'anchor_queries_with_target_bound':int(sum(gap-temperature*log_ratio>1e-12 for gap,log_ratio in pairs)),
        'scope':'anchor_gap_calibration_not_learned_weight_or_risk_guarantee'}
