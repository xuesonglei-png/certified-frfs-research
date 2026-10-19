"""Independent scalar signed witnesses plus anchor-knot/class-star replay.

No production certificate primitive is called. Intended for auditing the six
new joint-screen admissions, rather than running a new optimization.
"""
import numpy as np
from independent_signed_classwise import SignedWitnessValidator


def pad(v):
    return float(np.clip(v+1e-10*(1+abs(v)),0.,1.))


def nearest_and_chosen(A,y,wide):
    lo,hi=np.maximum(0.,A-.01),np.minimum(1.,A+.01)
    nearest=[];chosen=[]
    for i in range(len(A)):
        enemies=np.flatnonzero(y!=y[i]); dist=np.zeros(len(enemies))
        for a,w in enumerate(wide):
            dist+=abs(A[i,a]*w-A[enemies,a]*w)
        j=int(enemies[int(np.argmin(dist))]);nearest.append(j)
        competitors=enemies[enemies!=j];upper=np.zeros(len(competitors))
        for a,w in enumerate(wide):
            if w==0:continue
            # Each row is one possible enemy; explicitly enumerate every knot.
            points=[np.full(len(competitors),t) for t in (lo[i,a],hi[i,a])]
            points += [np.clip(v,lo[i,a],hi[i,a]) for v in
                       (np.full(len(competitors),lo[j,a]),np.full(len(competitors),hi[j,a]),
                        lo[competitors,a],hi[competitors,a],
                        np.full(len(competitors),(lo[j,a]+hi[j,a])/2))]
            values=[]
            for t in points:
                far=np.maximum(abs(t-lo[j,a]),abs(t-hi[j,a]))
                near=np.maximum(0.,np.maximum(lo[competitors,a]-t,t-hi[competitors,a]))
                values.append(far-near)
            upper+=w*np.max(values,axis=0)
        stable=len(upper)==0 or float(max(upper)) < -1e-12
        distance_cap=sum(w*max(abs(lo[i,a]-hi[j,a]),abs(hi[i,a]-lo[j,a])) for a,w in enumerate(wide))
        chosen.append(stable and distance_cap <= 1.-1e-12)
    return np.asarray(nearest),np.asarray(chosen)


def scalar_star_total(lo,hi,edges,coefficients):
    total=0.;stars=0
    for a,coefficient in enumerate(coefficients):
        for center in sorted({j for _,j in edges}):
            leaves=[i for i,j in edges if j==center]
            points={float(lo[center,a]),float(hi[center,a])}
            for i in leaves:
                points.add(float(np.clip(lo[i,a],lo[center,a],hi[center,a])))
                points.add(float(np.clip(hi[i,a],lo[center,a],hi[center,a])))
            values=[]
            for t in points:
                distances=[max(abs(t-lo[i,a]),abs(t-hi[i,a])) if coefficient>=0
                           else max(0.,lo[i,a]-t,t-hi[i,a]) for i in leaves]
                values.append(float(coefficient)*sum(distances))
            total+=max(values);stars+=1
    return float(total),stars


def audit_new_signed(A,y,fs,wf,wr,saved_report):
    """Reconstruct signed/star minima from geometry and preserved legacy bound."""
    validator=SignedWitnessValidator(A,y,wf)
    recorded=saved_report['signed_object_report'];base=recorded['base_dependency']
    values=[];objects=recorded['objects']
    assert len(objects)==len(A)
    for i,record in enumerate(objects):
        fresh=validator.object(i,fs,wr)
        for key in ('object','possible_enemies','full_witnesses','uniform_unsaturation'):
            assert fresh[key]==record[key],(i,key)
        for key in ('candidate_signed_upper_bounds','reduced_uniform_upper_cap','raw_signed_upper_bound','upper_bound'):
            np.testing.assert_allclose(fresh[key],record[key],atol=1e-10,rtol=1e-9)
        np.testing.assert_allclose(max(0.,fresh['raw_signed_upper_bound']),base['redistributed_object_bounds'][i],atol=1e-10,rtol=1e-9)
        values.append(fresh['upper_bound'])
    np.testing.assert_allclose(values,recorded['signed_object_upper_bounds'],atol=1e-10,rtol=1e-9)
    values=np.asarray(values);wide=np.zeros(A.shape[1]);wide[fs]=wr
    nearest,chosen=nearest_and_chosen(A,y,wide)
    lo,hi=np.maximum(0.,A-.01),np.minimum(1.,A+.01)
    old_values=np.asarray(base['redistributed_object_bounds']);classes,counts=np.unique(y,return_counts=True)
    star_bounds=[];signed_bounds=[];total_stars=0
    original=saved_report['original_class_certificate']
    for ci,c in enumerate(classes):
        own=y==c;selected=np.flatnonzero(own&chosen)
        edges=[(int(i),int(nearest[i])) for i in selected]
        total,stars=scalar_star_total(lo,hi,edges,np.asarray(wf)-wide);total_stars+=stars
        old_raw=(total+old_values[own&~chosen].sum())/counts[ci]
        star_bound=min(original['upper_bounds'][ci],pad(old_raw));star_bounds.append(star_bound)
        raw_independent=float(values[own].mean());raw_star=float((total+values[own&~chosen].sum())/counts[ci])
        bound=min(star_bound,pad(raw_independent),pad(raw_star));signed_bounds.append(bound)
        report=saved_report['class_reports'][ci]
        assert report['class']==c and report['count']==counts[ci]
        for key,fresh in (('signed_objectwise_raw_mean',raw_independent),
                          ('shared_star_plus_signed_remainder_raw',raw_star),('upper_bound',bound)):
            np.testing.assert_allclose(report[key],fresh,atol=1e-10,rtol=1e-9)
        assert report['negative_object_bounds']==int(np.sum(values[own]<-1e-12))
    np.testing.assert_allclose(star_bounds,saved_report['star_class_certificate']['upper_bounds'],atol=1e-10,rtol=1e-9)
    np.testing.assert_allclose(signed_bounds,saved_report['class_certificate']['upper_bounds'],atol=1e-10,rtol=1e-9)
    glob=min(base['upper_bound'],float(counts@star_bounds/len(y)),float(counts@signed_bounds/len(y)))
    np.testing.assert_allclose(glob,saved_report['global_upper_bound'],atol=1e-10,rtol=1e-9)
    return {'signed_objects':len(A),'class_bounds':len(classes),'stable_star_leaves':int(chosen.sum()),
            'scalar_star_coordinate_maxima':total_stars}
