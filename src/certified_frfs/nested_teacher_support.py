"""Complete inner refitting for certificate-constrained teacher supports."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from .validated_pool import preprocess_fit
from .linear_retention import linear_parameters
from .teacher_support import reconstruction_supports
from .feasible_transport import weight_certificate
from .majority_metric import MajorityTerms
from .distilled_box_linear_refit import fit_distilled_box_linear

FRACTIONS=(.5,.65,.8,1.)

def inner_splits(y,seed=20261010):
    y=np.asarray(y);labels,counts=np.unique(y,return_counts=True)
    repeated=np.flatnonzero(np.isin(y,labels[counts>1]))
    single=np.flatnonzero(np.isin(y,labels[counts==1]))
    if not len(repeated):raise ValueError('inner selection requires repeated classes')
    cv=StratifiedKFold(min(3,int(counts[counts>1].min())),shuffle=True,random_state=seed)
    return [(np.r_[repeated[a],single],repeated[b]) for a,b in cv.split(repeated,y[repeated])]

def fit_stage(raw,y):
    """No query argument: all teachers, supports and certificates are training-only."""
    raw=np.asarray(raw,float);y=np.asarray(y)
    A,_=preprocess_fit(raw,raw);d=A.shape[1];fullweights=np.ones(d)/np.sqrt(d)
    teacher=LogisticRegression(C=1,max_iter=2000,random_state=0).fit(A,y)
    W,b,classes=linear_parameters(teacher)
    protected=MajorityTerms(A,y,weights=fullweights,eps=.01).bounds(range(d),allowed_enemies=2)>1e-10
    full={'features':list(range(d)),'weights':fullweights.tolist(),'origin':'full',
          'certificate':weight_certificate(A,y,list(range(d)),fullweights,fullweights,protected=protected)}
    fit=fit_distilled_box_linear(A,y,W,b,classes,list(range(d)))
    full_failure=None
    if not fit['feasible']:
        full_failure=fit
        fit={'feasible':True,'coef':W.tolist(),'intercept':b.tolist(),'scope':'explicit_full_teacher_fallback'}
    full_record={'candidate':full,'fit':fit,'fallback':full_failure is not None,'failure':full_failure}
    generated=reconstruction_supports(A,y,W,b)
    bysize={len(c['features']):c for c in generated};records={'1.0':full_record};cache={}
    for fraction in FRACTIONS[:-1]:
        k=int(np.ceil(fraction*d));key=str(fraction)
        if k==d:
            records[key]=full_record;continue
        if k in cache:
            records[key]=cache[k];continue
        c=bysize[k];fs=c['features']
        cert=weight_certificate(A,y,fs,fullweights,np.array(c['weights'])[fs],protected=protected)
        c['certificate']=cert
        result=fit_distilled_box_linear(A,y,W,b,classes,fs) if cert['feasible'] else {'feasible':False,'status':'fuzzy_certificate_failed'}
        record={'candidate':c,'fit':result,'fallback':False,'failure':None}
        if not result['feasible']:
            record={'candidate':full,'fit':fit,'fallback':True,'failure':{'requested_candidate':c,'requested_fit':result}}
        records[key]=record;cache[k]=record
    return {'teacher_coef':W.tolist(),'teacher_intercept':b.tolist(),'teacher_classes':classes.tolist(),
            'policies':records,'d':d,'scope':'independent_reference_only_complete_stage'}

def choose_policy(rows,tolerance=.005):
    """Rows contain only inner scores and actual support fractions."""
    if tolerance<0 or not rows:raise ValueError('invalid selection inputs')
    scores={str(f):float(np.mean([r['policies'][str(f)]['clean'] for r in rows])) for f in FRACTIONS}
    retained={str(f):float(np.mean([r['policies'][str(f)]['retained'] for r in rows])) for f in FRACTIONS}
    best=max(scores.values());eligible=[str(f) for f in FRACTIONS if scores[str(f)]>=best-tolerance-1e-12]
    chosen=min(eligible,key=lambda k:(retained[k],-scores[k],float(k)))
    return chosen,{'scores':scores,'retained':retained,'eligible':eligible,'tolerance':tolerance}
