"""Deployable fixed-order restoration for SIGNED class-mean FRS targets.

Legacy positive-object loss guards are retained in each record for comparison.
The new acceptance target is signed class-mean decrease on the same finite
table and box, with unchanged original majority/mass/cap. It does not preserve
the stronger mean-positive-object-loss guarantee at newly admitted points.
"""
from __future__ import annotations
import numpy as np
from shared_graph_dependency_v2 import class_upper
from signed_classwise_dependency import signed_classwise_dependency

FAMILIES=('uniform_sqrt','half_majority')
FRACTIONS=(.5,.7,.85,1.)
MODES=('legacy','signed','graph')
SCREEN=.13
CLASS_SECONDS=8.


def enemy_cuts(A,y,features,reduced,signed):
    wide=np.zeros(A.shape[1]);wide[features]=reduced
    lo,hi=np.maximum(0.,A-.01),np.minimum(1.,A+.01);result={}
    for item in signed['signed_object_report']['objects']:
        i=item['object'];possible=item['possible_enemies']
        lower=np.maximum(0.,np.maximum(lo[i]-hi[possible],lo[possible]-hi[i]))@wide
        result[i]={j:min(float(1.-l),float(u)) for j,l,u in zip(
            possible,lower,item['candidate_signed_upper_bounds'],strict=True)}
    return wide,result


class GuardCache:
    """One training table only; deduplicate equal reference/metric LPs."""
    def __init__(self,A,y,*,signed_fn=None,graph_fn=None,state=None,checkpoint=None):
        self.A=np.asarray(A,float);self.y=np.asarray(y);self.classes=np.unique(y)
        self.signed_fn=signed_classwise_dependency if signed_fn is None else signed_fn
        self.graph_fn=class_upper if graph_fn is None else graph_fn
        self.refs={};self.graphs={};self.solves={};self.checkpoint=checkpoint
        self.counters={'signed_reference_evaluations':0,
            'graph_class_calls':0,'graph_reference_cache_hits':0}
        if state is not None:
            self.counters=state['counters']
            for entry in state['references']:
                self.refs[tuple(tuple(x) for x in entry['key'])]=entry['signed_report']
            for entry in state['solves']:
                identity=tuple(tuple(x) for x in entry['key'])
                self.solves[(identity,entry['target_class'])]=entry['solve']

    def snapshot(self):
        return {'counters':self.counters,
            'references':[{'key':[list(x) for x in k],'signed_report':v} for k,v in self.refs.items()],
            'solves':[{'key':[list(x) for x in k],'target_class':label,'solve':v}
                      for (k,label),v in self.solves.items()]}

    def _checkpoint(self):
        if self.checkpoint is not None:self.checkpoint(self.snapshot())

    def _signed(self,candidate,full,dep):
        key=(tuple(full),tuple(candidate['features']),tuple(candidate['weights']))
        reused=key in self.refs
        if not reused:
            signed=self.signed_fn(self.A,self.y,candidate['features'],np.asarray(full),
                np.asarray(candidate['weights']),base_dependency=dep)
            self.refs[key]=signed;self.counters['signed_reference_evaluations']+=1;self._checkpoint()
        return key,self.refs[key],reused

    def assess(self,candidate,reference,*,mode):
        if mode not in MODES:raise ValueError('unknown restoration mode')
        c=candidate;majority=bool(c['certificate']['majority_feasible'] and c['family_certificate']['majority_feasible'])
        base={'mode':mode,'legacy_positive_joint_feasible':bool(c['joint_feasible']),
              'original_majority_feasible':majority,'guards':[]}
        if mode=='legacy':return {**base,'accepted':bool(c['joint_feasible']),'reason':'legacy_positive_guard'}
        if not majority:return {**base,'accepted':False,'reason':'unchanged_original_majority_rejection'}
        if c['joint_feasible']:
            # Stronger old class-positive guarantee implies the signed target.
            return {**base,'accepted':True,'reason':'legacy_positive_implies_signed',
                    'graph_called':False,'signed_joint_feasible':True}
        d=self.A.shape[1];uniform=np.ones(d)/np.sqrt(d);prepared=[]
        for side,full,key in (('original',uniform,'certificate'),('family',np.asarray(reference),'family_certificate')):
            identity,signed,reused=self._signed(c,full,c[key]['dependency_certificate'])
            bounds=list(signed['class_certificate']['upper_bounds'])
            assert len(bounds)==len(self.classes) and all(np.isfinite(bounds))
            prepared.append((side,identity,full,signed,bounds,reused))
        signed_pass=all(all(b<=.1 for b in bounds) for _,_,_,_,bounds,_ in prepared)
        max_signed=max(b for _,_,_,_,bounds,_ in prepared for b in bounds)
        guards=[{'side':side,'full_weights':full.tolist(),'signed_class_bounds':bounds,
                 'signed_global_upper_bound':float(np.bincount(np.searchsorted(self.classes,self.y))@np.asarray(bounds)/len(self.y)),
                 'signed_cache_key':[list(x) for x in identity], 'reused_identical_reference':reused}
                for side,identity,full,signed,bounds,reused in prepared]
        base.update({'guards':guards,'signed_joint_feasible':signed_pass,'worst_signed_class_bound':max_signed})
        if signed_pass:return {**base,'accepted':True,'reason':'signed_class_guard','graph_called':False}
        if mode=='signed':return {**base,'accepted':False,'reason':'signed_class_rejection','graph_called':False}
        if max_signed>SCREEN:
            return {**base,'accepted':False,'reason':'prospective_graph_resource_screen','graph_called':False}
        for guard,(side,identity,full,signed,bounds,reused) in zip(guards,prepared,strict=True):
            if identity in self.graphs:
                saved=self.graphs[identity];self.counters['graph_reference_cache_hits']+=1
                guard.update(saved);guard['reused_graph_reference']=True;continue
            wide,cuts=enemy_cuts(self.A,self.y,c['features'],np.asarray(c['weights']),signed)
            object_upper=np.asarray(signed['signed_object_report']['signed_object_upper_bounds'])
            reports=[];newbounds=[]
            for label,old in zip(self.classes,bounds,strict=True):
                if old<=.1:
                    reports.append({'target_class':label.item(),'skipped_existing_signed_feasible':True,
                        'old_upper_bound':old,'new_upper_bound':old});newbounds.append(old);continue
                solvekey=(identity,label.item())
                if solvekey not in self.solves:
                    self.solves[solvekey]=self.graph_fn(self.A,self.y,full,wide,label,seconds=CLASS_SECONDS,
                                           object_upper=object_upper,enemy_upper=cuts)
                    self.counters['graph_class_calls']+=1;self._checkpoint()
                solve=self.solves[solvekey]
                # Missing-dual timeout is unresolved: do not accept an incumbent.
                bound=min(old,float(solve['upper_bound'])) if solve['solver_bound_available'] else old
                reports.append({'target_class':label.item(),'skipped_existing_signed_feasible':False,
                    'old_upper_bound':old,'new_upper_bound':bound,'solve':solve});newbounds.append(bound)
            report={'graph_classes':reports,'new_class_bounds':newbounds,
                    'graph_reference_feasible':all(b<=.1 for b in newbounds)}
            self.graphs[identity]=report;guard.update(report);guard['reused_graph_reference']=False
        accepted=all(g['graph_reference_feasible'] for g in guards)
        return {**base,'accepted':accepted,'reason':'graph_class_guard' if accepted else 'graph_class_unresolved_or_rejected',
                'graph_called':True,'graph_joint_feasible':accepted,
                'certificate_scope':'signed_same_table_class_mean_decrease_not_mean_positive_object_losses'}


def restored(A,y,reference,ranking,requested,mode,provide,fallback,guard_cache):
    """Fit-data-only restoration, first accepted prefix or checked uniform full.

    provide(count) supplies the unmodified cached V4 metric/guards. Prediction
    labels are absent from this API. Every visited count is explicit; admission
    need not be monotone. Uniform fallback carries its own original reference.
    """
    d=A.shape[1]
    if sorted(ranking)!=list(range(d)) or not int(np.ceil(d/2))<=requested<=d:
        raise ValueError('fixed permutation and unchanged cardinality floor required')
    trace=[]
    for count in range(requested,d+1):
        c=provide(count);assert c['features']==sorted(ranking[:count])
        report=guard_cache.assess(c,reference,mode=mode)
        trace.append({'count':count,'candidate':c,'admission':report})
        if report['accepted']:
            return {'model':c,'admission':report,'requested_count':requested,
                    'restored_count':count-requested,'uniform_fallback':False,'trace':trace}
    c=fallback();assert c['features']==list(range(d)) and c['joint_feasible']
    uniform=np.ones(d)/np.sqrt(d)
    np.testing.assert_array_equal(c['weights'],uniform)
    report=guard_cache.assess(c,uniform,mode=mode);assert report['accepted']
    trace.append({'count':d,'candidate':c,'admission':report,'explicit_uniform_fallback':True})
    return {'model':c,'admission':report,'requested_count':requested,
            'restored_count':d-requested,'uniform_fallback':True,'trace':trace}


def summarize(blocks,mode):
    options=[]
    for family in FAMILIES:
        for i,fraction in enumerate(FRACTIONS):
            rows=[b['families'][family]['modes'][mode][i] for b in blocks]
            options.append({'family':family,'recipe_index':i,'fraction':fraction,
                'mean_BA':float(np.mean([r['score']['clean_BA'] for r in rows])),
                'mean_certified_BA':float(np.mean([r['score']['query_certified_BA'] for r in rows])),
                'mean_count':float(np.mean([len(r['model']['features']) for r in rows]))})
    clean=max(options,key=lambda r:(r['mean_BA'],r['mean_certified_BA'],-r['mean_count'],
                                    -FAMILIES.index(r['family']),-r['recipe_index']))
    best=max(r['mean_BA'] for r in options)
    compact=max((r for r in options if r['mean_BA']>=best-.005-1e-12),
        key=lambda r:(-r['mean_count'],r['mean_BA'],r['mean_certified_BA'],
                      -FAMILIES.index(r['family']),-r['recipe_index']))
    return {'options':options,'clean_first':clean,'compression_policy':compact}
