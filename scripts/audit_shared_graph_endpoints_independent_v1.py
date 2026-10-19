"""Sparse actual-table embedding audit of frozen graph-bound screens.

Calls build only, never MILP/class_upper/attack/training. Checks every matrix
row at true q/t/r and actual nearest-enemy selector assignments. Solver dual
multipliers were not saved, so the reported dual number cannot be independently
reproved here. The baseline signed/star values remain frozen inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib,json,time,traceback
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from audit_classwise_shared_attack_all_rows_v1 import scalar_memberships
from shared_graph_dependency_v1 import build as build_v1
from shared_graph_dependency_v2 import build as build_v2

NAMES=('banknote','glass','heart_statlog','ionosphere','seeds','wdbc')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def enemy_cuts_from_saved(A,full_red,signed):
    """Separate scalar coordinate summation for conditional 1-LB caps."""
    lo,hi=np.maximum(0.,A-.01),np.minimum(1.,A+.01);answer={}
    for row in signed['signed_object_report']['objects']:
        i=row['object'];values={}
        for j,contrast in zip(row['possible_enemies'],row['candidate_signed_upper_bounds'],strict=True):
            lower=sum(float(w)*max(0.,float(lo[i,a]-hi[j,a]),float(lo[j,a]-hi[i,a]))
                      for a,w in enumerate(full_red))
            values[int(j)]=min(1.-lower,float(contrast))
        answer[int(i)]=values
    return answer


class SparseEmbedding:
    def __init__(self,A,y,full,reduced,target,signed,version,fit):
        self.A,self.y,self.full,self.reduced,self.target=A,y,np.asarray(full),np.asarray(reduced),target
        objects=signed['signed_object_report']['signed_object_upper_bounds']
        kwargs={'eps':.01,'object_upper':objects}
        if version==2:kwargs['enemy_upper']=enemy_cuts_from_saved(A,reduced,signed)
        builder=build_v1 if version==1 else build_v2
        self.model,self.z,self.meta=builder(A,y,full,reduced,target,**kwargs)
        matrix,self.rhs,self.cost=self.model.arrays();self.matrix=matrix.tocsr()
        self.integer=np.flatnonzero(self.model.integer);self.lower=np.asarray(self.model.lower);self.upper=np.asarray(self.model.upper)
        self.t=np.flatnonzero(self.cost<0);self.r=np.flatnonzero(self.cost>0);self.own=np.flatnonzero(y==target)
        self.pairs={};nd=self.z.size
        qindices=np.flatnonzero((np.arange(len(self.cost))>=nd)&(np.asarray(self.model.integer)==0)&(self.cost==0))
        for q in qindices:
            # The first incidence is a defining absolute equality/epigraph row.
            indices=matrix.indices[matrix.indptr[q]:matrix.indptr[q+1]];row=int(indices.min())
            cols=self.matrix.indices[self.matrix.indptr[row]:self.matrix.indptr[row+1]]
            vals=self.matrix.data[self.matrix.indptr[row]:self.matrix.indptr[row+1]]
            entries=dict(zip(cols,vals,strict=True));coords=[j for j in cols if j<nd]
            assert len(cols)==3 and len(coords)==2 and q in entries
            assert all(abs(entries[j])==1. for j in cols)
            assert coords[0]%A.shape[1]==coords[1]%A.shape[1]
            assert coords[0]//A.shape[1]!=coords[1]//A.shape[1]
            self.pairs[int(q)]=tuple(coords)
        assert len(set(self.pairs.values()))==len(self.pairs)==self.meta['shared_pair_coordinates']
        assert len(self.t)==len(self.r)==len(self.own)
        assert not self.meta['exact_abs']
        for key,value in self.meta.items():assert fit[key]==value,(version,key)
        lo,hi=np.maximum(0.,A-.01),np.minimum(1.,A+.01)
        for obj in self.meta['selections']:
            i=obj['object'];enemies=np.flatnonzero(y!=y[i]);lower={};upper={}
            for j in enemies:
                lower[int(j)]=sum(float(w)*max(0.,float(lo[i,a]-hi[j,a]),float(lo[j,a]-hi[i,a])) for a,w in enumerate(reduced))
                upper[int(j)]=sum(float(w)*max(abs(float(lo[i,a]-hi[j,a])),abs(float(hi[i,a]-lo[j,a]))) for a,w in enumerate(reduced))
            cap=min(upper.values())
            for j in enemies:
                if j in obj['possible_enemies']:assert lower[int(j)]<=cap+2e-12
                else:assert lower[int(j)]>cap-2e-12
            np.testing.assert_allclose(obj['reduced_lower'],min(1.,min(lower.values())),atol=1e-12,rtol=0)
            np.testing.assert_allclose(obj['reduced_upper'],min(1.,cap),atol=1e-12,rtol=0)
            assert obj['saturation_branch']==(cap>=1.-1e-12)
            assert all(y[k]!=y[i] for k in obj['full_witnesses']) and 1<=len(obj['full_witnesses'])<=8

    def table(self,Z):
        Z=np.asarray(Z);assert Z.shape==self.A.shape and np.isfinite(Z).all()
        assert np.all(Z>=np.maximum(0.,self.A-.01)-1e-12)
        assert np.all(Z<=np.minimum(1.,self.A+.01)+1e-12)
        x=np.zeros(len(self.cost));x[self.z]=Z
        for q,(left,right) in self.pairs.items():x[q]=abs(x[left]-x[right])
        full=scalar_memberships(Z,self.y,range(Z.shape[1]),self.full)
        reduced=scalar_memberships(Z,self.y,range(Z.shape[1]),self.reduced)
        x[self.t]=full[self.own];x[self.r]=reduced[self.own];position=0
        for obj in self.meta['selections']:
            i=obj['object'];possible=obj['possible_enemies'];saturated=obj['saturation_branch']
            count=len(possible)+int(saturated)
            if count>1:
                group=self.integer[position:position+count];assert len(group)==count;position+=count
                if reduced[i]>=1.:
                    assert saturated;x[group[-1]]=1.
                else:
                    distances=[sum(float(w)*abs(float(Z[i,a])-float(Z[j,a])) for a,w in enumerate(self.reduced)) for j in possible]
                    chosen=int(np.argmin(distances))
                    np.testing.assert_allclose(distances[chosen],reduced[i],atol=1e-12,rtol=0)
                    x[group[chosen]]=1.
        assert position==len(self.integer)
        residual=max(0.,float(np.max(self.matrix@x-self.rhs)),float(np.max(self.lower-x)),float(np.max(x-self.upper)))
        assert residual<=1e-8,residual
        loss=float((full-reduced)[self.own].mean())
        np.testing.assert_allclose(-self.cost@x,loss,atol=1e-12,rtol=0)
        return loss,residual


def audit(version):
    folder=ROOT/('results/shared-graph-dependency-v'+str(version));summary=json.loads((folder/'summary.json').read_text())
    assert summary['complete'];hashes=json.loads((folder/'source-freeze.json').read_text())['hashes']
    for rel,sha in hashes.items():assert digest(ROOT/rel)==sha,rel
    tally=Counter();details=[];newdeps=[];newjoints=[];status=Counter();tightenings=[];max_residual=0.;start=time.perf_counter()
    for name in NAMES:
        path=folder/(name+'.json');saved=json.loads(path.read_text());assert digest(path)==summary['artifact_hashes'][name]
        assert saved['complete'] and saved['source_hashes']==hashes and not saved['new_classifier_predictions']
        parent=json.loads((ROOT/'results/repair-relation-selector-v4'/(name+'.json')).read_text())
        attack=json.loads((ROOT/'results/classwise-shared-attack-v1'/(name+'.json')).read_text())
        panel=parent['rows'][0];inner=panel['fit']['inner'][0];ids=np.asarray(panel['train_indices'])[np.asarray(inner['fit_indices'])]
        X,y=load_dataset(name);A,_=preprocess_fit(X[ids],X[ids]);y=y[ids];classes=np.unique(y)
        assert len(saved['rows'])==len(attack['reports']);cache={};datasetcounts=Counter()
        for ci,(row,old_attack) in enumerate(zip(saved['rows'],attack['reports'],strict=True)):
            assert row==json.loads((folder/name/(str(ci)+'.json')).read_text())
            assert row['source_hashes']==hashes and row['complete'] and row['candidate_index']==ci
            assert not row['new_classifier_predictions'] and not row['validation_or_outer_observations_used']
            c=old_attack['candidate'];fs=c['features'];wr=c['weights'];wide=np.zeros(A.shape[1]);wide[fs]=wr
            assert row['family']==old_attack['family'] and row['stratum']==old_attack['stratum']
            assert row['features']==fs and row['weights']==wr and row['known_violation']==old_attack['class_violation_witness']
            assert len(row['guards'])==2
            for side,ck,g in zip(('original','family'),('certificate','family_certificate'),row['guards'],strict=True):
                wf=np.asarray(c[ck]['dependency_certificate'].get('full_weights',wr));np.testing.assert_array_equal(g['full_weights'],wf)
                assert g['side']==side;identity=tuple(wf),tuple(fs),tuple(wr)
                if identity in cache:
                    assert g['reused_identical_metric']
                    assert {k:v for k,v in g.items() if k not in ('side','reused_identical_metric')}==cache[identity]
                    tally['reused_identical_guards']+=1
                    continue
                assert not g['reused_identical_metric'];cache[identity]={k:v for k,v in g.items() if k not in ('side','reused_identical_metric')}
                signed=g['signed_report'];np.testing.assert_allclose(g['signed_class_bounds'],signed['class_certificate']['upper_bounds'],atol=0,rtol=0)
                assert len(g['classes'])==len(classes);bounds=[]
                for label,bound,record in zip(classes,g['signed_class_bounds'],g['classes'],strict=True):
                    assert record['target_class']==label and record['old_upper_bound']==bound
                    assert record['skipped_existing_feasible']==(bound<=.1)
                    if record['skipped_existing_feasible']:
                        assert record['new_upper_bound']==bound;bounds.append(bound);tightenings.append(0.);continue
                    fit=record['solve'];status[str(fit['solver_status'])]+=1;tally['solver_calls']+=1
                    upper=fit['solver_upper_bound'];allowance=1e-7*(1+abs(upper))
                    assert np.isfinite(upper) and fit['numerical_allowance']==allowance
                    if not fit['solver_bound_available']:assert upper==1.
                    np.testing.assert_allclose(fit['upper_bound'],float(np.clip(upper+allowance,0.,1.)),atol=1e-12,rtol=0)
                    combined=min(bound,fit['upper_bound']);assert record['new_upper_bound']==combined
                    bounds.append(combined);tightenings.append(bound-combined)
                    embedding=SparseEmbedding(A,y,wf,wide,label,signed,version,fit)
                    nominal,residual=embedding.table(A);max_residual=max(max_residual,residual)
                    np.testing.assert_allclose(nominal,fit['nominal_signed_loss'],atol=1e-10,rtol=0)
                    endpoint,residual=embedding.table(fit['perturbed_table']);max_residual=max(max_residual,residual)
                    np.testing.assert_allclose(endpoint,fit['observed_signed_loss_lower'],atol=1e-10,rtol=0)
                    assert endpoint+1e-12>=nominal and endpoint<=upper+allowance+1e-10
                    tally['nominal_or_saved_endpoint_embeddings']+=2
                    prior_count=0
                    for previous in old_attack['attacks']:
                        if previous['fit']['target_class']==label and np.array_equal(previous['full_weights'],wf):
                            prior,residual=embedding.table(previous['fit']['perturbed_table']);max_residual=max(max_residual,residual)
                            np.testing.assert_allclose(prior,previous['fit']['observed_signed_class_loss_lower'],atol=1e-10,rtol=0)
                            assert prior<=combined+1e-7
                            tally['old_attack_embeddings']+=1;prior_count+=1
                    details.append({'dataset':name,'candidate':ci,'side':side,'class':label.item(),
                        'old_bound':bound,'new_bound':combined,'reported_dual_upper':upper,
                        'actual_loss':endpoint,'solver_status':fit['solver_status'],'old_attacks_embedded':prior_count,
                        'variables':embedding.meta['variables'],'constraints':embedding.meta['constraints']})
                np.testing.assert_allclose(g['new_class_bounds'],bounds,atol=0,rtol=0)
                assert g['signed_feasible']==all(b<=.1 for b in g['signed_class_bounds'])
                assert g['new_feasible']==all(b<=.1 for b in bounds)
            signedpass=all(g['signed_feasible'] for g in row['guards']);newpass=all(g['new_feasible'] for g in row['guards'])
            majority=c['certificate']['majority_feasible'] and c['family_certificate']['majority_feasible']
            assert row['legacy_joint_feasible']==c['joint_feasible']
            assert row['signed_dependency_feasible']==signedpass and row['new_dependency_feasible']==newpass
            assert row['original_majority_feasible']==majority
            assert row['signed_joint_feasible']==(signedpass and majority) and row['new_joint_feasible']==(newpass and majority)
            assert not row['known_violation'] or not newpass
            item={'dataset':name,'candidate':ci,'family':row['family'],'features':fs}
            if newpass and not signedpass:newdeps.append(item)
            if row['new_joint_feasible'] and not row['signed_joint_feasible']:newjoints.append(item)
            datasetcounts['candidates']+=1;datasetcounts['known_violations']+=row['known_violation']
            datasetcounts['new_dependency_admissions']+=newpass and not signedpass
            datasetcounts['new_joint_admissions']+=row['new_joint_feasible'] and not row['signed_joint_feasible']
        assert dict(datasetcounts)==summary['dataset_counts'][name]
        tally.update({'candidates':datasetcounts['candidates'],'known_violations':datasetcounts['known_violations']})
        print('v'+str(version),name,'sparse actual endpoint/old attack/flags audit passed',flush=True)
    assert tally['candidates']==summary['candidates']==21 and tally['known_violations']==summary['known_violations']==9
    assert tally['solver_calls']==summary['solver_calls'] and dict(status)==summary['solver_status_counts']
    assert len(newdeps)==summary['new_dependency_admissions'] and len(newjoints)==summary['new_joint_admissions']
    gate=len(newdeps)>=3 and len({r['dataset'] for r in newdeps})>=2 and len(newjoints)>=1
    assert gate==summary['resource_gate']
    np.testing.assert_allclose(float(np.mean(tightenings)),summary['mean_class_tightening'],atol=1e-12,rtol=0)
    result={'passed':True,'version':version,'checks':dict(tally),'maximum_embedding_residual':max_residual,
        'new_dependency_admissions':newdeps,'new_joint_admissions':newjoints,'resource_gate':gate,
        'classes':details,'seconds':time.perf_counter()-start,'auditor_sha256':digest(__file__),
        'scope':'shared_box_actual_membership_embedding_saved_endpoints_old_attacks_and_all_admission_flags',
        'new_solver_or_fit_or_attack_runs':False,'classification_superiority_established':False,
        'limits':['Graph builder and preprocessing reused; true membership/selector assignment and matrix residual replay are independent.',
                  'Finite solver dual values are recorded inputs: no dual multipliers or formally checked proof were saved.',
                  'Existing signed/star and legacy majority bounds remain frozen certificate inputs.',
                  'This verifies signed class-mean loss, not average individual positive membership losses.',
                  'Training reference certification gives no test accuracy or generalization guarantee.']}
    out=folder/'independent-endpoint-audit-v1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='classes'},indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',type=int,choices=(1,2),required=True);args=p.parse_args();threadpool_limits(1)
    try:audit(args.version)
    except Exception:
        folder=ROOT/('results/shared-graph-dependency-v'+str(args.version))
        (folder/('independent-endpoint-audit-failure-'+str(time.time_ns())+'.json')).write_text(json.dumps({'traceback':traceback.format_exc()},indent=2))
        raise
