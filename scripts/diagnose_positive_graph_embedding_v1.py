"""Independent embedding of the seed-839 exact-sign attack in the hull model."""
from pathlib import Path
import hashlib,itertools,json,sys,time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from shared_graph_dependency_v3 import build,class_upper
from audit_classwise_shared_attack_all_rows_v1 import scalar_memberships


def main():
    threadpool_limits(1);rng=np.random.default_rng(839)
    A=rng.uniform(.2,.8,(5,2));y=np.array([0,0,1,1,1]);full,reduced=rng.uniform(.1,1.2,(2,2))
    exact=class_upper(A,y,full,reduced,0,eps=.08,seconds=10.,exact_abs=True)
    relaxed=class_upper(A,y,full,reduced,0,eps=.08,seconds=10.)
    Z=np.asarray(exact['perturbed_table']);model,z,meta=build(A,y,full,reduced,0,eps=.08)
    sparse,rhs,cost=model.arrays();matrix=sparse.toarray();nd=z.size;x=np.zeros(len(cost));x[z]=Z
    pairs={};qindices=[]
    for q in range(nd,len(cost)):
        if model.integer[q] or cost[q]!=0:continue
        for row in matrix:
            coords=np.flatnonzero(row[:nd])
            if row[q]!=0 and len(coords)==2 and np.count_nonzero(row)==3:
                pairs[q]=tuple(coords);x[q]=abs(x[coords[0]]-x[coords[1]]);qindices.append(q);break
    full_values=scalar_memberships(Z,y,range(A.shape[1]),full)
    red_values=scalar_memberships(Z,y,range(A.shape[1]),reduced)
    losses=np.flatnonzero(cost<0);delta=full_values[:2]-red_values[:2]
    for i,loss in enumerate(losses):
        x[loss]=max(0.,delta[i]);found=False
        for row,bound in zip(matrix,rhs,strict=True):
            if row[loss]==1. and bound==1. and np.count_nonzero(row)==4:
                t=[j for j in np.flatnonzero(row==-1.) if not model.integer[j]]
                r=[j for j in np.flatnonzero(row==1.) if j!=loss and not model.integer[j]]
                active=[j for j in np.flatnonzero(row==1.) if model.integer[j]]
                assert len(t)==len(r)==len(active)==1
                x[t[0]]=full_values[i];x[r[0]]=red_values[i];x[active[0]]=float(delta[i]>0);found=True;break
        assert found
    groups=[]
    for row,bound in zip(matrix,rhs,strict=True):
        ids=np.flatnonzero(row)
        if bound==1. and len(ids) and all(model.integer[j] and row[j]==1. for j in ids):groups.append(tuple(ids))
    groups=list(dict.fromkeys(groups));assert len(groups)==2
    best=None;worst_row=None
    for choices in itertools.product(*groups):
        for group in groups:x[list(group)]=0.
        x[list(choices)]=1.;residual=matrix@x-rhs
        violation=max(0.,float(np.max(residual)),float(np.max(np.asarray(model.lower)-x)),float(np.max(x-np.asarray(model.upper))))
        if best is None or violation<best:
            best=violation;worst_row=int(np.argmax(residual));bestx=x.copy();bestchoices=choices
    attack=float(np.maximum(0.,delta).mean());objective=float(-cost@bestx)
    result={'A':A.tolist(),'y':y.tolist(),'full':full.tolist(),'reduced':reduced.tolist(),'eps':.08,
        'exact':exact,'relaxed':relaxed,'independent_attack_value':attack,'embedded_objective':objective,
        'relaxed_embedding_max_violation':best,'worst_row':worst_row,'worst_row_coefficients':{
            str(j):float(matrix[worst_row,j]) for j in np.flatnonzero(matrix[worst_row])},
        'worst_row_rhs':float(rhs[worst_row]),'selected_groups':[int(j) for j in bestchoices],
        'actual_z_table':Z.tolist(),'actual_embedding_vector':bestx.tolist(),
        'exact_attack_contradicts_reported_relaxed_dual':attack>relaxed['upper_bound'],
        'source_sha256':hashlib.sha256((ROOT/'scripts/shared_graph_dependency_v3.py').read_bytes()).hexdigest(),
        'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output=ROOT/'results'/('positive-graph-independent-embedding-'+str(time.time_ns())+'.json')
    output.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'output':str(output),'actual_attack':attack,'embedded_objective':objective,
        'embedding_violation':best,'relaxed_reported_upper':relaxed['upper_bound'],
        'exact_reported_upper':exact['upper_bound'],'contradiction':result['exact_attack_contradicts_reported_relaxed_dual']},indent=2))


if __name__=='__main__':main()
