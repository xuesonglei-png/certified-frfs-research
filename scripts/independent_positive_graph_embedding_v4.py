"""Actual memberships/positive parts/nearest selectors in frozen V4 geometry.

Uses the builder, never solver. Finds pair, membership, active and one-hot
variables from their sparse incidence patterns, without production metadata
about their indices. Works with signed/positive objectives and both abs modes.
"""
import numpy as np
from shared_graph_dependency_v4 import build


def scalar_values(Z,y,w):
    answer=[]
    for i in range(len(Z)):
        values=[]
        for j in range(len(Z)):
            if y[i]!=y[j]:values.append(sum(float(w[a])*abs(float(Z[i,a])-float(Z[j,a])) for a in range(Z.shape[1])))
        answer.append(min(1.,min(values)))
    return np.asarray(answer)


class TrueEmbedding:
    def __init__(self,A,y,full,reduced,target,*,eps=.01,positive=True,exact_abs=False,object_upper=None,enemy_upper=None):
        self.A,self.y,self.full,self.reduced,self.target=np.asarray(A),np.asarray(y),np.asarray(full),np.asarray(reduced),target
        self.positive,self.eps=positive,eps;self.own=np.flatnonzero(self.y==target)
        self.model,self.z,self.meta=build(A,y,full,reduced,target,eps=eps,positive=positive,
            exact_abs=exact_abs,object_upper=object_upper,enemy_upper=enemy_upper)
        sparse,self.rhs,self.cost=self.model.arrays();self.csc=sparse;self.matrix=sparse.tocsr()
        nd=self.z.size;integer=np.asarray(self.model.integer);self.integer=np.flatnonzero(integer)
        self.pairs={};self.groups=[];self.active={};self.signs=[]
        for q in range(nd,len(self.cost)):
            if integer[q] or self.cost[q]!=0.:continue
            incident=sparse.indices[sparse.indptr[q]:sparse.indptr[q+1]]
            for row in incident:
                cols=self.matrix.indices[self.matrix.indptr[row]:self.matrix.indptr[row+1]]
                vals=self.matrix.data[self.matrix.indptr[row]:self.matrix.indptr[row+1]]
                if len(cols)==3:
                    coords=[j for j in cols if j<nd]
                    if len(coords)==2 and all(abs(v)==1. for v in vals):
                        assert coords[0]%self.A.shape[1]==coords[1]%self.A.shape[1]
                        self.pairs[q]=tuple(coords);break
        assert len(self.pairs)==self.meta['shared_pair_coordinates'] and len(set(self.pairs.values()))==len(self.pairs)
        if positive:
            self.p=np.flatnonzero(self.cost<0);self.t=[];self.r=[]
            assert len(self.p)==len(self.own) and not np.any(self.cost>0)
            for p in self.p:
                found=False
                incident=sparse.indices[sparse.indptr[p]:sparse.indptr[p+1]]
                for row in incident:
                    cols=self.matrix.indices[self.matrix.indptr[row]:self.matrix.indptr[row+1]]
                    vals=self.matrix.data[self.matrix.indptr[row]:self.matrix.indptr[row+1]]
                    if len(cols) not in (2,3,4):continue
                    entries=dict(zip(cols,vals,strict=True))
                    if len(cols)==2 and entries.get(p)==1.:
                        # The upper-link row is p - U*a <= 0, so the
                        # active binary has a negative coefficient.  A zero
                        # big-M branch removes the other occurrence of a.
                        acts=[j for j in cols if j!=p and integer[j] and entries[j]<0.]
                        if len(acts)==1:
                            self.active[int(p)]=int(acts[0])
                            continue
                    if entries.get(p)!=1.:continue
                    ts=[j for j in cols if entries[j]==-1. and not integer[j]]
                    rs=[j for j in cols if entries[j]==1. and j!=p and not integer[j]]
                    acts=[j for j in cols if entries[j]>0. and integer[j]]
                    if len(ts)==len(rs)==1 and len(acts)<=1 and self.rhs[row] >= -1e-12:
                        self.t.extend(ts);self.r.extend(rs)
                        if acts:self.active[int(p)]=int(acts[0])
                        found=True;break
                if not found:
                    # Production allocates t,r,p consecutively; p is constant zero.
                    assert self.model.upper[p]==0. and self.model.lower[p]==0.
                    assert not integer[p-1] and not integer[p-2] and p-1 not in self.pairs and p-2 not in self.pairs
                    self.t.append(int(p-2));self.r.append(int(p-1))
            self.t=np.asarray(self.t);self.r=np.asarray(self.r)
        else:
            self.p=[];self.t=np.flatnonzero(self.cost<0);self.r=np.flatnonzero(self.cost>0)
        for row in range(len(self.rhs)):
            cols=self.matrix.indices[self.matrix.indptr[row]:self.matrix.indptr[row+1]]
            vals=self.matrix.data[self.matrix.indptr[row]:self.matrix.indptr[row+1]]
            if self.rhs[row]==1. and len(cols) and all(integer[j] and v==1. for j,v in zip(cols,vals,strict=True)):
                self.groups.append(tuple(cols))
        self.groups=list(dict.fromkeys(self.groups));selected={j for group in self.groups for j in group}
        self.signs=[j for j in self.integer if j not in selected and j not in self.active.values()]
        if not exact_abs:assert not self.signs

    def table(self,Z):
        Z=np.asarray(Z);assert Z.shape==self.A.shape and np.isfinite(Z).all()
        low,high=np.maximum(0.,self.A-self.eps),np.minimum(1.,self.A+self.eps)
        assert np.all(Z>=low-1e-12) and np.all(Z<=high+1e-12)
        x=np.zeros(len(self.cost));x[self.z]=Z
        for q,(left,right) in self.pairs.items():x[q]=abs(x[left]-x[right])
        full=scalar_values(Z,self.y,self.full);red=scalar_values(Z,self.y,self.reduced);delta=full-red
        x[self.t]=full[self.own];x[self.r]=red[self.own]
        if self.positive:
            x[self.p]=np.maximum(0.,delta[self.own])
            for i,p in zip(self.own,self.p,strict=True):
                if p in self.active:x[self.active[p]]=float(delta[i]>0.)
        for sign in self.signs:
            # Only the two q sign disjunctions contain this binary.
            rows=self.csc.indices[self.csc.indptr[sign]:self.csc.indptr[sign+1]]
            x[sign]=0.
            if np.max(self.matrix[rows]@x-self.rhs[rows])>1e-10:x[sign]=1.
            assert np.max(self.matrix[rows]@x-self.rhs[rows])<=1e-10
        gp=0
        for obj in self.meta['selections']:
            possible=obj['possible_enemies'];sat=obj['saturation_branch'];count=len(possible)+int(sat);i=obj['object']
            if count<=1:continue
            group=self.groups[gp];gp+=1;assert len(group)==count
            if red[i]>=1.:
                assert sat;x[group[-1]]=1.
            else:
                distances=[sum(float(w)*abs(float(Z[i,a])-float(Z[j,a])) for a,w in enumerate(self.reduced)) for j in possible]
                chosen=int(np.argmin(distances));np.testing.assert_allclose(distances[chosen],red[i],atol=1e-12,rtol=0)
                x[group[chosen]]=1.
        assert gp==len(self.groups)
        residual=max(0.,float(np.max(self.matrix@x-self.rhs)),float(np.max(np.asarray(self.model.lower)-x)),float(np.max(x-np.asarray(self.model.upper))))
        assert residual<=1e-8,residual
        loss=float((np.maximum(0.,delta[self.own]) if self.positive else delta[self.own]).mean())
        np.testing.assert_allclose(-self.cost@x,loss,atol=1e-12,rtol=0)
        return loss,residual
