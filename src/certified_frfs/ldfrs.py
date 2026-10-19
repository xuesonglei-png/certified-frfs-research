"""Independent LDFRS/FDM conventions, Yang et al., TFS 2023.

Legacy outputs retain exp(-difference**2/delta) and exclusive neighbors.
New sensitivity variants use bandwidth inferred from Example 1. Its density,
factor and reflexivity conventions do not consistently reproduce the equations.
Full-table influence is frozen before per-attribute relations (Algorithm 1).
No variant is an author-validated reproduction.
"""
import numpy as np
from scipy.spatial.distance import cdist


def local_influence(X,y,*,k=10,threshold=1.5,density_self=False,factor_self=None):
    """Separate neighborhood choices to audit the inconsistent paper toy.

Factor neighborhoods truncate to available same-class samples. Legacy inputs
still require k other samples; mixed inclusive-density variants can have fewer.
"""
    X,y=np.asarray(X,float),np.asarray(y)
    if (X.ndim!=2 or len(y)!=len(X) or not np.isfinite(X).all() or
            int(k)!=k or k<1 or threshold<=1 or not np.isfinite(threshold)):
        raise ValueError('invalid local-density inputs')
    if factor_self is None:factor_self=density_self
    counts=np.unique(y,return_counts=True)[1]
    minimum=max(2,k+int(not density_self))
    if len(counts)<2 or counts.min()<minimum:
        raise ValueError('insufficient same-class neighbors for the chosen convention')
    distances=cdist(X,X,'euclidean')
    if not density_self:np.fill_diagonal(distances,np.inf)
    global_neighbors=np.argsort(distances,axis=1,kind='stable')[:,:k]
    same=np.where(y[:,None]==y[None,:],distances,np.inf)
    class_neighbors=np.argsort(same,axis=1,kind='stable')[:,:k]
    indices=np.arange(len(X))[:,None]
    global_density=(1+k)/(1+distances[indices,global_neighbors].sum(axis=1))
    class_density=(1+k)/(1+distances[indices,class_neighbors].sum(axis=1))
    factor_distances=same.copy()
    np.fill_diagonal(factor_distances,0 if factor_self else np.inf)
    # Small classes have fewer than k other samples in the published toy.
    factor_neighbors=[np.argsort(row,kind='stable')[:min(k,np.isfinite(row).sum())]
                      for row in factor_distances]
    factor=np.array([class_density[js].mean()/class_density[i]
                     for i,js in enumerate(factor_neighbors)])
    influence=class_density/global_density
    influence[factor>=threshold]=0
    return {'influence':influence,'local_density_factor':factor,
            'global_density':global_density,'class_density':class_density,
            'class_neighbors':class_neighbors,'global_neighbors':global_neighbors}


def fuzzy_entropy(relation):
    size=np.asarray(relation).sum(axis=1)
    if np.any(size<=0):raise ValueError('zero fuzzy granule: entropy undefined')
    return float(-np.log2(size/len(size)).mean())


def fdm_rank_relations(relations,decision):
    """Eqs21-25 relevance+indispensability, min-conjunction relations."""
    n=len(decision);d=len(relations);cache={}
    def mutual(fs):
        key=tuple(sorted(fs))
        if key not in cache:
            R=np.ones((n,n)) if not key else np.minimum.reduce([relations[i] for i in key])
            cache[key]=fuzzy_entropy(R)+fuzzy_entropy(decision)-fuzzy_entropy(np.minimum(R,decision))
        return cache[key]
    selected=[];remaining=set(range(d));trace=[]
    while remaining:
        current=mutual(selected);rest=mutual(remaining)
        values=[]
        for candidate in sorted(remaining):
            relevance=mutual(selected+[candidate])-current
            indispensable=rest-mutual(remaining-{candidate})
            values.append({'feature':candidate,'relevance':relevance,
                           'indispensability':indispensable,'score':relevance+indispensable})
        best=max(values,key=lambda row:(row['score'],-row['feature']))
        selected.append(best['feature']);remaining.remove(best['feature'])
        trace.append({'chosen':best['feature'],'candidates':values})
    return {'ranking':selected,'trace':trace}


def rank_ldfrs(X,y,*,k=10,threshold=1.5,delta=.2,convention='legacy'):
    X,y=np.asarray(X,float),np.asarray(y)
    if not np.isfinite(delta) or delta<=0:raise ValueError('delta must be positive')
    if convention not in ('legacy','inclusive_formula','toy_reconciled'):
        raise ValueError('unknown LDFRS convention')
    density=local_influence(X,y,k=k,threshold=threshold,
        density_self=convention!='legacy',
        factor_self=convention!='toy_reconciled' if convention!='legacy' else False)
    influence=density['influence'];same=y[:,None]==y[None,:]
    boosted=same&(influence[None,:]>0)&(influence[None,:]<1)
    relations=[]
    for a in range(X.shape[1]):
        denominator=delta if convention=='legacy' else 2*delta**2
        gaussian=np.exp(-(X[:,a,None]-X[None,:,a])**2/denominator)
        relation=np.where(boosted,np.minimum(1,(1+influence[None,:])*gaussian),
                          influence[None,:]*gaussian)
        if convention=='toy_reconciled':np.fill_diagonal(relation,1)
        relations.append(relation)
    decision=same.astype(float)
    result=fdm_rank_relations(relations,decision)
    return {**result,'density':{key:value.tolist() for key,value in density.items()},
            'conventions':{'k':k,'threshold':threshold,'delta':delta,
                'kernel':'exp(-difference_squared/delta)' if convention=='legacy' else 'exp(-difference_squared/(2*delta**2))',
                'neighbor_self':'excluded' if convention=='legacy' else 'included_for_density',
                'factor_self':convention=='inclusive_formula',
                'diagonal_repair':convention=='toy_reconciled','convention':convention,
                'influence':'full_table_frozen','decision':'crisp_equality'},
            'status':'independent_LDFRS_FDM_variant_not_author_validated'}
