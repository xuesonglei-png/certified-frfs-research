"""Independent numerical FGS-RFRAS implementation (Wan et al., TFS 2023).

The paper leaves empty s1 undefined. Here s1=max(1,floor(sum(p)*(1-beta)))
means no omission when that set is empty. This documented convention is not
claimed to match unavailable author code. Entropy intersections use minimum.
"""
import numpy as np


def entropy(relation):
    return float(-np.log2(np.asarray(relation).sum(axis=1)/len(relation)).mean())


def dependency(relation, y, beta):
    """Eqs. 2,4: omit s1-1 smallest complements, then max over classes."""
    lower = []
    for label in np.unique(y):
        p = 1-relation[:, y != label]
        if p.shape[1] == 0:
            lower.append(np.ones(len(y)))
            continue
        s1 = np.maximum(1, np.floor(p.sum(axis=1)*(1-beta)).astype(int))
        s1 = np.minimum(s1, p.shape[1])
        ordered = np.sort(p, axis=1)
        lower.append(ordered[np.arange(len(y)), s1-1])
    return float(np.max(lower, axis=0).mean())


def graph_select(cd, edges):
    """Deterministic Kruskal forest, IR-WCR-MDS (Algorithm 1)."""
    d = len(cd)
    parent = list(range(d))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    tree = []
    for weight, i, j in sorted(edges):
        if weight == 0:
            continue
        a, b = root(i), root(j)
        if a != b:
            parent[b] = a
            tree.append((float(weight), i, j))
    parent = list(range(d))
    retained = set()
    for weight, i, j in tree:
        if weight < 0 and abs(weight) >= max(cd[i], cd[j]):
            retained.update([i, j])
        if abs(weight) >= min(cd[i], cd[j]):
            parent[root(j)] = root(i)
    components = {}
    for i in range(d):
        components.setdefault(root(i), []).append(i)
    groups = sorted(components.values(), key=lambda g:g[0])
    for group in groups:
        retained.add(max(group, key=lambda i:(cd[i], -i)))
    return {'features': sorted(retained), 'groups': groups,
            'tree': tree, 'interactive_features': sorted(set(
                i for w,a,b in tree if w<0 and abs(w)>=max(cd[a],cd[b]) for i in [a,b]))}


def select_fgs(X, y, *, beta=.8, xi=1.):
    X, y = np.asarray(X, float), np.asarray(y)
    if (X.ndim!=2 or not X.shape[1] or len(y)!=len(X) or len(X)<2
            or not np.isfinite(X).all() or X.min()<0 or X.max()>1
            or not 0<=beta<=1 or xi<=0 or not np.isfinite(xi)):
        raise ValueError('FGS expects scaled finite numeric training data and valid parameters')
    relations = []
    for a in range(X.shape[1]):
        difference = abs(X[:,a,None]-X[None,:,a])
        radius = X[:,a].std(ddof=1)/xi
        relations.append(np.where(difference<=radius, 1-difference, 0.))
    D = (y[:,None] == y[None,:]).astype(float)
    hd = entropy(D)
    h = [entropy(r) for r in relations]
    joint_d = [entropy(np.minimum(r,D)) for r in relations]
    cd = [dependency(r,y,beta) for r in relations]
    edges=[]
    for i in range(len(relations)):
        for j in range(i+1,len(relations)):
            pair = np.minimum(relations[i],relations[j])
            # H(i)+H(j)+H(D)-H(i,j)-H(i,D)-H(j,D)+H(i,j,D).
            weight=h[i]+h[j]+hd-entropy(pair)-joint_d[i]-joint_d[j]+entropy(np.minimum(pair,D))
            edges.append((float(weight),i,j))
    result=graph_select(cd,edges)
    return {**result,'cd':cd,'beta':beta,'xi':xi,
            'implementation':'independent_FGS_RFRAS_numeric_documented_boundary_convention'}


def refine_group_candidates(X,y,base,*,eps=.01,penalty=.02,k=5,dependency_eta=None,weights=None):
    """Training-only robust majority evaluation with one graph-group edit.

    Include the original graph subset, full table, and adding/deleting each
    graph component. This tests grouped search beyond individual rank prefixes.
    """
    from .knn_decision import certified_knn_coverage
    original=set(base['features'])
    candidates={tuple(sorted(original)),tuple(range(X.shape[1]))}
    for group in base['groups']:
        for candidate in [original-set(group),original|set(group)]:
            if candidate:
                candidates.add(tuple(sorted(candidate)))
    trace=[]
    for fs in sorted(candidates):
        loss_certificate=None
        if dependency_eta is not None:
            from .core import certificate
            if not 0<=dependency_eta<=1:
                raise ValueError('invalid dependency tolerance')
            loss_certificate=certificate(X,y,fs,weights=weights,eps=.01,method='hybrid')
            if loss_certificate['upper_bound']>dependency_eta+1e-10:
                trace.append({'features':list(fs),'objective':None,'feasible':False,
                              'loss_certificate':loss_certificate})
                continue
        coverage=certified_knn_coverage(X,y,fs,k=k,eps=eps,weights=weights)
        objective=(coverage['balanced_coverage']+coverage['min_class_coverage'])/2-penalty*len(fs)/X.shape[1]
        trace.append({'features':list(fs),'objective':float(objective),'coverage':coverage,
                      'feasible':True,'loss_certificate':loss_certificate})
    best=max((r for r in trace if r['feasible']),key=lambda r:(round(r['objective'],12),-len(r['features'])))
    return {'features':best['features'],'trace':trace,'base':base,
            'loss_certificate':best['loss_certificate'],'dependency_eta':dependency_eta}


def certified_backward_candidates(X, y, ranking, *, eps=.01, eta=.1,
                                  weights=None,stability='interval'):
    """Return every feasible prefix encountered by certified backward deletion.

    The full subset is always included. This is a candidate generator, not a
    minimality claim; each returned subset has an independently recomputed bound.
    """
    from .core import certificate
    X=np.asarray(X,float); ranking=list(ranking)
    d=X.shape[1]
    if sorted(ranking)!=list(range(d)): raise ValueError('ranking must be a permutation')
    current=set(range(d)); candidates={tuple(sorted(current))}
    trace=[]
    for feature in ranking:
        candidate=current-{feature}
        if not candidate: continue
        cert=certificate(X,y,sorted(candidate),weights=weights,eps=eps,method='hybrid',stability=stability)
        feasible=cert['upper_bound']<=eta+1e-10
        trace.append({'removed':int(feature),'features':sorted(candidate),
                      'feasible':feasible,'loss_certificate':cert})
        if feasible:
            current=candidate; candidates.add(tuple(sorted(current)))
    return {'candidates':[list(v) for v in sorted(candidates,key=lambda v:(len(v),v))],
            'trace':trace}


def combined_certified_candidates(X,y,base,ranking,*,eps=.01,eta=.1,
                                  weights=None):
    """Union graph edits and certified backward paths, deduplicated."""
    backward=certified_backward_candidates(X,y,ranking,eps=eps,eta=eta,weights=weights)
    graph=refine_group_candidates(X,y,base,eps=eps,dependency_eta=eta,weights=weights)
    pool={tuple(fs) for fs in backward['candidates']}
    pool.update(tuple(row['features']) for row in graph['trace'] if row.get('feasible'))
    return {'candidates':[list(v) for v in sorted(pool,key=lambda v:(len(v),v))],
            'backward':backward,'graph':graph}
