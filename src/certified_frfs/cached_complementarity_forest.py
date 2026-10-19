"""Reuse train-only edge information across support-specific tree relearning."""
import numpy as np
from sklearn.metrics import mutual_info_score
from .triangular_classifier import triangular_memberships


class ComplementarityForestCache:
    def __init__(self,X,y):
        X,y = np.asarray(X,float),np.asarray(y); bins = triangular_memberships(X).argmax(axis=-1)
        self.d = X.shape[1]; self.single = [float(mutual_info_score(bins[:,a],y)) for a in range(self.d)]
        pairs = []
        for u in range(self.d):
            for v in range(u+1,self.d):
                joint = float(mutual_info_score(3*bins[:,u]+bins[:,v],y))
                pairs.append((max(0.,joint-max(self.single[u],self.single[v])),u,v))
        self.ordered = sorted(pairs,key=lambda r:(-r[0],r[1],r[2])); self.cache = {}

    def edges(self,features):
        fs = tuple(sorted(features))
        if len(set(fs)) != len(fs) or any(int(a) != a or not 0 <= a < self.d for a in fs): raise ValueError('invalid support')
        if fs not in self.cache:
            keep = set(fs); parents = {a:a for a in fs}; chosen = []
            def root(a):
                while parents[a] != a: a = parents[a]
                return a
            for _,u,v in self.ordered:
                if u not in keep or v not in keep: continue
                a,b = root(u),root(v)
                if a != b: parents[a] = b; chosen.append([u,v])
                if len(chosen) == len(fs)-1: break
            self.cache[fs] = chosen
        return self.cache[fs]
