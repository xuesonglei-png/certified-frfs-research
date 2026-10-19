"""Independent scalar geometry cache; same valid-face and tangency checks."""
from itertools import combinations
import numpy as np
from independent_majority_hinge import coefficient
from audit_forest_compact_transfer import equivalent


class CachedValidator:
    def __init__(self,A,info):
        self.A=np.array(A,float,copy=True);self.A.setflags(write=False);self.info=info;self.cache={}

    def query_coefficients(self,i):
        if i not in self.cache:
            row=self.info['witnesses'][i]
            self.cache[i]=np.array([[coefficient(self.A,i,e,f,self.info['eps']) for f in row['friends']] for e in row['enemies']])
        return self.cache[i]

    def validate(self,point,cut):
        info=self.info;i=cut['query'];row=info['witnesses'][i];ids=cut['enemy_indices'];friends=cut['worst_friend_indices']
        assert row['constant_loss'] is None and len(ids)==len(set(ids))==len(friends)==3
        assert all(int(j)==j and 0<=j<len(row['enemies']) for j in ids)
        assert all(int(j)==j and 0<=j<3 for j in friends)
        active=int(round(3*cut['intercept']));assert 0<=active<=3;equivalent(active/3,cut['intercept'])
        all_C=self.query_coefficients(i);selected_C=[all_C[e,f] for e,f in zip(ids,friends)];valid=[]
        for subset in combinations(range(3),active):
            gradient=-sum((selected_C[j] for j in subset),start=np.zeros(len(point)))/(3*info['margin'])
            if np.allclose(gradient,cut['gradient'],rtol=1e-9,atol=1e-10):valid.append(subset)
        assert valid,'Stored cut is not a valid selected-enemy hinge lower face'
        # Keep direct individual scalar dot products and explicit sort keys;
        # only coordinate geometry is cached, not faces, losses or gradients.
        contrasts=[[float(c@point) for c in enemy] for enemy in all_C]
        worst=[min(range(3),key=lambda j:(v[j],j)) for v in contrasts]
        hinge=[max(0.,1-v[f]/info['margin']) for v,f in zip(contrasts,worst)]
        selected=sorted(range(len(hinge)),key=lambda j:(-hinge[j],row['enemies'][j]))[:3]
        value=sum(hinge[j] for j in selected)/3
        equivalent(float(np.array(cut['gradient'])@point+cut['intercept']),value)
        canonical=np.zeros(len(point))
        for j in selected:
            if hinge[j]>0:canonical-=all_C[j,worst[j]]/(3*info['margin'])
        nonunique=not np.allclose(canonical,cut['gradient'],rtol=1e-9,atol=1e-10)
        return {'valid_lower_face':True,'independent_tangency_passed':True,'canonical_gradient_differs':bool(nonunique),
                'valid_active_subsets':[list(s) for s in valid]}
