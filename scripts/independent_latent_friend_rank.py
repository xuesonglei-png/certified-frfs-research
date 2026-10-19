"""Independent scalar pool/triple/order-statistic witness loss validator."""
from itertools import combinations
import numpy as np
from independent_majority_hinge import coefficient
from audit_forest_compact_transfer import equivalent


class ScalarLatentValidator:
    def __init__(self,A,info):
        self.A,self.info,self.cache=np.asarray(A,float),info,{}

    def coefficients(self,i):
        if i not in self.cache:
            row=self.info['witnesses'][i]
            self.cache[i]=np.asarray([[coefficient(self.A,i,e,f,self.info['eps']) for f in row['friends']] for e in row['enemies']])
        return self.cache[i]

    def hinges(self,i,w,choice):
        row=self.info['witnesses'][i];triple=self.info['triple_pool_slots'][i][choice]
        return [max(0.,1-min(float(c@w) for c in line[triple])/self.info['margin']) for line in self.coefficients(i)]

    def top_sum(self,w,k,choices):
        out=[]
        for i,(row,choice) in enumerate(zip(self.info['witnesses'],choices)):
            if row['constant_loss'] is not None:
                assert choice is None
                out.append(row['constant_loss'] if k==3 else 0.)
            else:
                out.append(sum(sorted(self.hinges(i,w,choice),reverse=True)[:k]))
        return np.asarray(out)

    def losses(self,w):
        values=[]
        for i,row in enumerate(self.info['witnesses']):
            if row['constant_loss'] is not None:
                values.append(row['constant_loss']);continue
            values.append(min(min(1.,sorted(self.hinges(i,w,choice),reverse=True)[2]) for choice in range(len(self.info['triple_pool_slots'][i]))))
        return np.asarray(values)

    def verify_choices(self,w,choices):
        for i,(row,choice) in enumerate(zip(self.info['witnesses'],choices)):
            assert self.info['triple_pool_slots'][i]==[list(t) for t in combinations(range(len(row['friends'])),3)]
            if row['constant_loss'] is not None:
                assert choice is None;continue
            assert int(choice)==choice and 0<=choice<len(self.info['triple_pool_slots'][i])
            value=min(1.,sorted(self.hinges(i,w,choice),reverse=True)[2])
            minimum=min(min(1.,sorted(self.hinges(i,w,q),reverse=True)[2]) for q in range(len(self.info['triple_pool_slots'][i])))
            equivalent(value,minimum)

    def validate(self,point,face):
        i,k,choice=face['query'],face['k'],face['triple_choice'];row=self.info['witnesses'][i]
        assert k in (2,3)
        if row['constant_loss'] is not None:
            assert choice is None and face['friend_pool_slots'] is None and not face['enemy_indices']
            assert not face['worst_friend_indices'] and not face['active_enemy_indices']
            equivalent(face['gradient'],np.zeros(len(point)).tolist());equivalent(face['intercept'],row['constant_loss'] if k==3 else 0.);return
        triple=self.info['triple_pool_slots'][i][choice];assert face['friend_pool_slots']==triple
        ids,friends,active=face['enemy_indices'],face['worst_friend_indices'],face['active_enemy_indices']
        assert len(ids)==len(set(ids))==len(friends)==k and set(active)<=set(ids) and len(active)==len(set(active))
        assert all(f in triple for f in friends) and all(0<=j<len(row['enemies']) for j in ids)
        C=self.coefficients(i);gradient=np.zeros(len(point))
        for j,f in zip(ids,friends):
            if j in active:
                gradient-=C[j,f]/self.info['margin']
        equivalent(face['gradient'],gradient.tolist());equivalent(face['intercept'],float(len(active)))
        equivalent(float(gradient@point+face['intercept']),sum(sorted(self.hinges(i,point,choice),reverse=True)[:k]))
