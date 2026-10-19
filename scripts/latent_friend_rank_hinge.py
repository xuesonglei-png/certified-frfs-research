"""Finite-pool witness adaptation for a bounded robust crossgroup loss.

Every friend triple supplies a valid pointwise error bound; their minimum
does too. The triple selected at a point gives a tangent majorizer, not a
global optimum or population-risk guarantee. Pools are fixed at the anchor.
"""
from itertools import combinations
import numpy as np
from scipy.spatial.distance import cdist
from third_enemy_hinge import ThirdEnemyHinge


class LatentFriendRankHinge(ThirdEnemyHinge):
    def __init__(self, A, y, anchor, *, margin=.02, eps=.01, pool_size=5):
        if pool_size != 5:
            raise ValueError('fixed declared five-friend pool required')
        super().__init__(A, y, anchor, margin=margin, eps=eps)
        self.fixed_three = [r['friends'].copy() for r in self.rows]
        distance = cdist(self.A*self.anchor, self.A*self.anchor, 'cityblock')
        self.triples = []
        for row in self.rows:
            i = row['query']; available = np.asarray([j for j in row['references'] if self.y[j] == self.y[i]], int)
            pool = available[np.lexsort((available, distance[i, available]))][:pool_size]
            row['friends'] = pool.tolist()
            self.triples.append(list(combinations(range(len(pool)), 3)))
        assert all(r['friends'][:3] == f for r,f in zip(self.rows, self.fixed_three))

    def triple_hinges(self, i, w, triple):
        C = self.coefficients(i)[:, triple, :]; contrast = C@w
        worst = contrast.argmin(axis=1)
        return np.maximum(0., 1-contrast[np.arange(len(C)), worst]/self.margin), worst

    def choose_triples(self, w):
        w = np.asarray(w, float); choices = []
        for i, row in enumerate(self.rows):
            if row['constant_loss'] is not None:
                choices.append(None); continue
            losses = []
            for triple in self.triples[i]:
                hinge, _ = self.triple_hinges(i, w, triple)
                losses.append(float(min(1., np.partition(hinge, -3)[-3])))
            choices.append(min(range(len(losses)), key=lambda j: (losses[j], j)))
        return choices

    def top_sum(self, w, k, choices=None):
        if k not in (2, 3):
            raise ValueError('only top2/top3 are declared')
        w = np.asarray(w, float); choices = self.choose_triples(w) if choices is None else choices
        if len(choices) != len(self.rows):
            raise ValueError('one triple choice per query required')
        values, faces = [], []
        for i, (row, choice) in enumerate(zip(self.rows, choices)):
            gradient = np.zeros(len(w)); ids, friends, active = [], [], []; triple = None
            if row['constant_loss'] is not None:
                assert choice is None
                value = row['constant_loss'] if k == 3 else 0.; intercept = value
            else:
                if choice is None or not 0 <= choice < len(self.triples[i]):
                    raise ValueError('valid fixed-pool triple required')
                triple = self.triples[i][choice]; C = self.coefficients(i)
                hinge, local_worst = self.triple_hinges(i, w, triple)
                ids = np.lexsort((np.asarray(row['enemies']), -hinge))[:k].tolist()
                friends = [int(triple[local_worst[j]]) for j in ids]
                active = [j for j in ids if hinge[j] > 0]
                selected_friend = {j:f for j,f in zip(ids,friends)}
                gradient = -sum((C[j,selected_friend[j]] for j in active), start=np.zeros(len(w)))/self.margin
                intercept = float(len(active)); value = float(hinge[ids].sum())
            values.append(value)
            faces.append({'query':i,'k':k,'triple_choice':choice,'friend_pool_slots':None if triple is None else list(triple),
                'gradient':gradient.tolist(),'intercept':intercept,'enemy_indices':ids,
                'worst_friend_indices':friends,'active_enemy_indices':active})
        return np.asarray(values), faces

    def losses(self, w):
        choices = self.choose_triples(w); three,_ = self.top_sum(w,3,choices); two,_ = self.top_sum(w,2,choices)
        return np.clip(three-two,0,1)

    def majorizer_at(self, w):
        choices = self.choose_triples(w); two,f2 = self.top_sum(w,2,choices); three,f3 = self.top_sum(w,3,choices)
        shifts = (three-1. > two).astype(int).tolist()
        faces = [f3[i] if shift else f2[i] for i,shift in enumerate(shifts)]
        return {'triple_choices':choices,'minor_faces':faces,'minor_cap_shifts':shifts,
            'tangent_H':np.maximum(two,three-1.).tolist(),'loss_at_point':np.clip(three-two,0,1).tolist()}

    def export(self):
        return {**super().export(),'pool_size':5,'original_three_friends':self.fixed_three,
            'triple_pool_slots':[[list(t) for t in row] for row in self.triples],
            'decomposition':'per_query_min_over_fixed_triples_of_S3_minus_max_S2_S3minus1',
            'scope':'fixed_five_friend_latent_majorizer_bounded_training_loss_not_population_risk_or_verified_novelty'}
