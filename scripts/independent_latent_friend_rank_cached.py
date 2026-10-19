"""Scalar validator with exact dot-product reuse across friend triples.

Every coefficient is still reconstructed with the independent scalar box
formula. This changes no geometry, comparison tolerance, witness rule or
LP check. One last weight vector per query bounds cache memory.
"""
import numpy as np
from independent_latent_friend_rank import ScalarLatentValidator as UncachedValidator


class ScalarLatentValidator(UncachedValidator):
    def __init__(self, A, info):
        super().__init__(A, info)
        self.last_contrasts = {}

    def hinges(self, i, w, choice):
        point = tuple(np.asarray(w, float))
        last = self.last_contrasts.get(i)
        if last is None or last[0] != point:
            # The same scalar dot as the uncached validator, computed once
            # per enemy/friend and reused for all up to ten triples.
            contrasts = [[float(c@w) for c in line] for line in self.coefficients(i)]
            self.last_contrasts[i] = (point, contrasts)
        else:
            contrasts = last[1]
        triple = self.info['triple_pool_slots'][i][choice]
        return [max(0., 1-min(line[f] for f in triple)/self.info['margin']) for line in contrasts]
