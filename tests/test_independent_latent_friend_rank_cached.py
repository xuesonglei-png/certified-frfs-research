import numpy as np
from latent_friend_rank_hinge import LatentFriendRankHinge
from independent_latent_friend_rank import ScalarLatentValidator as OriginalValidator
from independent_latent_friend_rank_cached import ScalarLatentValidator as CachedValidator


def test_scalar_cache_preserves_every_triple_point_and_tangent_face():
    rng = np.random.default_rng(904)
    A = rng.uniform(0, 1, (24, 4)); y = np.repeat([0, 1], 12); anchor = np.ones(4)/4
    oracle = LatentFriendRankHinge(A, y, anchor)
    original = OriginalValidator(A, oracle.export()); cached = CachedValidator(A, oracle.export())
    for point in (anchor, np.array([.4, .3, .2, .1]), anchor):
        for i, row in enumerate(oracle.rows):
            if row['constant_loss'] is None:
                for choice in range(len(oracle.triples[i])):
                    assert original.hinges(i, point, choice) == cached.hinges(i, point, choice)
        assert np.array_equal(original.losses(point), cached.losses(point))
        choices = oracle.choose_triples(point)
        cached.verify_choices(point, choices)
        for k in (2, 3):
            assert np.array_equal(original.top_sum(point, k, choices), cached.top_sum(point, k, choices))
            _, faces = oracle.top_sum(point, k, choices)
            for face in faces:
                cached.validate(point, face)
    assert len(cached.last_contrasts) <= len(A)
