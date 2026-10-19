import numpy as np
from scipy.spatial.distance import cdist
from certified_frfs.recent import sfrip_dependency_from_distances, select_sfrip


def test_dependency_matches_literal_closed_neighborhoods():
    X = np.array([[0.], [.05], [.3], [.35], [.7], [1.]])
    y = np.array([0, 0, 1, 1, 0, 1])
    distance = cdist(X, X)
    total = 0
    for i in range(len(y)):
        radii = sorted(distance[i, y != y[i]])[:3]
        counts = [sum(distance[i] <= radius) for radius in radii]
        t = max(range(len(radii)), key=lambda t: counts[t] - 2 * t)
        neighborhood = distance[i] <= radii[t]
        size = sum(neighborhood)
        normal = sum(neighborhood & (y == y[i]))
        if size >= 2 and normal / size >= .5:
            total += normal
    actual = sfrip_dependency_from_distances(distance, y, beta=2, k=3,
                                             support=2, confidence=.5)
    assert actual == total / len(y) ** 2


def test_radial_similarity_neighborhood_equivalence_with_ties():
    X = np.array([[0.], [0.], [1.], [1.]])
    distance = cdist(X, X)
    similarity = np.exp(-distance ** 2)
    for i in range(len(X)):
        for j in range(len(X)):
            assert np.array_equal(distance[i] <= distance[i, j],
                                  similarity[i] >= similarity[i, j])


def test_native_stopping_and_candidate_specific_core_sets():
    X = np.array([[0., 0.], [.01, 0.], [1., 0.], [.99, 0.]])
    y = np.array([0, 0, 1, 1])
    result = select_sfrip(X, y, beta=2, k=1, support=1, confidence=.5)
    assert result['features'] == [0]
    assert not result['history'][-1]['accepted']
