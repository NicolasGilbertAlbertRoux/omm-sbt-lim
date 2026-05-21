import numpy as np

from lim.metrics import dominance, recovered, recovery_rate


def test_dominance_and_recovery():
    competitors = np.array([0.1, 0.2, 0.3])
    assert dominance(0.5, competitors) == 0.2
    assert recovered(0.5, competitors) == 1
    assert recovered(0.2, competitors) == 0


def test_recovery_rate():
    assert recovery_rate(np.array([1, 0, 1, 1])) == 0.75
