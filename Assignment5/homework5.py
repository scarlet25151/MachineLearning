import numpy as np
import copy

def JCT4Markovechain(potentials):
    marginals = potentials
    n = len(potentials)
    separators = np.ones((n - 1, 2))
    # forward calculation
    for i in range(n - 1):
        separators[i, :] = np.sum(marginals[i], axis=0)
        marginals[i + 1] = (marginals[i + 1].transpose() * separators[i, :]).transpose()

    # backward calculation
    for i in range(n - 2, -1, -1):
        old_separator = copy.deepcopy(separators[i, :])
        separators[i, :] = np.sum(marginals[i + 1], axis=1)
        marginals[i] = marginals[i] * (separators[i, :] / old_separator)

    # Normalization
    for i in range(n):
        marginals[i] = marginals[i] / np.sum(marginals[i])
    return marginals

def argmaxJTA():
    pass

if __name__ == "__main__":
    n = 5
    potentials = []
    for i in range(n - 1):
        potentials.append(np.random.rand(2, 2))
    print(potentials)
    marginals = JCT4Markovechain(potentials)
    print(marginals)

    p_test = []
    p_test.append(np.array([[0.1, 0.7], [0.8, 0.3]]))
    p_test.append(np.array([[0.5, 0.1], [0.1, 0.5]]))
    p_test.append(np.array([[0.1, 0.5], [0.5, 0.1]]))
    p_test.append(np.array([[0.9, 0.3], [0.1, 0.3]]))
    print(p_test)
    marginals = JCT4Markovechain(p_test)
    print(marginals)

    test = np.random.rand(2, 2)
    test_2 = np.sum(test, 0)
