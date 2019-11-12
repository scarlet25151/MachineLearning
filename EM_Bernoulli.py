import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

class EM_bernoulli():
    def __init__(self, K):
        self.K = K

    def fit(self):
        K = self.K
        data = loadmat('problem2.mat').get('dataset')
        permutation = np.random.permutation(data.shape[0])
        train_data = data[permutation[:500]]
        test_data = data[permutation[500:]]

        [N, T] = train_data.shape
        tau = np.zeros((N, K))
        alpha = np.zeros((K, 1))
        mix = np.ones(K) / K

        train_X = np.sum(train_data, axis=1)
        test_X = np.sum(test_data, axis=1)
        for k in range(K):
            alpha[k] = 0.5 * k / K
        train_likelihood_set = []
        test_likelihood_set = []

        for step in range(10):
            # Expectation
            for n in range(N):
                S = 0
                for k in range(K):
                    S += mix[k] * np.power(alpha[k], train_X[n]) * np.power(1 - alpha[k], T - train_X[n])
                for k in range(K):
                    tau[n, k] = mix[k] * np.power(alpha[k], train_X[n]) * np.power(1 - alpha[k], T - train_X[n]) / S

            # Maximization
            for k in range(K):
                if (np.abs(np.sum(tau[:, k])) < 1e-9):
                    alpha[k] = 0
                else:
                    alpha[k] = np.sum(tau[:, k] * train_X) / T / np.sum(tau[:, k])
            mix = np.sum(tau, axis=0) / N
            train_likelihood = 0
            for n in range(N):
                p = 0
                for k in range(K):
                    p += mix[k] * np.power(alpha[k], train_X[n]) * np.power(1 - alpha[k], T - train_X[n])
                train_likelihood += np.log(p)
            train_likelihood_set.extend(train_likelihood)
            test_likelihood = 0
            for n in range(N):
                p = 0
                for k in range(K):
                    p += mix[k] * np.power(alpha[k], test_X[n]) * np.power(1 - alpha[k], T - test_X[n])
                test_likelihood += np.log(p)
            test_likelihood_set.extend(test_likelihood)
        return train_likelihood_set, test_likelihood_set, alpha

def plot_result():
    plt.figure(1)
    for K in range(2, 8):
        train_set, test_set, _ = EM_bernolli(K).fit()
        plt.xlabel("Iterations")
        plt.ylabel("likelihood")
        plt.plot(train_set, label='K=%d' % (K - 1))
    plt.grid()
    plt.title('Likelihood by Iteration on Training Set')
    plt.legend()
    plt.show()
    plt.savefig('Bpart_train.jpeg')
    plt.close()

    plt.figure(2)
    for K in range(2, 8):
        train_set, test_set, _ = EM_bernolli(K).fit()
        plt.xlabel("Iterations")
        plt.ylabel("likelihood")
        plt.plot(test_set, label='K=%d' % (K - 1))
    plt.grid()
    plt.title('Likelihood by Iteration on Testing Set')
    plt.legend()
    plt.show()
    plt.savefig('Bpart_test.jpeg')
    plt.close()
    for K in range(2, 8):
        _, _, alpha = EM_bernolli(K).fit()
        print(alpha)


if __name__ == '__main__':
     plot_result()