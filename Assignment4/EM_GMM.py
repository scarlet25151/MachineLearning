import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

np.seterr(divide='ignore', invalid='ignore')

class mix_model():

    def __init__(self, inputs, M, max_iter):
        self.inputs = inputs
        self.M = M
        self.max_iter = max_iter

    def rand_init(self):
        dim1, dim2 = 1, 2
        vscale = 2
        grain = 40
        N, D = self.inputs.shape

        # Randomly initialize M Gaussians

        dataMax = np.max(self.inputs, axis=0).reshape(1, -1)
        dataMin = np.min(self.inputs, axis=0).reshape(1, -1)
        initmu = (dataMax + dataMin) / 2
        initsd = (dataMax - dataMin) / np.power(self.M, 1/D)
        mu = []
        for i in range(self.M):
            mux = np.random.random(dataMax.shape)
            mux = mux * (dataMax - dataMin) + dataMin
            mu.extend(mux)
        mu = np.asarray(mu)

        initcv = np.diag(np.power(initsd[0] / vscale, 2))

        R = np.zeros((self.M * D, D))
        initsd_mean = np.mean(initsd)
        for i in range(self.M):
            q = np.random.random(initcv.shape) - 0.5 * np.ones_like(initcv)
            q = q * q.transpose() * 2 * initsd_mean
            scal = (initsd_mean / self.M) ** 2
            q = scal * np.eye(initcv.shape[0])
            R[i * D : (i + 1) * D, :] = np.linalg.inv(q)

        covar = np.zeros((self.M * D, D))
        for i in range(self.M):
            covar[i * D : (i + 1) * D, :] = np.linalg.inv(R[i * D : (i + 1) * D, :])

        mix = np.ones(self.M) / self.M
        plt.figure(1)
        plt.axis([dataMin[0][0], dataMax[0][0], dataMin[0][1], dataMax[0][1]])

        plt.show()
        return mu, covar, mix

    def gen_data(self, mu, covar, numpts):
        M = mu.shape[0]
        D = mu.shpae[1]
        if (numpts.shape[1] == 1):
            numpts = np.ceil(numpts[0] / M) * np.ones((1, M))

        if (covar.shape[0] < M * D):
            tmp_covar = covar
            covar = np.zeros((M * D), D)
            for i in range(M):
                covar[(i * D + 1) : (i + 1) * D, :] = tmp_covar

        A = np.zeros((M * D), D)
        for i in range(M):
            cvi = covar[(i * D + 1) : (i + 1) * D, :]
            vv, dd = np.linalg.eig(cvi)
            A[(i * D + 1) : (i + 1) * D, :] = (vv * np.sqrt(dd)).transpose()

        data = np.zeros((np.sum(numpts), D))

        p = 0
        for i in range(M):
            Ai = A[(i * D + 1) : (i + 1) * D, :]
            for n in range(numpts[i]):
                z = np.random.randn(1, D)
                data[p, :] = z * Ai + mu[i, :]
                p += 1

        return data

    def fit(self):
        pi = np.pi
        N, D = self.inputs.shape
        M = self.M

        likelihood = []
        threshold = 1e-6
        converged = 0
        iter = 0
        ll = -1e9
        mu, covar, mix = self.rand_init()
        tau = np.zeros((N, M))
        while (iter < self.max_iter):
            prev = ll
            ll = 0
            for i in range(M):
                print(covar[i * D: (i + 1) * D, :])
                det_cov = np.abs(np.linalg.det(covar[i * D: (i + 1) * D, :]))
                inv_cov = np.linalg.inv(covar[i * D: (i + 1) * D, :] + 1e-6 * np.eye(D))

                for n in range(N):
                    tau[n, i] = mix[i] * np.power(2 * pi, -D / 2) * np.power(det_cov, -1 / 2) \
                                * np.exp(-1 / 2 * np.matmul(np.matmul(
                        self.inputs[n] - mu[i], inv_cov), self.inputs[n] - mu[i]).astype('float128'))
            # for n in range(N):
            #     S = 0
            #     for i in range(M):
            #         det_cov = np.abs(np.linalg.det(covar[i * D: (i + 1) * D, :]))
            #         inv_cov = np.linalg.inv(covar[i * D: (i + 1) * D, :] + 1e-6 * np.eye(D))
            #         S += mix[i] * np.power(2 * pi, -D / 2) * np.power(det_cov, -1 / 2) \
            #                     * np.exp(-1 / 2 * np.matmul(np.matmul(
            #             self.inputs[n] - mu[i], inv_cov), self.inputs[n] - mu[i]))
            #     for i in range(M):
            #         det_cov = np.abs(np.linalg.det(covar[i * D: (i + 1) * D, :]))
            #         inv_cov = np.linalg.inv(covar[i * D: (i + 1) * D, :] + 1e-6 * np.eye(D))
            #         if (np.abs(S) < 1e-9):
            #             tau[n, i] = 0
            #         else :
            #             tau[n, i] = mix[i] * np.power(2 * pi, -D / 2) * np.power(det_cov, -1 / 2) \
            #                 * np.exp(-1 / 2 * np.matmul(np.matmul(
            #                 self.inputs[n] - mu[i], inv_cov), self.inputs[n] - mu[i])) / S

            for n in range(N):
                l = np.sum(tau[n])
                tau[n] = tau[n] / l
                ll += np.log(l)

            if (np.abs(ll - prev) < threshold):
                converged = 1
            likelihood.append(ll)

            for i in range(M):
                sum_tau = np.sum(tau[:, i])
                mu[i] = 0
                for n in range(N):
                    mu[i] += tau[n, i] * self.inputs[n]
                if (np.abs(sum_tau) < 1e-9):
                    mu[i] = 0
                else:
                    mu[i] /= sum_tau
                covar[i * D: (i + 1) * D, :] = 0
                for n in range(N):
                    covar[i * D: (i + 1) * D, :] += tau[n, i] * \
                                                    (self.inputs[n] - mu[i]).reshape(-1, 1) \
                                                    * self.inputs[n] - mu[i]
                covar[i * D: (i + 1) * D, :] = covar[i * D: (i + 1) * D, :] / sum_tau + 1e-5 * np.eye(D)
                mix[i] = sum_tau / N
            print(mu)
            print(covar)
            iter += 1
        return likelihood, mu, covar, mix, converged

    def plot_gauss(self, mu1, mu2, var1, var2, covar):
        pi = np.pi
        t = np.linspace(-pi, pi, 100)
        k = t.shape[0]
        x = np.sin(t)
        y = np.cos(t)
        R = [[var1,covar],[covar, var2]]
        vv, dd = np.linalg.eig(R)


if __name__ == "__main__":
    data = loadmat('problem2.mat').get('dataset')
    # load data from the txt files
    datasetA = np.loadtxt('datasetA.txt')
    datasetB = np.loadtxt('datasetB.txt')
    M = 3
    max_iter = 100
    mixmodel = mix_model(datasetA, M, max_iter)
    like, mu, covar, mix, converged = mixmodel.fit()

