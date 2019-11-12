import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

def kmeans(K):
    image = loadmat('problem3.mat').get('im')

    plt.figure(1)
    plt.imshow(image)
    plt.show()
    plt.title('Original Picture')
    plt.close()
    height, width, _ = image.shape
    # initialization
    K = 5
    mu = np.zeros((K, 3))

    for k in range(K):
        mu_h = np.random.randint(1, height)
        mu_w = np.random.randint(1, width)
        mu[k] = image[mu_h, mu_w, :]

    # Iteration
    Z = np.zeros((height, width, K))
    print(np.sum(Z[:, :, 1]))
    iter = 0
    max_iter = 15
    while iter < max_iter:
        prev_Z = Z
        for h in range(height):
            for w in range(width):
                dist2mu = float('inf')
                for k in range(K):
                    norm = mu[k] - image[h, w, :]
                    norm = np.sqrt(np.sum(np.power(norm, 2)))
                    if norm < dist2mu:
                        dist2mu = norm
                        Z[h, w, :] = 0
                        Z[h, w, k] = 1
        iter += 1
        print(np.sum(prev_Z - Z))
        print("Iteration:%d" % iter)
        print('Valid Mu=')
        print(mu)

        for k in range(K):
            for channel in range(3):
                mu[k, channel] = np.sum(image[:, :, channel] * Z[:, :, k]) / np.sum(Z[:, :, k])
    NaNIdx = np.where([])
    segmented_image = np.zeros_like(image)
    for k in range(K):
        for channel in range(3):
            segmented_image[:, :, channel] += mu[k, channel] * Z[:, :, k]

    plt.figure(2)
    plt.imshow(segmented_image)
    plt.show()
    plt.title('K=')
    plt.close()

if __name__ == '__main__':
    likelihood = loadmat('likelihood.mat')
    likelihoodA = likelihood.get('likeA')
    likelihoodB = likelihood.get('likeB')
    print(likelihoodB)
    ax = plt.figure(1)
    plt.xlabel('Iteration')
    plt.ylabel('Likelihood')
    plt.plot(likelihoodA[0], label='DatasetA')
    plt.plot(likelihoodB[0], label='DatasetB')
    plt.grid()
    plt.xlim([0, 20])
    plt.xticks(np.arange(0, 21, 5))
    plt.title('Likelihood by Iteration')
    plt.legend(loc=4)
    plt.show()
