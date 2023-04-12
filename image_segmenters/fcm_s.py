# implement fuzzy c-means with spatial information
import numpy as np
from scipy.spatial.distance import cdist

class FCM_S():
    def __init__(self, n_clusters=2, m=2, max_iter=100, error=1e-5, random_state=42):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)
        self.X = X
        self.n_samples, self.n_features = X.shape
        self.U = np.random.rand(self.n_clusters, self.n_samples)
        self.U = self.U / np.sum(self.U, axis=0)
        self.centers = np.random.rand(self.n_clusters, self.n_features)
        self.centers = self.centers / np.sum(self.centers, axis=0)
        self.old_centers = np.zeros(self.centers.shape)
        self.iter = 0
        while self.iter < self.max_iter:
            self.old_centers = self.centers
            self.centers = self.next_centers()
            self.U = self.next_U()
            self.iter += 1
            if np.linalg.norm(self.centers - self.old_centers) < self.error:
                break

    def next_centers(self):
        um = self.U ** self.m
        return um.dot(self.X) / np.sum(um, axis=1)[:, np.newaxis]

    def next_U(self):
        dist = cdist(self.X, self.centers)
        distm = dist ** (-1 / (self.m - 1))
        return (distm / np.sum(distm, axis=1)[:, np.newaxis]).T

    def predict(self, X):
        dist = cdist(X, self.centers)
        return np.argmin(dist, axis=1)
