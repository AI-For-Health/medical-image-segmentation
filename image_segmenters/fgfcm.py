import numpy as np
from scipy.spatial.distance import cdist

class FGFCM():
    def __init__(self, n_clusters=2, m=2, window_size=3, lambda_g=3, lambda_s=1, num_gray_levels=256, max_iter=1000, error=1e-5, random_state=42):
        self.n_clusters = n_clusters
        self.m = m
        self.window_size = window_size
        self.lambda_g = lambda_g
        self.lambda_s = lambda_s
        self.q = num_gray_levels  # number of gray levels   
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state
    
    def fit(self, X):
        np.random.seed(self.random_state)
        self.X = X
        self.centers = np.random.rand(self.n_clusters)

        m, n = X.shape
        N_R=(self.window_size**2)-1
        # pad X with zeros
        self.X = np.pad(self.X, pad_width=self.window_size//2, mode='constant', constant_values=0)
        Ss=np.array([[0]])
        for i in range(self.window_size//2):
            Ss=np.pad(Ss, pad_width=1, mode='constant', constant_values=i+1)
        Ss=Ss/self.lambda_s
        # an empty numpy array of shape (m, n, window_size, window_size)
        Sg = np.empty((m, n, self.window_size, self.window_size))
        # fill Sg[i,j] with euclidean distance matrix of pixel (i,j) and its neighbors
        for i in range(m):
            for j in range(n):
                # get neighbors of pixel (i, j) in the unpadded X
                neighbors = self.X[i:i+self.window_size, j:j+self.window_size]
                # compute euclidean distance matrix of pixel (i,j) and its neighbors
                Sg[i,j] = cdist(neighbors.reshape(-1, 1), self.X[i+self.window_size//2,j+self.window_size//2].reshape(1, 1)).reshape(self.window_size, self.window_size)**2
                Sg[i,j] = (Sg[i,j] / np.sum(Sg[i,j]+1e-10))*(N_R/self.lambda_g)
                Sg[i,j] = np.exp(-Sg[i,j])
        S = np.empty((m, n, self.window_size, self.window_size))
        for i in range(m):
            for j in range(n):
                S[i,j]  = Sg[i,j] * Ss
        Xi_matrix = np.empty((m,n))
        for i in range(m):
            for j in range(n):
                Xi_matrix[i,j] = np.sum(self.X[i:i+self.window_size, j:j+self.window_size] * S[i,j])/np.sum(S[i,j])
        self.bin_edges=np.arange(-1/(255*2), 1+1/(255), 1/255)
        self.gamma_l, _ = np.histogram(Xi_matrix, bins=self.bin_edges)
        self.Xi_l = np.arange(256)/255.0
        self.U = np.random.rand(self.n_clusters, self.q)
        self.old_centers=np.zeros(self.centers.shape)
        self.iter = 0
        while self.iter < self.max_iter:
            self.U = self.next_U()
            self.old_centers = self.centers
            self.centers = self.next_centers()
            self.iter += 1
            if np.linalg.norm(self.centers - self.old_centers) < self.error:
                break

    def next_centers(self):
        w=self.gamma_l*self.U
        return (np.dot(w, self.Xi_l))/np.sum(w)

    def next_U(self):
        self.U=self.Xi_l.reshape((-1,1))-self.centers.reshape((1,-1))
        self.U=self.U ** (-2/(self.m-1))
        return (self.U / np.sum(self.U, axis=0)).T
    
    def predict(self, X):
        dist = cdist(X.reshape((-1,1)), self.centers.reshape((-1,1)))
        return np.argmin(dist, axis=1)