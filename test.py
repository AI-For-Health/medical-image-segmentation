import numpy as np
from scipy.spatial.distance import cdist

image_size=512
X=np.arange(image_size**2).reshape(image_size,image_size)
m, n = X.shape
window_size = 3
N_R=window_size**2
lambda_g=0.5

# pad X with zeros
X = np.pad(X, window_size//2, mode='constant')

# Create a spatial matrix Ss of shape (windows_size, window_size)
Ss=np.array([[0]])
for i in range(window_size//2):
    Ss=np.pad(Ss, pad_width=1, mode='constant', constant_values=i+1)
print(Ss)

# an empty numpy array of shape (m, n, window_size, window_size)
Sg = np.empty((m, n, window_size, window_size))

# fill Sg[i,j] with euclidean distance matrix of pixel (i,j) and its neighbors
for i in range(m):
    for j in range(n):
        # get neighbors of pixel (i, j) in the unpadded X
        neighbors = X[i:i+window_size, j:j+window_size]
        # compute euclidean distance matrix of pixel (i,j) and its neighbors
        Sg[i,j] = cdist(neighbors.reshape(-1, 1), X[i+window_size//2,j+window_size//2].reshape(1, 1)).reshape(window_size, window_size)**2
        Sg[i,j] = (Sg[i,j] / np.sum(Sg[i,j]))*(N_R/lambda_g)
        Sg[i,j] = np.exp(-Sg[i,j])

S = np.empty((m, n, window_size, window_size))
for i in range(m):
    for j in range(n):
        S[i,j]  = Sg[i,j] * Ss

Xi_matrix = np.empty((m,n))
for i in range(m):
    for j in range(n):
        Xi_matrix[i,j] = np.sum(X[i:i+window_size, j:j+window_size] * S[i,j])/np.sum(S[i,j])

print(Xi_matrix)


import numpy as np
from datasets import load_dataset
import cv2

dataset = load_dataset('kowndinya23/Kvasir-SEG')

# get first image in gray scale
img = np.array(dataset['validation']['image'][0])
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# convert image into [0,1] range
img = img/255

# get histogram of image
bin_edges=np.arange(-1/(255*2), 1+1/(255), 1/255)
hist, _ = np.histogram(img, bins=bin_edges)
print(hist.shape)