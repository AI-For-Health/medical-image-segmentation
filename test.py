# import numpy as np
# from scipy.spatial.distance import cdist

# image_size=512
# X=np.arange(image_size**2).reshape(image_size,image_size)
# m, n = X.shape
# window_size = 3
# N_R=window_size**2
# lambda_g=0.5

# # pad X with zeros
# X = np.pad(X, window_size//2, mode='constant')

# # Create a spatial matrix Ss of shape (windows_size, window_size)
# Ss=np.array([[0]])
# for i in range(window_size//2):
#     Ss=np.pad(Ss, pad_width=1, mode='constant', constant_values=i+1)
# print(Ss)

# # an empty numpy array of shape (m, n, window_size, window_size)
# Sg = np.empty((m, n, window_size, window_size))

# # fill Sg[i,j] with euclidean distance matrix of pixel (i,j) and its neighbors
# for i in range(m):
#     for j in range(n):
#         # get neighbors of pixel (i, j) in the unpadded X
#         neighbors = X[i:i+window_size, j:j+window_size]
#         # compute euclidean distance matrix of pixel (i,j) and its neighbors
#         Sg[i,j] = cdist(neighbors.reshape(-1, 1), X[i+window_size//2,j+window_size//2].reshape(1, 1)).reshape(window_size, window_size)**2
#         Sg[i,j] = (Sg[i,j] / np.sum(Sg[i,j]))*(N_R/lambda_g)
#         Sg[i,j] = np.exp(-Sg[i,j])

# S = np.empty((m, n, window_size, window_size))
# for i in range(m):
#     for j in range(n):
#         S[i,j]  = Sg[i,j] * Ss

# Xi_matrix = np.empty((m,n))
# for i in range(m):
#     for j in range(n):
#         Xi_matrix[i,j] = np.sum(X[i:i+window_size, j:j+window_size] * S[i,j])/np.sum(S[i,j])

# print(Xi_matrix)


# import numpy as np
# from datasets import load_dataset
# import cv2

# dataset = load_dataset('kowndinya23/Kvasir-SEG')

# # get first image in gray scale
# img = np.array(dataset['validation']['image'][0])
# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# # convert image into [0,1] range
# img = img/255

# # get histogram of image
# bin_edges=np.arange(-1/(255*2), 1+1/(255), 1/255)
# hist, _ = np.histogram(img, bins=bin_edges)
# print(hist.shape)
import os
import numpy as np
import cv2
from datasets import load_dataset

datatset=load_dataset('kowndinya23/Kvasir-SEG')

os.makedirs('data/kvasir-seg', exist_ok=True)
os.makedirs('data/kvasir-seg/train', exist_ok=True)
os.makedirs('data/kvasir-seg/validation', exist_ok=True)
# make images and masks directory in train and validation subdirectories above
os.makedirs('data/kvasir-seg/train/images', exist_ok=True)
os.makedirs('data/kvasir-seg/train/masks', exist_ok=True)
os.makedirs('data/kvasir-seg/validation/images', exist_ok=True)
os.makedirs('data/kvasir-seg/validation/masks', exist_ok=True)

# save images and masks in train and validation subdirectories above
N_train=len(datatset['train'])
for i in range(N_train):
    # get image in gray scale
    img = np.array(datatset['train'][i]['image'])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # get mask in gray scale
    mask = np.array(datatset['train'][i]['annotation'])
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'data/kvasir-seg/train/images/{i}.jpg', img)
    cv2.imwrite(f'data/kvasir-seg/train/masks/{i}.jpg', mask)
# do the same for validation split
N_validation=len(datatset['validation'])
for i in range(N_validation):
    # get image in gray scale
    img = np.array(datatset['validation'][i]['image'])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # get mask in gray scale
    mask = np.array(datatset['validation'][i]['annotation'])
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'data/kvasir-seg/validation/images/{i}.jpg', img)
    cv2.imwrite(f'data/kvasir-seg/validation/masks/{i}.jpg', mask)


# import torch 
# # imoprt transforms from torchvision
# import torchvision
# from torchvision import transforms
# import torchvision.transforms.functional as TF
# import random
# import numpy as np
# import cv2
# from PIL import Image

# # load images in data/kvasir-seg/train/images and data/kvasir-seg/validation/images as torch datasets
# train_images = torchvision.datasets.ImageFolder('data/kvasir-seg/train/images', transform=transforms.ToTensor())
# validation_images = torchvision.datasets.ImageFolder('data/kvasir-seg/validation/images', transform=transforms.ToTensor())

# # load masks in data/kvasir-seg/train/masks and data/kvasir-seg/validation/masks as torch datasets
# train_masks = torchvision.datasets.ImageFolder('data/kvasir-seg/train/masks', transform=transforms.ToTensor())
# validation_masks = torchvision.datasets.ImageFolder('data/kvasir-seg/validation/masks', transform=transforms.ToTensor())

# # write a functional class to perform random horizontal flip and random vertical flip
# class RandomFlip(object):
#     def __init__(self, p=0.5):
#         self.p = p

#     def __call__(self, image, mask):
#         if random.random() < self.p:
#             image = TF.hflip(image)
#             mask = TF.hflip(mask)
#         if random.random() < self.p:
#             image = TF.vflip(image)
#             mask = TF.vflip(mask)
#         return image, mask

# # perform random horizontal flip and random vertical flip on images and masks
# random_flip = RandomFlip()
# train_images = [random_flip(image, mask) for image, mask in zip(train_images, train_masks)]
# validation_images = [random_flip(image, mask) for image, mask in zip(validation_images, validation_masks)]

# # create a train dataloader
# train_dataloader = torch.utils.data.DataLoader(train_images, batch_size=4, shuffle=True)
# # create a validation dataloader
# validation_dataloader = torch.utils.data.DataLoader(validation_images, batch_size=4, shuffle=True)

# for i, (image, mask) in enumerate(train_dataloader):
#     print(image.shape)
#     print(mask.shape)
#     break