# import torchvision, functional transforms
import torchvision
import torchvision.transforms.functional as TF
import random
import numpy as np
import cv2
import os
from PIL import Image
from tqdm.auto import tqdm

# read image in cv2
train_images_path='data/kvasir-seg/train/images'
train_masks_path='data/kvasir-seg/train/masks'
validation_images_path='data/kvasir-seg/validation/images'
validation_masks_path='data/kvasir-seg/validation/masks'
train_images=[cv2.imread(os.path.join(train_images_path, f"{i}.jpg")) for i in range(len(os.listdir(train_images_path)))]
train_masks=[cv2.imread(os.path.join(train_masks_path, f"{i}.jpg")) for i in range(len(os.listdir(train_masks_path)))]
validation_images=[cv2.imread(os.path.join(validation_images_path, f"{i}.jpg")) for i in range(len(os.listdir(validation_images_path)))]
validation_masks=[cv2.imread(os.path.join(validation_masks_path, f"{i}.jpg")) for i in range(len(os.listdir(validation_masks_path)))]

# convert all images and masks to gray scale
# train_images=[cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in train_images]
# train_masks=[cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) for mask in train_masks]
# validation_images=[cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in validation_images]
# validation_masks=[cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) for mask in validation_masks]

# scale all images to (224, 224)
# train_images=[cv2.resize(img, (224, 224)) for img in train_images]
# train_masks=[cv2.resize(mask, (224, 224)) for mask in train_masks]
# validation_images=[cv2.resize(img, (224, 224)) for img in validation_images]
# validation_masks=[cv2.resize(mask, (224, 224)) for mask in validation_masks]
# convert all images to RGB 
train_images=[cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in train_images]
train_masks=[cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) for mask in train_masks]
validation_images=[cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in validation_images]
validation_masks=[cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) for mask in validation_masks]

# convert all images to torch tensors
train_images=[torchvision.transforms.functional.to_tensor(img) for img in train_images]
train_masks=[torchvision.transforms.functional.to_tensor(mask) for mask in train_masks]
validation_images=[torchvision.transforms.functional.to_tensor(img) for img in validation_images]
validation_masks=[torchvision.transforms.functional.to_tensor(mask) for mask in validation_masks]


# for img in train_images:
#     print(img.shape)
#     # convert tensor back to numpy image
#     img=img.numpy()
#     print(img.shape)
#     exit()
# # convert all images to PIL images
# train_images=[Image.fromarray(img) for img in train_images]
# train_masks=[Image.fromarray(mask) for mask in train_masks]
# validation_images=[Image.fromarray(img) for img in validation_images]
# validation_masks=[Image.fromarray(mask) for mask in validation_masks]

# for img in train_images:
#     print(img.size)
#     exit()
# define a functional transform to perform random horizontal flip and random vertical flip
def random_flip(image, mask):
    if random.random() < 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    if random.random() < 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)
    return image, mask
# define a functional transform to perform random crop with rescaling
def random_crop(image, mask, size):
    if random.random() < 0.5:
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(image, output_size=size)
        image = TF.resized_crop(image, i, j, h, w, size)
        mask = TF.resized_crop(mask, i, j, h, w, size)
        return image, mask
    return image, mask
# define a functional transform to perform random rotation
def random_rotation(image, mask, angle):
    if random.random() < 0.5:
        angle = random.uniform(-angle, angle)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
        return image, mask
    return image, mask
# define a functional transform to perform random brightness and contrast adjustment
def random_brightness_contrast(image, mask, brightness, contrast):
    if random.random() < 0.5:
        brightness = random.uniform(0, brightness)
        contrast = random.uniform(0, contrast)
        image = TF.adjust_brightness(image, brightness)
        image = TF.adjust_contrast(image, contrast)
        return image, mask
    return image, mask

# define a functional transform to perfrom random erase
def random_erase(image, mask, p, value, inplace=False): 
    i, j, h, w, _ = torchvision.transforms.RandomErasing.get_params(image, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    if random.random() < p:
        image = TF.erase(image, i, j, h, w, value, inplace)
        mask = TF.erase(mask, i, j, h, w, value, inplace)
        return image, mask
    return image, mask

# define a function MyAugmentation to perform all the above defined functional transforms
def MyAugmentation(image, mask):
    image, mask = random_flip(image, mask)
    image, mask = random_crop(image, mask, size=(224, 224))
    image, mask = random_rotation(image, mask, angle=45)
    image, mask = random_brightness_contrast(image, mask, brightness=2, contrast=2)
    image, mask = random_erase(image, mask, p=0.2, value=0, inplace=False)
    return image, mask

os.makedirs("data/kvasir-seg-aug/", exist_ok=True)
os.makedirs("data/kvasir-seg-aug/train/", exist_ok=True)
os.makedirs("data/kvasir-seg-aug/train/images/", exist_ok=True)
os.makedirs("data/kvasir-seg-aug/train/masks/", exist_ok=True)
os.makedirs("data/kvasir-seg-aug/validation/", exist_ok=True)
os.makedirs("data/kvasir-seg-aug/validation/images/", exist_ok=True)
os.makedirs("data/kvasir-seg-aug/validation/masks/", exist_ok=True)
# first save the original train images and masks after converting them to numpy array and changing to BGR
pbar=tqdm(total=len(train_images))
for i in range(len(train_images)):
    #change image and mask to numpy, then to BGR and save
    # permute the channel dimension to the last dimension
    img=TF.resize(train_images[i], size=(224, 224))
    mask=TF.resize(train_masks[i], size=(224, 224))
    img=cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
    mask=cv2.cvtColor(mask.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
    # convert img to [0, 255] and to uint8
    img=(img*255).astype(np.uint8)
    mask=(mask*255).astype(np.uint8)
    cv2.imwrite(f"data/kvasir-seg-aug/train/images/{i}.jpg", img)
    cv2.imwrite(f"data/kvasir-seg-aug/train/masks/{i}.jpg", mask)
    pbar.update(1)
# perform data augmentation on train images and masks for 3 times
for i in range(3):
    pbar = tqdm(range(len(train_images)))
    for j in range(len(train_images)):
        image, mask = MyAugmentation(train_images[j], train_masks[j])
        image=TF.resize(image, size=(224, 224))
        mask=TF.resize(mask, size=(224, 224))
        img = cv2.cvtColor(image.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
        mask = cv2.cvtColor(mask.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
        img=(img*255).astype(np.uint8)
        mask=(mask*255).astype(np.uint8)
        cv2.imwrite(f"data/kvasir-seg-aug/train/images/{j}_{i}.jpg", img)
        cv2.imwrite(f"data/kvasir-seg-aug/train/masks/{j}_{i}.jpg", mask)
        pbar.update(1)

# save the original validation images and masks after converting them to numpy array
pbar = tqdm(range(len(validation_images)))
for i in range(len(validation_images)):
    img=TF.resize(validation_images[i], size=(224, 224))
    mask=TF.resize(validation_masks[i], size=(224, 224))
    img=cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
    mask=cv2.cvtColor(mask.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
    img=(img*255).astype(np.uint8)
    mask=(mask*255).astype(np.uint8)
    cv2.imwrite(f"data/kvasir-seg-aug/validation/images/{i}.jpg", img)
    cv2.imwrite(f"data/kvasir-seg-aug/validation/masks/{i}.jpg", mask)
    pbar.update(1)