from datasets import load_dataset
from image_segmenters import FGFCM
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm.auto import tqdm

dataset=load_dataset("kowndinya23/Kvasir-SEG")

def preprocess(img):
    img = np.array(img)
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # apply median based otsu thresholding to get roi image
    blurred = cv2.medianBlur(img, 5)
    _, roi = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    roi_img=(img*roi)
    roi_img=roi_img.astype(np.uint8)
    # Edge detection on roi image
    blurred_roi_img=cv2.GaussianBlur(roi_img, (5, 5), 0)
    edges = cv2.Canny(blurred_roi_img, 240, 250)
    # perfrom dilation on edges
    kernel = np.ones((3,3),np.uint8)
    edges = cv2.dilate(edges,kernel,iterations = 1)
    # subtract edges from roi image
    roi_img=roi_img-edges
    # convert roi image to [0,1] range
    roi_img=roi_img/255
    return roi_img

def binary_dice_score(y_true, y_pred):
    '''Compute the Sørensen-Dice coefficient for binary segmentations.'''
    overlap = y_true*y_pred # Logical AND
    union = y_true+y_pred # Logical OR
    iou_score = np.divide(np.count_nonzero(overlap),np.count_nonzero(union)) #
    return 2*iou_score/(iou_score+1)

pbar=tqdm(total=len(dataset["validation"]["image"]))
dice_scores=[]
for idx in range(len(dataset["validation"]["image"])):
    fcm=FGFCM(n_clusters=2, m=2, max_iter=100, error=1e-5, random_state=42)
    X=preprocess(dataset["validation"]["image"][idx])
    fcm.fit(X)
    labels=fcm.predict(X).reshape(np.array(dataset["validation"]["image"][idx]).shape[:2])
    # save labels to disk
    plt.imsave(f"debugging/labels_{idx}.png", 1-labels, cmap="gray")
    dice_scores.append(binary_dice_score(np.array(dataset["validation"]["annotation"][idx]), 1-labels))
    print(dice_scores[-1])
    pbar.update(1)

print("Mean Dice score: ", np.mean(dice_scores))