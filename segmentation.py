import cv2
import os
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu


def _get_images(dataset_dir):
    photo_filenames = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir,filename)
        photo_filenames.append(path)
    print(photo_filenames)
    return photo_filenames

data_set = _get_images("retrain_hand\\test\\")

for i in data_set:
    image = cv2.imread(i)
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = threshold_otsu(gray)
    binary = gray > thresh

    plt.imsave("retrain_hand\\output\\"+i.split("\\")[-1]+"_masked.jpg",binary)