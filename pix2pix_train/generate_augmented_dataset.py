''''

Shuntos-24-July-2021

'''

import cv2
import numpy as np 
import os 
from imgaug import augmenters as iaa

seq = iaa.Sequential([
    iaa.Rotate((-10, 10)),

    iaa.PiecewiseAffine(scale=(0.01, 0.07)),
    iaa.ShearX((-10, 10))
])


def get_mask(mask, change_color = True ):
    if change_color:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        (thresh, mask) = cv2.threshold(mask, 10, 255, 0)
    try:
        (_, contours, hierarchy) = cv2.findContours(image = mask, 
            mode = cv2.RETR_EXTERNAL,
            method = cv2.CHAIN_APPROX_SIMPLE)
    except:
        (contours, hierarchy) = cv2.findContours(image = mask, 
            mode = cv2.RETR_EXTERNAL,
            method = cv2.CHAIN_APPROX_SIMPLE)

    contours_sizes= [(cv2.contourArea(cnt), cnt) for cnt in contours]
    biggest_contour = max(contours_sizes, key=lambda x: x[0])[1]


    blank_img = np.zeros([mask.shape[0],mask.shape[1],3], dtype=np.uint8)

    blank_img = cv2.drawContours(blank_img, [biggest_contour], -1, (255, 255, 255), -1)

    return blank_img


input_folder = "img_folders/"
dim = (512,512)
train_folder_A = "dataset_augmented/train_A/"
train_folder_B = "dataset_augmented/train_B/"
# Creating directory if not exists 
if not os.path.exists(train_folder_A):
    os.makedirs(train_folder_A)
if not os.path.exists(train_folder_B):
    os.makedirs(train_folder_B)

count = 600
for folder in os.listdir(input_folder):
    print(folder[-5:])
    if folder[-5:] == "cloth":
        sub_folder = input_folder+folder+"/"
        for file in os.listdir( sub_folder):
            img_file = sub_folder+ file
            mask_file = input_folder + folder[:-5]+"mask/"+file

            print(img_file, mask_file)
            img = cv2.imread(img_file)
            img = cv2.resize(img, dim)
            mask = cv2.imread(mask_file)
            mask = cv2.resize(mask, dim)
            mask_refined= get_mask(mask)
            original_img = img.copy()
            imglist_img = []
            imglist_mask = []
            imglist_img.append(img)
            imglist_mask.append(mask_refined)

            for i in range(2):
                aug_image, aug_mask =seq (images=imglist_img, segmentation_maps=imglist_mask)            

                #Concate two image to create single image
                concatenated =  cv2.hconcat([original_img, aug_mask[0]])
                train_A_file = train_folder_A+ str(count) +".jpg"
                cv2.imwrite(train_A_file, concatenated)

                blank_img = np.zeros([mask.shape[0],mask.shape[1],3], dtype=np.uint8)
                concatenated_out = cv2.hconcat([aug_image[0], blank_img])
                train_B_file = train_folder_B +str(count)+".jpg"
                cv2.imwrite(train_B_file, concatenated_out)
                count +=1 
