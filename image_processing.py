import cv2
import PIL
import os
import random
from tqdm import tqdm

ds_folder = 'dataset_final'
img_file = 'test.png'

# Variables to signify amount of cropping within the X-axis and Y-axis
X_TO_CROP = 19
Y_TO_CROP = 131


page_folders = ['Home Page', 'News Listing', 'Article Pages']

news_sites_list = os.listdir(ds_folder)

# Iterate through subfolders and add all images into imgs_list_full list
selected_imgs = []
imgs_list_full = []
for site in news_sites_list:
    for page_folder in page_folders:
        if page_folder == 'Article Pages':
            full_path = os.path.join(ds_folder, site, page_folder, 'imgs')
        else:
            full_path = os.path.join(ds_folder, site, page_folder)
        imgs_list = [os.path.join(full_path, f) for f in os.listdir(full_path) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
        if len(imgs_list) == 0:
            continue
        imgs_list_full.append(imgs_list)

# Iterate through each image and perform the necessary preprocessing
for img_list in tqdm(imgs_list_full, desc='Processing images...'):
    for img_file in img_list:
        img = cv2.imread(img_file)
        if img is None:
            raise ValueError(f"Unable to load image: {img_file}")
        
        w, h = img.shape[1], img.shape[0]

        if w < 1008:
            continue
        
        if h < 880:
            Y_TO_CROP = 0
        
        crop_img = img[Y_TO_CROP:h, :w-X_TO_CROP]

        if Y_TO_CROP == 0:
            Y_TO_CROP = 131

        cv2.imwrite(img_file, crop_img)

cv2.destroyAllWindows()

# img = cv2.imread(img_file)
# if img is None:
#     raise ValueError(f"Unable to load image: {img_file}")

# w, h = img.shape[1], img.shape[0]

# print('Width:', img.shape[1])
# print('Height:', img.shape[0])

# crop_img = img[Y_TO_CROP:h, :w]

# cv2.imshow('femeifeifijfewe', crop_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
