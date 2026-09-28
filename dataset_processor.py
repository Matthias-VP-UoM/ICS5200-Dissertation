import cv2
import PIL
import os
from pathlib import Path
import random
import shutil
from tqdm import tqdm

ds_folder = 'dataset_raw'
final_ds_folder = 'dataset_final'
img_file = 'test.png'
Y_TO_CROP = 131
page_folders = ['Home Page', 'News Listing', 'Article Pages']

news_sites_list = os.listdir(ds_folder)

dir_path = Path(final_ds_folder)

def init_folder(path):
    os.makedirs(path)

def copy_and_rename_file(file_path, id=None):
    img_path_parts = file_path.split('\\')
    source_file_name = os.path.basename(file_path)
    destination_file_name = f'{str(id).zfill(4)}{os.path.splitext(source_file_name)[-1]}'
    if len(img_path_parts) == 4:
        destination_file_path = os.path.join(final_ds_folder, img_path_parts[1], img_path_parts[2], destination_file_name)
    else:
        destination_file_path = os.path.join(final_ds_folder, img_path_parts[1], img_path_parts[2], img_path_parts[3], destination_file_name)

    try:
        new_path = shutil.copy(file_path, destination_file_path)
        print(f"File {file_path} copied successfully to: {new_path}")
    except FileNotFoundError:
        print("The source file does not exist.")
    except PermissionError:
        print("Permission denied.")

def crop_img(img):
    pass

# Initialise the final dataset folder, so that the contents of the raw dataset are stored without losing important changes
raw_news_site_paths = [os.path.join(ds_folder, site, pf) for pf in page_folders for site in news_sites_list]
final_news_site_paths = [os.path.join(final_ds_folder, site, pf) for pf in page_folders for site in news_sites_list]

for site_path in final_news_site_paths:
    if 'Article Pages' in site_path:
        for sf in ['urls', 'imgs']:
            sub_folder_path = os.path.join(site_path, sf)
            
            if not os.path.exists(sub_folder_path):
                init_folder(sub_folder_path)
    else:
        if not os.path.exists(site_path):
            init_folder(site_path)

# Copy all the image files from the raw dataset into the final dataset folder and rename with id
imgs_list_full = []
selected_imgs = []
for site in news_sites_list:
    for page_folder in page_folders:
        if page_folder == 'Article Pages':
            full_path = os.path.join(ds_folder, site, page_folder, 'imgs')
        else:
            full_path = os.path.join(ds_folder, site, page_folder)
        imgs_list = [os.path.join(full_path, f) for f in os.listdir(full_path) if f.lower().endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
        imgs_list_full.append(imgs_list)

# Flatten the nested list
all_images = [img for sublist in imgs_list_full for img in sublist]
print(all_images)

for id_count, img_file in enumerate(tqdm(all_images, desc="Copying images"), start=1):
    copy_and_rename_file(img_file, id=id_count)
