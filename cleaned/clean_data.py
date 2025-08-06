import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


# Clean the ImageNet dataset, keeping the 50 images with the most information
imgnet_path = '/path/to/ImageNet/train'
clean_num = 50
cate_list = os.listdir(imgnet_path)
ith = 1
save_dir = './'

def process_image(args):
    img_name, cate_path = args
    img_path = os.path.join(cate_path, img_name)
    img_size = os.path.getsize(img_path)

    img = Image.open(img_path)
    img_pixels = img.size[0] * img.size[1]

    ratio = img_size / img_pixels
    return [ratio, img_name]

for cate_name in cate_list:
    ratios = []
    cate_path = os.path.join(imgnet_path, cate_name)
    img_list = os.listdir(cate_path)

    # Use multiprocessing.Pool to process each image in parallel
    with Pool(processes=16) as pool:  # the number of parallel processes is set to 16
        results = list(tqdm(
            pool.imap(process_image, 
                      [(img_name, cate_path) for img_name in img_list]), 
                      total=len(img_list), desc=f"Processing {cate_name}"))

    ratios = np.array(results)
    order = np.argsort(ratios[:, 0])[::-1]
    clean = ratios[order[(ith-1)*clean_num:ith*clean_num]][:, 1]

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'{cate_name}.txt'), 'w') as f:
        for item in clean:
            f.write("%s\n" % item)