#!/usr/bin/env python3

### Converts image type/name.

from PIL import Image
import os

int_stream = '000000'
folder = 9
data_dir = '/home/jc-merlab/lama/results/08_22_2023/rcnn'


if __name__ == '__main__':
    data_files = os.listdir(data_dir)  # all files in the data folder
    # Ending with...
    in_files = sorted([f for f in data_files if f.endswith('.png')])

    for f in in_files:
        # process file names

        name, ext = os.path.splitext(f)

        # To .png
        # new_img_filename = name + '.png'
        # To .jpg
        # new_img_filename = name + '.jpg'
        # To .rgb.jpg
        new_img_filename = name[:-5] + '.jpg'

        print(f'Renaming {f} to {new_img_filename}')
        image = Image.open(os.path.join(data_dir, f))
        image.save(os.path.join(data_dir, new_img_filename))
