"""
Base mosaic algorithm class

"""

import cv2
from glob import glob
import os
import pickle
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

def load_image(input):
    image_path, target_shape = input
    img = cv2.imread(image_path)
    target_width, target_height = target_shape
    if img.shape[1] != target_width or img.shape[0] != target_height:
        img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return img

class Mosaic:

    def __init__(self, input_image_scale = 1.0, tile_scale=0.01, alpha=0.3, use_cache=True, verbose=False, show_lines=False):
        self.input_image_scale=input_image_scale
        self.tile_scale = tile_scale
        self.alpha=alpha
        self.beta=1.0-alpha
        self.use_cache=use_cache
        self.verbose = verbose
        self.show_lines = show_lines

    def get_mosaic_images(self, album_path, target_size):
        target_width, target_height = target_size

        database_path = os.path.join(album_path, "database.pkl")
        if os.path.exists(database_path) and self.use_cache:
            mosaic_images = pickle.load(open(database_path, "rb"))
            if self.verbose:
                print("Scaling mosaic images")
            for i in tqdm(range(len(mosaic_images))):
                if mosaic_images[i].shape[1] != target_width or mosaic_images[i].shape[0] != target_height:
                    mosaic_images[i] = cv2.resize(mosaic_images[i], (target_width, target_height), interpolation=cv2.INTER_AREA)
        else:
            filetypes = ["*.jpg", "*.jpeg", "*.png"]
            candidate_mosaic_images = []
            for ending in filetypes:
                candidate_mosaic_images += glob(os.path.join(album_path, ending))
            candidate_mosaic_images = [(path, (target_width, target_height)) for path in candidate_mosaic_images]

            if self.verbose:
                print("Start reading images")

            with Pool(processes=None) as pool:
                mosaic_images = pool.map(load_image, candidate_mosaic_images)

            if self.verbose:
                print("Finshed reading images!")

            if self.use_cache:
                pickle.dump(mosaic_images, open(database_path, "wb"))
        return mosaic_images
            

    def create(self, input_image_path, album_path, output_path):
        # Read input images
        input_image = cv2.imread(input_image_path)
        if self.input_image_scale != 1.0:
            input_image = cv2.resize(input_image, None, fx=self.input_image_scale, fy=self.input_image_scale, interpolation=cv2.INTER_AREA)
        target_size = (round(input_image.shape[1] * self.tile_scale), round(input_image.shape[0] * self.tile_scale))
        mosaic_images = self.get_mosaic_images(album_path, target_size)

        # Build mosaic
        raw_mosaic_image = self._create(input_image, mosaic_images, self.show_lines)
        if raw_mosaic_image.shape != input_image.shape:
            input_image = cv2.resize(input_image, (raw_mosaic_image.shape[1], raw_mosaic_image.shape[0]), interpolation=cv2.INTER_AREA)

        # blend raw mosaic with original image
        mosaic_image = cv2.addWeighted(input_image, self.alpha, raw_mosaic_image, self.beta, 0.0)
        if self.show_lines:
            black_cells = np.all(raw_mosaic_image == 0, axis=2)
            mosaic_image[black_cells, :] = (0,0,0)        
        output_mosaic_path = os.path.join(output_path, "mosaic.png")
        cv2.imwrite(output_mosaic_path, mosaic_image)
        merged = cv2.hconcat((input_image, mosaic_image))
        side_by_side_path = os.path.join(output_path, "side_by_side.png")
        cv2.imwrite(side_by_side_path, merged)
        if self.verbose:
            print(f"Output saved to \n {output_mosaic_path}\n and \n{side_by_side_path}")

