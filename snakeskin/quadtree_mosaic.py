# !/usr/bin/python3

import cv2
import numpy as np
from tqdm import tqdm
import time
import faiss

import snakeskin.quadtree as quadtree
from snakeskin.mosaic import Mosaic

class Quad(Mosaic):
    def _create(self, input_image, mosaic_images, show_lines):
        input_height, input_width, _ = input_image.shape
        num_mosaic_images = len(mosaic_images)

        depth = 7
        patch_size = 70

        # Extract embeddings
        mosaic_embeddings = np.zeros((num_mosaic_images, patch_size*patch_size*3), dtype=np.float32)
        for i in tqdm(range(num_mosaic_images)):
            mosaic_embeddings[i,:] = cv2.resize(mosaic_images[i], (patch_size, patch_size), interpolation=cv2.INTER_AREA).ravel()

        # Construct quadtree over input image

        # create quadtree
        if self.verbose:
            print("Build tree")
        start = time.time()
        qtree = quadtree.QuadTree(input_image)
        if self.verbose:
            print(f"Build took: {time.time() - start}")

        # Convert each patch of the quadtree into a vector
        leaf_quadrants = qtree.get_leaf_quadrants(depth)
        if self.verbose:
            print(f"number of quadrants: {len(leaf_quadrants)}")
        patch_embeddings = np.zeros((len(leaf_quadrants), patch_size*patch_size*3), dtype=np.float32)

        for i in tqdm(range(len(leaf_quadrants))):
            quadrant = leaf_quadrants[i]
            col_min, row_min, col_max, row_max = [int(x) for x in quadrant.bbox]
            image_patch = input_image[row_min:row_max, col_min:col_max,:]
            image_patch = cv2.resize(image_patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
            patch_embeddings[i, :] = image_patch.ravel()

        # match mosaic tiles with quadtree patches
        if self.verbose:
            print(f"Build index")
        dim = patch_embeddings[0].size
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(mosaic_embeddings))

        start = time.time()

        # construct raw mosaic image
        raw_mosaic_image = np.zeros_like(input_image)
        k = 1
        _, I = index.search(np.array(patch_embeddings), k)
        for i in tqdm(range(len(leaf_quadrants))):
            quadrant = leaf_quadrants[i]
            j = I[i,0]
            col_min, row_min, col_max, row_max = [int(x) for x in quadrant.bbox]
            raw_mosaic_image[row_min:row_max, col_min:col_max,:] = cv2.resize(mosaic_images[j], (col_max-col_min, row_max-row_min), interpolation=cv2.INTER_AREA)

        if self.verbose:
            print(f"Building mosaic took: {time.time() - start}")

        if show_lines:
            # draw rectangle size of quadrant for each leaf quadrant
            for quadrant in leaf_quadrants:
                col_min, row_min, col_max, row_max = [int(x) for x in quadrant.bbox]
                raw_mosaic_image = cv2.rectangle(raw_mosaic_image, (col_min, row_min), (col_max, row_max), color=(0, 0, 0), thickness=1)

        return raw_mosaic_image
