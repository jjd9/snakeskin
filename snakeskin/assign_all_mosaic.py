import cv2
import numpy as np
from math import sqrt
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool
import time
from itertools import product

from snakeskin.mosaic import Mosaic

def calcDistance(image1, image2):
    return cv2.absdiff(image1, image2).astype(int).sum()

def f(work):
    global g_mosaic_width, g_input_image, g_cell_height, g_cell_width, g_scaled_mosaic_images
    i,j = work
    row = int(i / g_mosaic_width)
    col = i % g_mosaic_width
    image_patch = g_input_image[row*g_cell_height:(row+1)*g_cell_height, col*g_cell_width:(col+1)*g_cell_width, :]
    return calcDistance(image_patch, g_scaled_mosaic_images[j])


class AssignAll(Mosaic):

    def _create(self, input_image, mosaic_images, show_lines):
        num_mosaic_images = len(mosaic_images)
        input_height, input_width, _ = input_image.shape

        # determine mosaic properties
        mosaic_height = round(sqrt(num_mosaic_images * input_height / input_width))
        cell_height = int(input_height / mosaic_height)
        mosaic_width = round(mosaic_height * input_width / input_height)
        cell_width = int(input_width / mosaic_width)
        num_mosaic_images = mosaic_height * mosaic_width

        # force height to be divisible by image height
        if cell_height % input_height != 0:
           input_height = cell_height * int(input_height / cell_height)

        mosaic_width = round(input_width / cell_width)
        # force width to be divisible by image width
        if cell_width % input_width != 0:
           input_width = cell_width * int(input_width / cell_width)
        input_image = cv2.resize(input_image, (input_width, input_height), interpolation=cv2.INTER_AREA)

        scaled_mosaic_images = [cv2.resize(img, (cell_width, cell_height), interpolation=cv2.INTER_AREA) for img in mosaic_images]

        if self.verbose:
            print("Mosaic properties:")
            print(f"Cell dims: ({cell_height}x{cell_width})")
            print(f"Mosaic dims: ({mosaic_height}x{mosaic_width})")

            print("Constructing distance matrix")

        start = time.time()
        # construct distance matrix
        distance_matrix = np.zeros((num_mosaic_images,num_mosaic_images), dtype=float)
        tasks = product(range(num_mosaic_images), repeat=2)
        tasks = list(tasks)

        # update the global variables used by the global function f
        global g_mosaic_width, g_input_image, g_cell_height, g_cell_width, g_scaled_mosaic_images
        g_mosaic_width = mosaic_width
        g_input_image = input_image
        g_cell_height = cell_height
        g_cell_width = cell_width
        g_scaled_mosaic_images = scaled_mosaic_images

        # processes == os.cpu_count()
        with Pool(processes=None) as pool:
            data = pool.map(f, tasks)
        for k, (i,j) in enumerate(tasks):
            distance_matrix[i,j] = data[k]

        if self.verbose:
            print(f"Computing distance matrix took: {time.time() - start} seconds")

        # solve assignment problem
        sol = linear_sum_assignment(distance_matrix, maximize=False)

        # construct raw mosaic image
        raw_mosaic_image = np.zeros_like(input_image)
        for i, j in zip(sol[0], sol[1]):
            row = int(i / mosaic_width)
            col = i % mosaic_width
            raw_mosaic_image[row*cell_height:(row+1)*cell_height, col*cell_width:(col+1)*cell_width, :] = scaled_mosaic_images[j]

        if show_lines:
            for i, j in zip(sol[0], sol[1]):
                row = int(i / mosaic_width)
                col = i % mosaic_width
                raw_mosaic_image = cv2.rectangle(raw_mosaic_image, (col*cell_width, row*cell_height), ((col+1)*cell_width-1, (row+1)*cell_height-1), color=(0,0,0), thickness=1)
            
        return raw_mosaic_image