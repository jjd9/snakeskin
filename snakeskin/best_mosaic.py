import cv2
import numpy as np
from tqdm import tqdm
import time
import faiss

from snakeskin.mosaic import Mosaic

def get_vec(img):
    return img.ravel()

class Best(Mosaic):

    def _create(self, input_image, mosaic_images, show_lines):        
        num_mosaic_images = len(mosaic_images)
        input_height, input_width, _ = input_image.shape

        # determine mosaic properties
        cell_height = mosaic_images[0].shape[0]
        cell_width = mosaic_images[0].shape[1]
        mosaic_height = round(input_height / cell_height)
        # force height to be divisible by image height
        if mosaic_height % input_height != 0:
           input_height = mosaic_height * round(input_height / mosaic_height)

        mosaic_width = round(input_width / cell_width)
        # force width to be divisible by image width
        if mosaic_width % input_width != 0:
           input_width = mosaic_width * round(input_width / mosaic_width)

        input_image = cv2.resize(input_image, (input_width, input_height), interpolation=cv2.INTER_AREA)
        total_tiles = mosaic_height * mosaic_width

        if self.verbose:
            print(f"Input image size: ({input_height}x{input_width})")
            print("Mosaic properties:")
            print(f"Cell dims: ({cell_height}x{cell_width})")
            print(f"Mosaic dims: ({mosaic_height}x{mosaic_width})")

            print("Compute image embeddings")
        start = time.time()
        mosaic_embeddings = np.zeros((num_mosaic_images, cell_width * cell_height * 3), dtype=np.float32)
        patch_embeddings = np.zeros((total_tiles, cell_width * cell_height * 3), dtype=np.float32)
        for i in tqdm(range(total_tiles)):
            row = int(i / mosaic_width)
            col = i % mosaic_width
            image_patch = input_image[row*cell_height:(row+1)*cell_height, col*cell_width:(col+1)*cell_width, :]
            patch_embeddings[i,:] = get_vec(image_patch).astype(np.float32)

        for i in tqdm(range(num_mosaic_images)):
            mosaic_embeddings[i,:] = get_vec(mosaic_images[i]).astype(np.float32)

        if self.verbose:
            print(f"Computing embeddings took: {time.time() - start}")

            print(f"Build index")
        dim = patch_embeddings[0].size
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(mosaic_embeddings))

        start = time.time()

        # construct raw mosaic image
        raw_mosaic_image = np.zeros_like(input_image)
        k = 1
        _, I = index.search(np.array(patch_embeddings), k)
        for i in range(total_tiles):
            if k == 1:
                j = I[i,0]
            else:
                j = I[i,np.random.randint(0,k-1)]
            row = int(i / mosaic_width)
            col = i % mosaic_width
            raw_mosaic_image[row*cell_height:(row+1)*cell_height, col*cell_width:(col+1)*cell_width, :] = mosaic_images[j]
        
        if show_lines:
            for i in range(total_tiles):
                row = int(i / mosaic_width)
                col = i % mosaic_width
                raw_mosaic_image = cv2.rectangle(raw_mosaic_image, (col*cell_width, row*cell_height), ((col+1)*cell_width-1, (row+1)*cell_height-1), color=(0,0,0), thickness=1)

        return raw_mosaic_image
