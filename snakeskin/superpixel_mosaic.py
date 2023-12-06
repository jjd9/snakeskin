import cv2
import numpy as np
from tqdm import tqdm
import time
from multiprocessing import Pool

from snakeskin.mosaic import Mosaic

def cropContour(image, contour_mask, contour_bbox, output_image):

    # find the contour bbox
    x,y,w,h = contour_bbox

    # resize the image to that bbox
    resized_image = cv2.resize(image, (w,h), interpolation=cv2.INTER_NEAREST)

    # replace contour with the image
    output_image[...] = 0
    output_image[y:y+h, x:x+w, :] = cv2.bitwise_and(resized_image, resized_image, mask=contour_mask[y:y+h, x:x+w])
    return output_image

def f(i):
    global g_mosaic_images, g_contour_mask, g_contour_bbox, g_label_mask, g_patch, g_output_image
    mosaic_patch = cropContour(g_mosaic_images[i], g_contour_mask, g_contour_bbox, g_output_image)[g_label_mask,:]
    dist = np.abs(g_patch - mosaic_patch).sum()
    return dist


class SuperPixel(Mosaic):

    def _create(self, original_input_image, mosaic_images, show_lines):
        global g_mosaic_images, g_contour_mask, g_contour_bbox, g_label_mask, g_patch, g_output_image, g_output_image

        input_image = cv2.resize(original_input_image, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)

        num_mosaic_images = len(mosaic_images)

        if self.verbose:
            print("Compute superpixels")

        output = cv2.ximgproc.createSuperpixelLSC(input_image)
        start = time.time()
        output.iterate()
        output.enforceLabelConnectivity()
        if self.verbose:
            print(f"Computing super pixels took: {time.time() - start} seconds")
        labels = output.getLabels()
        num_labels = output.getNumberOfSuperpixels()
        contour_lines = output.getLabelContourMask()
        contour_lines = cv2.resize(contour_lines, (original_input_image.shape[1], original_input_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # for each label convert it to a contour, find the mean color, and get the image with the nearest mean
        # color
        raw_mosaic_image = np.zeros_like(original_input_image)
        if self.verbose:
            print("Construct mosaic")

        output_image_buffer = np.zeros_like(input_image)
        g_mosaic_images = mosaic_images 
        g_output_image = output_image_buffer

        for label in tqdm(range(num_labels)):
            label_mask = labels == label
            patch = input_image[label_mask,:].astype(int)

            # find the contour
            contour_mask = (label_mask.astype(np.uint8) * 255)
            resized_contour_mask = cv2.resize(contour_mask, (original_input_image.shape[1], original_input_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            resized_label_mask = resized_contour_mask == 255
            contours, _ = cv2.findContours(contour_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            g_contour_mask = contour_mask 
            g_label_mask = label_mask
            g_patch = patch

            for contour_poly in contours:
                # find the contour bbox
                contour_bbox = cv2.boundingRect(contour_poly)

                g_contour_bbox = contour_bbox 

                # processes == os.cpu_count()
                with Pool(processes=None) as pool:
                    dists = pool.map(f, list(range(num_mosaic_images)))
                min_index = np.argmin(dists)
                
                x_scale = original_input_image.shape[1] / input_image.shape[1]
                y_scale = original_input_image.shape[0] / input_image.shape[0]
                resized_contour_bbox = [
                    int(contour_bbox[0] * x_scale),
                    int(contour_bbox[1] * y_scale),
                    int(contour_bbox[2] * x_scale),
                    int(contour_bbox[3] * y_scale),
                ]
                raw_mosaic_image[resized_label_mask,:] = cropContour(mosaic_images[min_index], resized_contour_mask, resized_contour_bbox, np.zeros_like(raw_mosaic_image))[resized_label_mask,:]

        if show_lines:
            raw_mosaic_image = cv2.bitwise_and(raw_mosaic_image, raw_mosaic_image, mask=~contour_lines)

        return raw_mosaic_image
