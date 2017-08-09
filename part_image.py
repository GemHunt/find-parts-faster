import cv2
import numpy as np

def add_border(img, width,color):
    h,w = img.shape

    w += width * 2
    h += width * 2

    img_with_border = np.zeros((h, w), dtype=np.uint8)
    img_with_border = img_with_border + color
    img_with_border[width:h - width, width:w - width] = img.copy()
    return img_with_border
