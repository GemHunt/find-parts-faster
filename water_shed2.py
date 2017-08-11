#https://stackoverflow.com/questions/11294859/how-to-define-the-markers-for-watershed-in-opencv
#Nice post!

import sys
import cv2
import numpy
from scipy.ndimage import label

def segment_on_dt(a, img):
    border = cv2.dilate(img, None, iterations=5)
    border = border - cv2.erode(border, None)

    dt = cv2.distanceTransform(img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(numpy.uint8)
    _, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
    lbl, ncc = label(dt)
    lbl = lbl * (255/ncc)
    # Completing the markers now.
    lbl[border == 255] = 255

    lbl = lbl.astype(numpy.int32)
    cv2.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(numpy.uint8)
    return 255 - lbl


def get_watershed_contours(img_to_search,min_area,max_area,threshold_dist, global_x,global_y):
    img_to_search = 255 - img_to_search
    img_to_search = cv2.cvtColor(img_to_search, cv2.COLOR_GRAY2BGR)

    img_gray = cv2.cvtColor(img_to_search, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, 0, 255,cv2.THRESH_OTSU)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,numpy.ones((3, 3), dtype=int))

    result = segment_on_dt(img_to_search, img_bin)
    #cv2.imshow("result", result)

    result[result != 255] = 0
    result = cv2.dilate(result, None)
    img_to_search[result == 255] = (0, 0, 255)

    #cv2.imshow("img", img_to_search)
    #key = cv2.waitKey(0)
    #if key & 0xFF == ord('q'):
        #sys.exit()
    return None
