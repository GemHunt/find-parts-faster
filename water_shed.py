#http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_marked_watershed.html
import sys
import cv2
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np
import time
from skimage.morphology import watershed, disk
from skimage.filters import rank
import part_image


def get_watershed_contours(img):
    start_time = time.time()
    # noise removal

    img = 255 - img.copy()
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.cv.CV_DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    # Finding unknown region
    # sure_fg = np.uint8(sure_fg)

    markers = ndi.label(sure_fg)[0]
    #print 'Done in %s seconds' % (time.time() - start_time,)
    # local gradient (disk(2) is used to keep edges thin)
    gradient = rank.gradient(opening, disk(2))

    #print 'Done in %s seconds' % (time.time() - start_time,)
    # process the watershed
    labels = watershed(gradient, markers)

    #print 'Done in %s seconds' % (time.time() - start_time,)

    markers = np.uint8(markers)
    #labels = part_image.add_border(labels,1,0)

    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue

        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(labels.shape, dtype="uint8")
        mask[labels == label] = 255
        mask = 255 - mask
        mask[img == mask] = 255
        mask = 255 - mask
        # cv2.imshow( "mask", mask)
        # key = cv2.waitKey(0)
        # if key & 0xFF == ord('q'):
        #     sys.exit()

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        #c = max(cnts, key=cv2.contourArea)
        for c in cnts:
            #if len(c) > 4:
            #for x in range(0,mask.shape[0]):
                #for y in range(0,mask.shape[1]):
                    #print mask[x,y],
               #print

            area = cv2.contourArea(c)
            min_area = 210
            max_area = 330

            if min_area < area < max_area:
                print label, ' contour length,area:', len(c),area

            # draw a circle enclosing the object
            #((x, y), r) = cv2.minEnclosingCircle(c)
            #cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 2)
            #cv2.putText(img, "#{}".format(label), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)



    #cv2.imshow("sure_fg", sure_fg)
    #cv2.imshow("opening", opening)
    #cv2.imshow("markers", markers)
    #cv2.imshow("gradient", gradient)
    #cv2.imshow("labels", labels)

    # display results
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True, sharey=True,
    #                          subplot_kw={'adjustable': 'box-forced'})
    # ax = axes.ravel()
    #
    # ax[0].imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    # ax[0].set_title("Original")
    #
    # ax[1].imshow(gradient, cmap=plt.cm.spectral, interpolation='nearest')
    # ax[1].set_title("Local Gradient")
    #
    # ax[2].imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
    # ax[2].set_title("Markers")
    #
    # ax[3].imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    # ax[3].imshow(labels, cmap=plt.cm.spectral, interpolation='nearest', alpha=.2)
    # ax[3].set_title("Segmented")
    #
    # for a in ax:
    #     a.axis('off')
    #
    # fig.tight_layout()
    # plt.show()
    # key = cv2.waitKey(0)
    # if key & 0xFF == ord('q'):
    #     sys.exit()

    return labels