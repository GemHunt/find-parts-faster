#http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_marked_watershed.html
import sys
import cv2
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np

from skimage.morphology import watershed, disk
from skimage.filters import rank


dir = '/home/pkrush/find-parts-faster-data/screws/2/'

img = cv2.imread(dir + '2.png',cv2.IMREAD_GRAYSCALE)
img = img[300:900,700:1600]

cv2.threshold(img, 100, 255, 0, img)
img = 255 - img


# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.cv.CV_DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255,0)
# Finding unknown region
#sure_fg = np.uint8(sure_fg)

markers = ndi.label(sure_fg)[0]

# local gradient (disk(2) is used to keep edges thin)
gradient = rank.gradient(opening, disk(2))


# process the watershed
labels = watershed(gradient, markers)

markers = np.uint8(markers)

cv2.imshow("sure_fg", sure_fg)
cv2.imshow("opening", opening)
cv2.imshow("markers", markers)
cv2.imshow("gradient", gradient)
cv2.imshow("labels", labels)

key = cv2.waitKey(0)
if key & 0xFF == ord('q'):
    sys.exit()

# display results
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title("Original")

ax[1].imshow(gradient, cmap=plt.cm.spectral, interpolation='nearest')
ax[1].set_title("Local Gradient")

ax[2].imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
ax[2].set_title("Markers")

ax[3].imshow(img, cmap=plt.cm.gray, interpolation='nearest')
ax[3].imshow(labels, cmap=plt.cm.spectral, interpolation='nearest', alpha=.2)
ax[3].set_title("Segmented")

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()