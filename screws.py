import cv2
import numpy as np
import time
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def get_crop(img):
    img = img[0:1010,270:1600]
    img = cv2.resize(img,(1280,972))
    return img

def get_thresh(img,flip):
    cv2.threshold(img, 100, 255, 0, img)
    if flip:
        img = 255 - img
    #img = cv2.resize(img, (2540, 1944))

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.cv.CV_DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    contours, hierarchy = cv2.findContours(sure_fg.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    markers = np.zeros((972,1280), dtype=np.uint8)
    cv2.drawContours(markers,contours, 0, 255, -1)

    cv2.imshow("img", img)
    cv2.imshow("opening", opening)
    cv2.imshow("sure_bg", sure_bg)
    cv2.imshow("dist_transform", dist_transform)
    cv2.imshow("unknown", unknown)
    cv2.imshow("sure_fg", sure_fg)
    cv2.imshow("markers", markers)
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        sys.exit()


    return img

def get_contours(img):
    # min_area = 910
    # max_area = 1240
    # max_area = 1800

    min_area = 230
    max_area = 350

    size_factor = 1
    #small = cv2.resize(im,(int(1920 *size_factor) ,int(1080 * size_factor)))
    #img = small.copy()
    thresh = get_thresh(img,True)

    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours_filtered_in = []
    contours_filtered_out = []
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if hierarchy[0][count][3] == -1:
            if min_area * size_factor * size_factor < area < max_area * size_factor * size_factor:
                contours_filtered_in.append(cnt)
            else:
                if 100 < area:
                    contours_filtered_out.append(cnt)
                    #print area

        count +=1
    #cv2.drawContours(img,filtered_contours,-1,255,1)
    #cv2.imshow("Contours",img)
    #cv2.waitKey(0)
    return contours_filtered_in,contours_filtered_out


from matplotlib import pyplot as plt
#Use template matching to break apart contours that watershed can't

def center_rotate(img, angle):
    rows, cols = img.shape
    m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    cv2.warpAffine(img, m, (cols, rows), img, cv2.INTER_CUBIC)
    return img

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def break_contours(contours,frame,template_contour):
    start_time = time.time()

    broken_contours = []
    max_area = 350
    rect = cv2.minAreaRect(template_contour)
    angle = rect[2]

    x, y, template_width, template_height = cv2.boundingRect(template_contour)
    template_main = frame[y:y + template_height,x:x+template_width]
    templates = {}
    for angle in range(0,360,3):
        template = 255- template_main.copy()
        template = rotate_bound(template.copy(), angle)
        template = template * .5 + 127
        template = np.uint8(template)
        template = 255 - template
        templates[angle] = template

    for cnt in contours:
        max_max_val = 0
        max_angle = 0
        max_template = []
        for angle,template in templates.iteritems():
            template_height,template_width  = template.shape

            area = cv2.contourArea(cnt)
            if area > max_area:
                x, y, img_to_search_width, img_to_search_height = cv2.boundingRect(cnt)
                img_to_search = frame[y:y + img_to_search_height, x:x + img_to_search_width]

                border = 15
                img_to_search_width += border + border
                img_to_search_height += border + border

                if template_width > img_to_search_width or template_height > img_to_search_height:
                    continue

                img3 = np.zeros((img_to_search_height,img_to_search_width), dtype=np.uint8)
                img3 = img3 + 255
                img3[border:img_to_search_height-border,border:img_to_search_width-border] = img_to_search
                img2 = img3.copy()

                w, h = template.shape[::-1]

                # All the 6 methods for comparison in a list
                methods = ['cv2.TM_CCOEFF']

                for meth in methods:
                    img = img2.copy()
                    method = eval(meth)

                    # Apply template Matching

                    res = cv2.matchTemplate(img, template, method)

                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                    # # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                    # if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    #     top_left = min_loc
                    # else:
                    #     top_left = max_loc
                    # bottom_right = (top_left[0] + w, top_left[1] + h)
                    #
                    # cv2.rectangle(img, top_left, bottom_right, 128, 2)

                    if max_max_val < max_val:
                        max_max_val = max_val
                        max_angle = angle
                        max_template = template.copy()
                    #if max_val > 40075832.0:
                    # plt.subplot(121), plt.imshow(res, cmap='gray')
                    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
                    # plt.subplot(122), plt.imshow(img, cmap='gray')
                    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
                    # plt.suptitle(meth)
                    # plt.show()

        if max_max_val <> 0:
            print max_angle,max_max_val
            # cv2.imshow('max_template', max_template)
            # cv2.moveWindow('max_template', 1400, 0)
            # cv2.imshow('img3', img3)
            # cv2.moveWindow('img3', 1500, 0)
            # cv2.waitKey(0)

    print 'Done in %s seconds' % (time.time() - start_time,)

    return broken_contours

def get_features(contours):
    all_features = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        features = np.zeros(16)
        perimeter = cv2.arcLength(cnt, True)
        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        bounding_area = x * y
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        if defects is not None:
            hull_defect_count = len(cv2.convexityDefects(cnt, hull))
        features[0:9] = [x,y,angle,area, perimeter, ma, MA, bounding_area, hull_defect_count]

        mom = cv2.moments(cnt)
        hu = cv2.HuMoments(mom)
        hu = hu.flatten()
        features[9:16] = hu
        # print count, x, y, angle, area, perimeter, ma, MA, bounding_area, hull_defect_count
        # print hu
        # This is really slow to get an average mean:
        # mask = np.zeros(img.shape, np.uint8)
        # cv2.drawContours(mask, [cnt], 0, 255, -1)
        # pixelpoints = np.transpose(np.nonzero(mask))
        # mean_val = cv2.mean(img, mask=mask)
        # print mean_val
        all_features.append(features)
    return all_features

start_time = time.time()
dir = '/home/pkrush/find-parts-faster-data/screws/2/'
contours = []
for filename in os.listdir(dir):
    print dir + filename
    im = cv2.imread(dir + filename, cv2.IMREAD_GRAYSCALE)
    im = get_crop(im)
    contours_filtered_in, contours_filtered_out = get_contours(im)
    #I need to create an average one instead? contours_filtered_in[0]
    #broken_contours= break_contours(contours_filtered_out,im,contours_filtered_in[0])
    contours.extend(contours_filtered_in)
count_of_good_contours = len(contours)
dir = '/home/pkrush/find-parts-faster-data/screws/3/'
for filename in os.listdir(dir):
    print dir + filename
    im = cv2.imread(dir + filename, cv2.IMREAD_GRAYSCALE)
    im = get_crop(im)
    contours_filtered_in, contours_filtered_out = get_contours(im)
    contours.extend(contours_filtered_in)
labels = np.zeros(len(contours))
labels[0:count_of_good_contours] = 1


sys.exit()
count = 0
all_features = get_features(contours)

names = ["Nearest Neighbors", "Gaussian Process",
         "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes","RBF SVM","QDA","Decision Tree"]

classifiers = [
    KNeighborsClassifier(3),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    SVC(gamma=2, C=1),
    QuadraticDiscriminantAnalysis(),
    DecisionTreeClassifier(max_depth=5)]

for name, clf in zip(names, classifiers):

    clf.fit(all_features,labels)
    print name, 'Train %s' % (time.time() - start_time,),
    start_time = time.time()
    count = 0


    wrong_count = 0

    for features in all_features:
        X = [features]
        y = labels[count]
        predicted_class = clf.predict(X)
        #print logistic.predict_proba(X),
        #print logistic.decision_function(X),
        if predicted_class <> y:
            #print
            #print
            #print logistic.decision_function(X),
            #print clf.predict_proba(X),
            #print 'Predicted class %s, real class %s' % ( clf.predict(X),y)
            wrong_count += 1
        count +=1
    print 'Test %s' % (time.time() - start_time,),
    print wrong_count, len(all_features)

#print(logistic.coef_)
# Those values, however, will show that the second parameter
# is more influential
#print(np.std(X, 0)*logistic.coef_)

sys.exit()

cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)

for x in range(0,400000):
    start_time = time.time()
    ret, frame = cap.read()

    if frame == None:
        continue
    frame = get_crop(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = get_thresh(gray,False)

    contours, contours_filtered_out = get_contours(gray)
    all_features = get_features(contours)
    count = 0
    good_contours = []
    bad_contours = []
    for features in all_features:
        X = [features]
        predicted_class = clf.predict(X)
        if predicted_class == 0:
            good_contours.append(contours[count])
        else:
            bad_contours.append(contours[count])
        count += 1
        #if count > 30:
         #   break

    background = np.zeros((972, 1280, 3), np.uint8)
    background[:, :] = (255, 255, 255)

    cv2.drawContours(background, good_contours, -1, (150,150,255), -5)
    cv2.drawContours(background, bad_contours, -1, (220, 255, 220), 15)
    cv2.drawContours(background, bad_contours, -1, (150, 255 ,150), -5)
    cv2.drawContours(background, contours_filtered_out, -1, (255, 200, 200), -5)

    kernel = np.ones((3, 3), np.uint8)
    background = cv2.dilate(background, kernel, iterations=1)

    cv2.imshow('background', background)
    cv2.moveWindow('background', 0, 0)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    print 'Done in %s seconds' % (time.time() - start_time,)

cap.release()
cv2.destroyAllWindows()
















