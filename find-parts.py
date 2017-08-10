import cv2
import numpy as np
import time
import sys
import os
import water_shed
import template_match
import part_image
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

# min_area = 910
# max_area = 1240

min_area = 210
max_area = 330


def get_crop(img):
    img = img[0:1010,270:1600]
    img = cv2.resize(img,(1280,972))
    return img

def get_thresh(img,flip):
    cv2.threshold(img, 100, 255, 0, img)
    if flip:
        img = 255 - img
    return img

    #watershed testing:

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

def get_contours(dir):
    for filename in os.listdir(dir):
        #print dir + filename
        img = cv2.imread(dir + filename, cv2.IMREAD_GRAYSCALE)
        img = get_crop(img)

        size_factor = 1
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
            count +=1

        broken_contours = break_contours(contours_filtered_out,contours_filtered_in,img,contours_filtered_in[0])
        #todo contours_filtered_out will need to be redone, or there will be none.
    return contours_filtered_in,contours_filtered_out


def break_contours(contours,contours_filtered_in, img,template_contour):
    broken_contours = []

    templates = template_match.get_templates(img,template_contour)
    cv2.drawContours(img, contours_filtered_in, -1, 255, -1)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < max_area:
            continue

        x, y, img_to_search_width, img_to_search_height = cv2.boundingRect(cnt)
        img_to_search = img[y:y + img_to_search_height, x:x + img_to_search_width]
        img_to_search = img_to_search.copy()

        thresholds = [.1,.2,.3]
        for threshold_dist in thresholds:
            broken_contours.extend(water_shed.get_watershed_contours(img_to_search, min_area, max_area,threshold_dist,x,y))
        local_thresh = img_to_search.copy()
        local_thresh = 255 - local_thresh
        local_contours, hierarchy = cv2.findContours(local_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_still_left = False
        for local_cnt in local_contours:
            #print cv2.contourArea(local_cnt)
            if min_area < cv2.contourArea(local_cnt):
                contours_still_left = True
        if not contours_still_left:
            continue

        broken_contours.extend(template_match.break_contours(local_thresh,max_area,templates))

    # cv2.imshow('frame', frame)
    # key = cv2.waitKey(0)
    # if key & 0xFF == ord('q'):
    #     sys.exit()
    # cv2.drawContours(frame, broken_contours, -1, 255, -1)
    # cv2.imshow('frame', frame)
    # key =  cv2.waitKey(0)
    # if key & 0xFF == ord('q'):
    #     sys.exit()
    return broken_contours


def center_rotate(img, angle):
    rows, cols = img.shape
    m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    cv2.warpAffine(img, m, (cols, rows), img, cv2.INTER_CUBIC)
    return img



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
        all_features.append(features)
    return all_features

start_time = time.time()

dir = '/home/pkrush/find-parts-faster-data/screws/2/'
contours = list(get_contours(dir))
count_of_good_contours = len(contours)
dir = '/home/pkrush/find-parts-faster-data/screws/3/'
contours.extend(get_contours(dir))
labels = np.zeros(len(contours))
labels[0:count_of_good_contours] = 1
print 'Done in %s seconds' % (time.time() - start_time,)

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
















