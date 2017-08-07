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
    return cv2.resize(img,(1280,972))

def get_thresh(img,flip):
    cv2.threshold(img, 100, 255, 0, img)
    if flip:
        img = 255 - img
    return img

def get_contours(img):
    min_area = 910
    max_area = 1240
    max_area = 1800
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
        count +=1
    #cv2.drawContours(img,filtered_contours,-1,255,1)
    #cv2.imshow("Contours",img)
    #cv2.waitKey(0)
    return contours_filtered_in,contours_filtered_out

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
dir = '/home/pkrush/find-parts-faster-data/screws/0/'
contours = []
for filename in os.listdir(dir):
    im = cv2.imread(dir + filename, cv2.IMREAD_GRAYSCALE)
    im = get_crop(im)

    # im = get_thresh(im,True)
    # kernel = np.ones((3, 3), np.uint8)
    # im = cv2.erode(im, kernel, iterations=)
    # #im = cv2.dilate(im, kernel, iterations=1)

    # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    contours_filtered_in, contours_filtered_out = get_contours(im)
    contours.extend(contours_filtered_in)
count_of_good_contours = len(contours)
dir = '/home/pkrush/find-parts-faster-data/screws/1/'
for filename in os.listdir(dir):
    im = cv2.imread(dir + filename, cv2.IMREAD_GRAYSCALE)
    im = get_crop(im)
    contours_filtered_in, contours_filtered_out = get_contours(im)
    contours.extend(contours_filtered_in)
labels = np.zeros(len(contours))
labels[0:count_of_good_contours] = 1

count = 0

all_features = get_features(contours)

names = ["Nearest Neighbors", "Gaussian Process",
         "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes","Decision Tree", "RBF SVM","QDA"]

classifiers = [
    KNeighborsClassifier(3),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    DecisionTreeClassifier(max_depth=5),
    SVC(gamma=2, C=1),
    QuadraticDiscriminantAnalysis()]



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
















