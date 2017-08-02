import cv2
import numpy as np
import time
import sys
import os


def getContours(min_area, max_area, filename):
    im = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    size_factor = 1
    small = cv2.resize(im,(int(1920 *size_factor) ,int(1080 * size_factor)))
    img = small.copy()
    cv2.threshold(small,120,255,0,small)
    small = 255 - small

    contours, hierarchy = cv2.findContours(small,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    count = 0
    labels = np.zeros(len(contours))
    all_features = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 1100 * size_factor * size_factor < area < 1500 * size_factor * size_factor and hierarchy[0][count][3] == -1:
            labels[count] = 1
            features = np.zeros(13)
            filtered_contours.append(cnt)
            perimeter = cv2.arcLength(cnt,True)
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            bounding_area = x * y
            hull = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull)
            if defects is not None:
                hull_defect_count = len(cv2.convexityDefects(cnt, hull))
            features[0:6] = [area, perimeter, ma, MA, bounding_area, hull_defect_count]

            mom = cv2.moments(cnt)
            hu = cv2.HuMoments(mom)
            hu = hu.flatten()
            features[6:13] = hu

            print count, x,y,angle,area,perimeter,ma,MA,bounding_area,hull_defect_count
            #print hu

            all_features.append(features)

        #This is really slow to get an average mean:
        #mask = np.zeros(img.shape, np.uint8)
        #cv2.drawContours(mask, [cnt], 0, 255, -1)
        #pixelpoints = np.transpose(np.nonzero(mask))
        #mean_val = cv2.mean(img, mask=mask)
        #print mean_val

        count +=1
    cv2.drawContours(img,filtered_contours,-1,255,1)
    print len(all_features)
    cv2.imshow("Contours",img)
    cv2.waitKey(0)
    return filtered_contours

start_time = time.time()
dir = '/home/pkrush/find-parts-faster-data/screws/4/'

os.listdir("somedirectory")

filename = '/home/pkrush/find-parts-faster-data/screws/4/0.png'
getContours(1100,1500,filename)

sys.exit()



from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(all_features,labels)
count = 0

print 'Done in %s seconds' % (time.time() - start_time,)

for features in all_features:
    X = [features]
    y = labels[count]
    predicted_class = logistic.predict(X)
    print logistic.predict_proba(X),
    #print logistic.decision_function(X),
    if predicted_class <> y:
        print
        print
        #print logistic.decision_function(X),
        print logistic.predict_proba(X),
        print 'Predicted class %s, real class %s' % ( logistic.predict(X),y)
    count +=1
print
print(logistic.coef_)

# Those values, however, will show that the second parameter
# is more influential
print(np.std(X, 0)*logistic.coef_)
print 'Done in %s seconds' % (time.time() - start_time,)





