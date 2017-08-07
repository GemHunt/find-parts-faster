import time
import cv2

cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)

count = 0
for x in range(0,400000):
    start_time = time.time()
    ret, frame = cap.read()

    if frame == None:
        continue
    cv2.imshow('frame', frame)

    #grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    else:
        if key == -1:
            continue
        cv2.imwrite('/home/pkrush/find-parts-faster-data/screws/0/' + str(count) + '.png',frame)
        count += 1
        print count
    print 'In %s seconds' % (time.time() - start_time,)

cap.release()
cv2.destroyAllWindows()