import numpy as np
import os
import random
import subprocess
import sys
import time
import zbar

import cv2
import pandas as pd

import create_lmdb
import label
import local_dir as dir

sys.path.append('/home/pkrush/caffe/python')
import caffe

def get_classifier(crop_size):
    MODEL_FILE = dir.model + 'deploy.prototxt'
    PRETRAINED = dir.model + 'snapshot.caffemodel'
    meanFile = dir.model + 'mean.binaryproto'

    # Open mean.binaryproto file
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(meanFile, 'rb').read()
    blob.ParseFromString(data)
    mean_arr = np.array(caffe.io.blobproto_to_array(blob)).reshape(1, crop_size, crop_size)
    print mean_arr.shape

    net = caffe.Classifier(MODEL_FILE, PRETRAINED, image_dims=(crop_size, crop_size), mean=mean_arr, raw_scale=255)
    return net

def get_labels(model_name):
    labels_file = dir.model + 'labels.txt'
    labels = [line.rstrip('\n') for line in open(labels_file)]
    return labels

def get_caffe_image(crop, crop_size):
    # this is how you get the image from file:
    # coinImage = [caffe.io.load_image("some file", color=False)]

    caffe_image = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    caffe_image = caffe_image.astype(np.float32) / 255
    caffe_image = np.array(caffe_image).reshape(crop_size, crop_size, 1)
    # Caffe wants a list so []:
    return [caffe_image]

def get_composite_image(images, rows, cols):
    crop_rows, crop_cols, channels = images[0].shape
    composite_rows = crop_rows * rows
    composite_cols = crop_cols * cols
    composite_image = np.zeros((composite_rows, composite_cols, 3), np.uint8)
    key = 0
    for x in range(0, rows):
        for y in range(0, cols):
            key += 1
            if len(images) <= key:
                break
            if images[key] is not None:
                composite_image[x * crop_rows:((x + 1) * crop_rows), y * crop_cols:((y + 1) * crop_cols)] = images[key]
    return composite_image

def infer_dir(dir_to_infer):
    start_time = time.time()
    crop_size = 28
    model = get_classifier(crop_size)
    model_labels = get_labels('model')
    count = 0

    scores = {}
    for root, dirnames, walk_filenames in os.walk(dir_to_infer):
        for filename in walk_filenames:
            point = filename[7:17]

            crop = cv2.imread(root + '/' + filename)
            if crop is None:
                continue
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            score = model.predict(get_caffe_image(crop, crop_size), oversample=False)
            if point in scores.iterkeys():
                scores[point] = scores[point] + score
            else:
                scores[point] = score

            coin_type = model_labels[np.argmax(score)]
            max_value = np.amax(score)
            count +=1
            if count % 400 == 0 and count != 0:
                for key, score in scores.iteritems():
                    print score[0][0]/(score[0][0] +score[0][1])
                print "Next"

    #print 'Done in %s seconds' % (time.time() - start_time,)

def create_heat_map(filename):
    start_time = time.time()
    crop_radius = 21
    model = get_classifier(28)
    model_labels = get_labels('model')
    count = 0
    scores = {}

    test_image = cv2.imread(filename)
    cols = test_image.shape[0]
    rows = test_image.shape[1]

    heatmap = np.zeros((cols,rows), dtype=np.uint8)

    for x in range(0, rows - crop_radius * 2, 9):
        for y in range(0, cols - crop_radius * 2, 9):
            test_crop = test_image[y:y+(crop_radius * 2),x:x+(crop_radius * 2)]
            crop0 = cv2.cvtColor(test_crop, cv2.COLOR_BGR2GRAY)
            crop = crop0.copy()
            angle = random.random() * 360
            m = cv2.getRotationMatrix2D((crop_radius, crop_radius), angle, 1)
            cv2.warpAffine(crop, m, (crop_radius * 2, crop_radius * 2), crop, cv2.INTER_CUBIC)

            crop = cv2.resize(crop, (28,28), interpolation=cv2.INTER_AREA)
            score = model.predict(get_caffe_image(crop, 28), oversample=False)
            heatmap[y+crop_radius,x+crop_radius] = int(score[0][0] * 255)
        print x,y

    cv2.imwrite(dir.data + 'heatmap.png',heatmap)
    #print 'Done in %s seconds' % (time.time() - start_time,)


def get_heat_map(filename, step_size):
    start_time = time.time()
    crop_radius = 21
    cols = label.output_height
    rows = label.output_width
    heatmap = np.zeros((int(cols / step_size), int(rows / step_size)), dtype=np.uint8)
    heatmap_rows, heatmap_cols = heatmap.shape

    df = pd.read_csv(filename)
    for index, row in df.iterrows():
        x = int(row[0])
        y = int(row[1])
        score = row[3]
        if score < .5:
            continue
        if row[2] == 'no':
            score = 1 - score
        heatmap_x = int((x + crop_radius) / step_size)
        heatmap_y = int((y + crop_radius) / step_size)

        # print heatmap_x, heatmap_y

        if heatmap_cols > heatmap_x and heatmap_rows > heatmap_y:
            heatmap[heatmap_y, heatmap_x] = int(score * 255)

    full_heatmap = cv2.resize(heatmap, (rows, cols), interpolation=cv2.INTER_LANCZOS4)
    return full_heatmap


def get_next_image_from_phone():
    # base_dir = '/run/user/1000/gvfs/'
    # for mtp_dir in os.listdir(base_dir):
    #     if mtp_dir.startswith('mtp:host'):
    #         base_dir = base_dir + mtp_dir + '/Phone/DCIM/simple_interval_camera/'
    #     else:
    #         print '/run/user/1000/gvfs/mtp:host  not found'
    #         sys.exit()

    for root, dirnames, filenames in os.walk(dir.capture):
        filenames.sort()
        for filename in filenames:
            if filename.endswith('.jpg'):
                full_filename = root + '/' + filename
                next_image = cv2.imread(full_filename,)
                if next_image is None:
                    #Wait for the file to download:
                    cv2.waitKey(200)
                else:
                    #print "Next Image Captured"
                    os.remove(full_filename)
                    return next_image
    return None


def infer_from_camera():
    step_size = 8
    scanner = zbar.ImageScanner()
    scanner.parse_config('enable')
    loop = True
    heat_map = None
    #background = label.get_background(True)
    #cv2.imshow("Background", background)
    #cv2.waitKey(1)
    start_time = time.time()
    window_x = 1670
     = 30
    window_y_spacing = label.output_height + 30

    while loop:
        image_to_window = get_next_image_from_phone()

        if image_to_window != None:
            display_image = cv2.resize(image_to_window,(label.output_width,label.output_height), interpolation=cv2.INTER_AREA)
            cv2.imshow("image_to_window", display_image)
            cv2.moveWindow("image_to_window",window_x,window_y)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            warped = label.get_warped(image_to_window, scanner)
            if warped == None:
                print 'QR codes not found on capture.'
            else:
                cv2.imshow("warped", warped)
                cv2.moveWindow("warped", window_x, window_y + window_y_spacing)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                create_lmdb.create_lmdb(warped, dir.train, 0, step_size)
                subprocess.call(dir.data + 'infer.sh')
                heat_map = get_heat_map(dir.train + '0.dat', step_size)

            if heat_map != None:
                cv2.imshow("heat_map", heat_map)
                cv2.moveWindow("heat_map", window_x,  window_y + (window_y_spacing * 2))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                print 'Done in %s seconds' % (time.time() - start_time,)
                start_time = time.time()




#start_time = time.time()
#infer_dir(dir.no)
# print 'Done in %s seconds' % (time.time() - start_time,)

infer_from_camera()
