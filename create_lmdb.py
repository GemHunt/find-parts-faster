import os
import random
import shutil
import time

import cv2
import lmdb
from caffe.proto import caffe_pb2

import caffe_lmdb


def create_lmdb(image_to_window, lmdb_dir, label, step_size):
    start_time = time.time()
    crop_radius = 21

    if os.path.exists(lmdb_dir):
        shutil.rmtree(lmdb_dir)

    if not os.path.exists(lmdb_dir):
        os.makedirs(lmdb_dir)

    # create DBs
    train_image_db = lmdb.open(os.path.join(lmdb_dir, 'train_db'), map_async=True, max_dbs=0)

    # arrays for image and label batch writing
    train_image_batch = []

    cols = image_to_window.shape[0]
    rows = image_to_window.shape[1]

    for x in range(0, rows - crop_radius * 2, step_size):
        for y in range(0, cols - crop_radius * 2, step_size):
            test_crop = image_to_window[y:y + (crop_radius * 2), x:x + (crop_radius * 2)]
            crop0 = cv2.cvtColor(test_crop, cv2.COLOR_BGR2GRAY)
            crop = crop0.copy()
            angle = random.random() * 360
            m = cv2.getRotationMatrix2D((crop_radius, crop_radius), angle, 1)
            cv2.warpAffine(crop, m, (crop_radius * 2, crop_radius * 2), crop, cv2.INTER_CUBIC)
            crop = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)

            datum = caffe_pb2.Datum()
            datum.data = cv2.imencode('.png', crop)[1].tostring()
            datum.label = label
            datum.encoded = 1

            str_id = '{:04}'.format(x) + ',' + '{:04}'.format(y)
            train_image_batch.append([str_id.encode('ascii'), datum])

    if len(train_image_batch) > 0:
        caffe_lmdb.write_batch_to_lmdb(train_image_db, train_image_batch)

    train_image_db.close()

    #print lmdb_dir, 'Done after %s seconds' % (time.time() - start_time,)
    return
