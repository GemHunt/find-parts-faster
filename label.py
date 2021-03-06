import Image
import glob
import numpy as np
import os
import pickle
import random
import serial
import time
import zbar

import cv2
import pyqrcode

import local_dir as dir

#output_width = 480
#output_height = 270
output_width = 640
output_height = 360
background_width = 640
background_height = 360
padding = 20
qr_scale = 6
qr_unscaled_size = 21  # version 1
qr_size = qr_unscaled_size * qr_scale
qr_padded_size = qr_unscaled_size * qr_scale + padding * 2
x_qr_interval = background_width - (padding * 2) - qr_size
y_qr_interval = background_height - (padding * 2) - qr_size
yes_points = []
mouse_pointer = (100,100)
crop_radius = 21
augmentation_factor = 10
last_symbols = None


def get_qr_code(data,scale):
    qr = pyqrcode.create(data, error='H', version=1, mode='binary')
    qr_code = np.array(qr.code, dtype=np.uint8)
    qr_code[qr_code == 0] = 255
    qr_code[qr_code == 1] = 0
    qr_code_size = qr_code.shape[0] * scale
    qr_code = cv2.resize(qr_code, (qr_code_size, qr_code_size), interpolation=cv2.INTER_AREA)
    return qr_code

def mark(event, x, y, flags, param):
    global yes_points
    global mouse_pointer

    if event == cv2.EVENT_LBUTTONDOWN:
        yes_points.append((x, y))

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pointer = (x,y)

def get_points_inside_border(points):
    new_points = []
    for x,y in points:
        if is_point_inside_border(x,y):
            new_points.append((x, y))
    return new_points

def is_point_inside_border(x,y):
    if x < crop_radius:
        return False
    if x > output_width - crop_radius:
        return False
    if y < crop_radius:
        return False
    if y > output_height - crop_radius:
        return False
    return True

def save_crops(warped, points,filename_start):
    for count in range(0, augmentation_factor):
        for x, y in points:
            pixels_to_jitter = 4  # Old Way
            jitter_x = x + (random.random() * pixels_to_jitter * 2) - pixels_to_jitter
            jitter_y = y + (random.random() * pixels_to_jitter * 2) - pixels_to_jitter
            if is_point_inside_border(jitter_x, jitter_y):
                crop0 = warped[jitter_y - crop_radius:jitter_y + crop_radius,
                        jitter_x - crop_radius:jitter_x + crop_radius]
                crop = crop0.copy()

                mirror = random.random()
                if mirror > .5:
                    mirrored_crop = np.zeros((crop_radius * 2, crop_radius * 2), dtype=np.uint8)
                    cv2.flip(crop, 1, mirrored_crop)
                    crop = mirrored_crop


                angle = random.random() * 360
                m = cv2.getRotationMatrix2D((crop_radius, crop_radius), angle, 1)
                cv2.warpAffine(crop, m, (crop_radius * 2, crop_radius * 2), crop, cv2.INTER_CUBIC)
                crop_filename = filename_start + 'id' + str(jitter_x).zfill(4) + 'x' + str(jitter_y).zfill(
                    4) + 'y' + '.png'

                cv2.imwrite(crop_filename, crop)


def get_background(backlight):
    background = np.zeros((background_height,background_width), dtype=np.uint8)
    background = background + 255

    for x in range(0,2):
        for y in range(0, 2):
            data = str(x) + ',' + str(y)
            qr_code = get_qr_code(data, qr_scale)
            qr_x = x * x_qr_interval + padding
            qr_y = y * y_qr_interval + padding
            background[qr_y:qr_y + qr_size,qr_x:qr_x + qr_size] = qr_code

    if not backlight:
        background[0:background_height, qr_padded_size:background_width-qr_padded_size] = 0
        background[qr_padded_size:background_height - qr_padded_size, 0:background_width] = 0

    return background


def get_warped(input, scanner):
    global last_symbols

    if last_symbols is None:
        gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY, dstCn=0)
        pil = Image.fromarray(gray)
        width, height = pil.size
        raw = pil.tostring()

        # create a reader
        image = zbar.Image(width, height, 'Y800', raw)
        scanner.scan(image)

        print"                                             for symbol in image:" + str(len(image.symbols))

        if len(image.symbols) == 4:
            symbols = image.symbols
            last_symbols = image.symbols
        else:
            return None
    else:
        symbols = last_symbols


    max_src = np.zeros((4, 2), dtype=np.float32)
    max_dst = np.zeros((4, 2), dtype=np.float32)
    for symbol in symbols:
        # do something useful with results
        if symbol.data in ["0,0", "1,0", "0,1", "1,1"]:
            # print 'decoded', symbol.type, 'symbol', '"%s"' % symbol.data
            loc = symbol.location
            #print symbol.data, loc
            # cv2.line(output, loc[0], loc[1], (0, 0, 0))
            # cv2.line(output, loc[1], loc[2], (0, 0, 0))
            # cv2.line(output, loc[2], loc[3], (0, 0, 0))
            # cv2.line(output, loc[3], loc[0], (0, 0, 0))
            src = np.array(loc, dtype=np.float32)
            rad = 3
            # dst = np.float32(((240+rad,240-rad), (240+rad, 320-rad), (320+rad,320-rad), (320+rad,240-rad)))

            offset_x = 0
            offset_y = 0

            if symbol.data == "0,0":
                max_src[0] = loc[0]
                qr_code_x = 0
                qr_code_y = 0

            if symbol.data == "0,1":
                max_src[1] = loc[1]
                qr_code_x = 0
                qr_code_y = 1

            if symbol.data == "1,0":
                max_src[3] = loc[3]
                qr_code_x = 1
                qr_code_y = 0

            if symbol.data == "1,1":
                max_src[2] = loc[2]
                qr_code_x = 1
                qr_code_y = 1

            local_offset_x = offset_x + x_qr_interval * qr_code_x
            local_offset_y = offset_y + y_qr_interval * qr_code_y

            local_dst = np.zeros((4, 2), dtype=np.float32)
            local_dst[0] = [local_offset_x, local_offset_y]
            local_dst[1] = [local_offset_x, local_offset_y + qr_size]
            local_dst[2] = [local_offset_x + qr_size, local_offset_y + qr_size]
            local_dst[3] = [local_offset_x + qr_size, local_offset_y]

        max_dst[0] = [offset_x, offset_y]
        max_dst[1] = [offset_x, y_qr_interval + qr_size]
        max_dst[2] = [x_qr_interval + qr_size, y_qr_interval + qr_size]
        max_dst[3] = [x_qr_interval + qr_size, offset_y]

    M = cv2.getPerspectiveTransform(max_src, max_dst)
    warped = cv2.warpPerspective(input, M,
                                 (background_width - (padding * 2), background_height - (padding * 2)))
    warped = cv2.resize(warped, (output_width, output_height), interpolation=cv2.INTER_AREA)
    return warped


def process_video():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(str(dir.data + "0.mp4"))
    # cap.set(3,1920)
    # cap.set(4,1080)

    scanner = zbar.ImageScanner()
    scanner.parse_config('enable')

    for count in range(0, 400000):
        # Capture frame-by-frame
        start_time = time.time()
        ret, frame = cap.read()

        if frame == None:
            break
        # cv2.imwrite(str(x) + '.png', frame)

        output = frame.copy()

        warped = get_warped(output, scanner)
        cv2.imwrite(dir.warped + str(count).zfill(5) + '.png', warped)
        cv2.imshow("warped", warped)

        # output = cv2.resize(output, (output_width, output_height), interpolation=cv2.INTER_AREA)
        #cv2.imshow("Camera", output)
        print '4 in %s seconds' % (time.time() - start_time,)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #print 'In %s seconds' % (time.time() - start_time,)

    cap.release()
    cv2.destroyAllWindows()


def rotate_led(led_on):
    if led_on:
        ser = serial.Serial(port='/dev/ttyUSB1', baudrate=115200)
    cv2.waitKey(1)
    while True:
        for backlight_on in [True, False]:
            background = get_background(backlight_on)
            for count in range(0, 48):
                cv2.imshow("Background", background)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if led_on:
                    ser.write(str(count) + "\n")
                cv2.waitKey(66)

def label_warped():
    cv2.namedWindow("Warped")
    cv2.setMouseCallback("Warped", mark)
    global yes_points

    if os.path.exists(dir.data + 'yes_points.pickle'):
        yes_points = pickle.load(open(dir.data + 'yes_points.pickle', "rb"))

    warped_filenames = []
    for filename in glob.iglob(dir.data + 'warped/' + '*.png'):
        #crops.append([random.random(), filename])
        warped_filenames.append(filename)
    warped_filenames.sort()
    loop = True
    while loop:
        for filename in warped_filenames:
            yes_points = get_points_inside_border(yes_points )

            warped = cv2.imread(filename)
            cv2.circle(warped,mouse_pointer, crop_radius, (0,255,0), 1)

            for marked_point in yes_points:
                cv2.circle(warped,marked_point, crop_radius, (255, 255,255), 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(warped, 'Yes:' + str(len(yes_points)), (4, 20), font, .7, (0, 255, 0), 2)
            cv2.imshow("Warped", warped)
            if cv2.waitKey(33) & 0xFF == ord('q'):
                loop = False
                break
    pickle.dump(yes_points, open(dir.data + 'yes_points.pickle', "wb"))

def fill_train_dir():
    global yes_points

    yes_points = pickle.load(open(dir.data + 'yes_points.pickle', "rb"))

    warped_filenames = []
    for filename in glob.iglob(dir.warped + '*.png'):
        #crops.append([random.random(), filename])
        warped_filenames.append(filename)
        warped_filenames.sort()

    for filename in warped_filenames:
        warped = cv2.imread(filename)
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped_id = str(filename).replace('.png','')
        warped_id = warped_id[(len(warped_id)-5):] #last 5 is ID
        filename_start = dir.yes +  warped_id
        save_crops(warped, yes_points,filename_start)
        filename_start = dir.no + warped_id
        random_no_points = []
        while len(random_no_points) < len(yes_points):
            output_width = 480
            output_height = 270
            x = random.randrange(crop_radius, output_width - crop_radius)
            y = random.randrange(crop_radius, output_height - crop_radius)
            too_close = False
            for yes_x, yes_y in yes_points:
                if (abs(x - yes_x) < crop_radius - 7) and (abs(y - yes_y) < crop_radius - 7):
                    too_close = True
            if not too_close:
                random_no_points.append((x, y))
                # todo if not in the QR code area
        save_crops(warped, random_no_points, filename_start)


def show_background(backlight):
    background = get_background(backlight)
    while True:
        cv2.imshow("Background", background)
        cv2.moveWindow("Background",210,30)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

#show_background(True)

# dir.init_directories()
#rotate_led(False)
#process_video()
#label_warped()
#fill_train_dir()

# 4:Tray: No LED lighting.
# 5:Tray: No LED lighting: Flashlight (blurry, needs top light)
# 6:
# 7:

