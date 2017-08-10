import time
import cv2
import part_image
import numpy as np
import sys

def get_templates(frame,template_contour):
    templates = {}
    x, y, template_width, template_height = cv2.boundingRect(template_contour)
    template_main = frame[y:y + template_height, x:x + template_width]
    template_main = 255 - template_main.copy()

    for angle in range(0,360):
        template = part_image.rotate_bound(template_main.copy(), angle)

        #make it small as possible:
        local_contours, hierarchy = cv2.findContours(template, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(local_contours) == 0:

            cv2.imshow('template', template)
            key = cv2.waitKey(0)
            if key & 0xFF == ord('q'):
                sys.exit()

            print angle, "No local contours found!"
            return templates
            #sys.exit()
        max_area = 0
        if len(local_contours) == 1:
            local_contour = local_contours[0]
        else:
            for cnt in local_contours:
                area = cv2.contourArea(cnt)
                if max_area > cv2.contourArea(cnt):
                    max_area = area
                    local_contour = local_contours[0]
        if max_area == 0:
            print max_area
            cv2.imshow('template', template)
            key = cv2.waitKey(0)
            if key & 0xFF == ord('q'):
                sys.exit()

        x, y, img_to_search_width, img_to_search_height = cv2.boundingRect(local_contour)
        template = template[y:y + img_to_search_height, x:x + img_to_search_width]

        template = template * .5 + 127
        template = np.uint8(template)
        template = 255 - template
        templates[angle] = template
    return templates

def break_contours(local_thresh,max_area,templates):
    broken_contours = []

    max_max_val = 0
    max_angle = 0
    max_template = []
    for angle,template in templates.iteritems():
        template_height,template_width  = template.shape
        #template_area = template_height * template_width
        local_thresh_height, local_thresh_width = local_thresh.shape
        local_thresh_area  = local_thresh_height * local_thresh_width

        #local_thresh = part_image.add_border(local_thresh.copy(), width=3, color=255)
        if template_height > local_thresh_height or template_width > local_thresh_width:
            continue

        method = eval('cv2.TM_CCOEFF')

        res = cv2.matchTemplate(local_thresh, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        #top_left = max_loc
        #bottom_right = (top_left[0] + w, top_left[1] + h)
        # cv2.rectangle(img, top_left, bottom_right, 128, 2)

        if max_max_val < max_val:
            max_max_val = max_val
            max_angle = angle
            max_template = template.copy()

    if max_max_val <> 0:
        print max_angle,max_max_val

    return broken_contours