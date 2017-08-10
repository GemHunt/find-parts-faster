import time
import cv2
import part_image
import numpy as np

def get_template(img,angle):
        template = 255- img.copy()
        template = part_image.rotate_bound(template.copy(), angle)
        template = template * .5 + 127
        template = np.uint8(template)
        template = 255 - template
        return template

def break_contours(local_thresh,max_area,template):
    broken_contours = []
    rect = cv2.minAreaRect(template_contour)
    angle = rect[2]

    x, y, template_width, template_height = cv2.boundingRect(template_contour)
    template_main = frame[y:y + template_height,x:x+template_width]
    templates = {}


    for angle in range(0,360,360):
        template = 255- template_main.copy()
        template = part_image.rotate_bound(template.copy(), angle)
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

                img = part_image.add_border(img_to_search, width=15, color=255)
                if template_width > img_to_search_width or template_height > img_to_search_height:
                    continue

                method = eval('cv2.TM_CCOEFF')

                res = cv2.matchTemplate(img, template, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                # cv2.rectangle(img, top_left, bottom_right, 128, 2)

                if max_max_val < max_val:
                    max_max_val = max_val
                    max_angle = angle
                    max_template = template.copy()

        if max_max_val <> 0:
            print max_angle,max_max_val

    return broken_contours