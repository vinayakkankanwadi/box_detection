import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy
import os
import sys
import json
import argparse


def detect_threshold(image):
    # detect the binarization threshold
    image_size = image.size
    y, x, _ = plt.hist(image.ravel(), 256, [0, 256])
    # sort grayscale by number of pixels that belong to that scale
    sorted_pixel = sorted([(x, i) for (i, x) in enumerate(y)], reverse=True)
    for i in range(len(sorted_pixel))[1:]:
        if sorted_pixel[i][0]/sorted_pixel[i-1][0] < 0.5 and sorted_pixel[i][0]/image_size < 0.05:
            # Pick the value as threshold which: making up less than 5% of total number of pixels
            # and 50% of total number of pixels less than its predecessor
            return min(list(zip(*sorted_pixel[:i]))[1])-10


def find_all_forms(input_image, flag=8):
    # find form by finding connected components
    # a image may contain multiple forms
    # every time we find a form we add the form the the output image
    image_size = input_image.size
    form_locations = []  # location of forms
    sum_form_area = 0  # summation of form areas
    find_form = True  # indicator: True if we find the form
    out_image = input_image[:] - input_image[:]  # blank image (all black)
    while sum_form_area < image_size*0.8 and find_form:
        # we continue to find next form if the total area covered by form is less than 80% of the original image
        # and we find a new form in the last round of the loop
        new_form, form_locations, sum_form_area, find_form = search_new_form(input_image, flag, form_locations, sum_form_area)
        if find_form:
            out_image = out_image+new_form
        else:
            break
    return out_image


def search_new_form(input_image, flag, form_locations, sum_form_area):
    exist_form = []
    largest_blob = [0, 0, 0, 0]
    largest_blob_size = 0
    find_form = False
    new_form = None
    process_image = input_image.copy()
    rows, cols = process_image.shape
    if len(form_locations) != 0:
        # determine the top-left and bottom right coordinate of the exist forms
        for f in form_locations:
            exist_form.append([f[0], f[1], f[0]+f[2], f[1]+f[3]])
    for i in range(cols):
        for j in range(rows):
            already_in_form = False  # indicator: True if the pixel is within exist form areas
            if len(exist_form) != 0:
                for f in exist_form:
                    # if the pixel within the exist form areas we ignore the pixel
                    if f[0] <= i <= f[2] and f[1] <= j <= f[3]:
                        already_in_form = True
            if not already_in_form and process_image[j][i] != 0:
                # process the pixel to find the connect component of it
                before_process_image = process_image.copy()
                mask = np.zeros((rows+2, cols+2), np.uint8)
                # floodFill form edge with black
                curr_blob = cv2.floodFill(process_image, mask, (i, j), 0, 200, 255, flag)
                curr_blob_size = curr_blob[3][2] * curr_blob[3][3]
                if largest_blob_size < curr_blob_size:
                    # find and return the largest blob(new form) in the image(except the area that already covered by exist forms)
                    largest_blob_size = curr_blob_size
                    largest_blob = curr_blob
                    # (image before flood fill edge) - (image after flood fill)  = the new form edge
                    new_form = before_process_image - process_image
                    find_form = True  # indicator: True if we find a new form

    if find_form:
        form_locations.append(largest_blob[3])    # add found form location
        sum_form_area += largest_blob_size    # update total area covered by found forms

    return new_form, form_locations, sum_form_area, find_form


def flood_fill_margin(image, flag=4):
    rows, cols = image.shape
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    for i in range(cols):
        cv2.floodFill(image, mask, (i, 0), 200, 0, 50, flag)
        cv2.floodFill(image, mask, (i, rows-1), 200, 0, 50, flag)
    for j in range(rows):
        cv2.floodFill(image, mask, (0, j), 200, 0, 50, flag)
        cv2.floodFill(image, mask, (cols-1, j), 200, 0, 50, flag)
    return image


def find_boxes(input_image, flag=4):
    # find boxes in forms
    boxes = []
    rows, cols = input_image.shape
    for i in range(cols):
        for j in range(rows):
            if input_image[j][i] == 0:
                mask = np.zeros((rows+2, cols+2), np.uint8)
                # find box and then floodFill the box area to prevent repetition
                bounding_box = cv2.floodFill(input_image, mask, (i, j), 200, 0, 50, flag)[3]
                boxes.append(bounding_box)
    return boxes


def draw_boxes(input_image, boxes):
    out_image = input_image.copy()
    idx = 0
    # sort box by location of the top left corner
    sorted_boxes = sorted(sorted(boxes, key=lambda e: e[0]), key=lambda e: e[1])
    for box in sorted_boxes:
        # draw boxes
        cv2.rectangle(out_image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255, 255, 0), 2)
        # put box index in the center of each box
        cv2.putText(out_image, str(idx), (box[0]+int(box[2]/2), box[1]+int(box[3]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        idx += 1
    return out_image


def convert_boxes_to_json(boxes):
    box_dict = {"boxes": []}
    for box in boxes:
        tl = [box[0], box[1]]
        tr = [box[0]+box[2], box[1]]
        br = [box[0]+box[2], box[1]+box[3]]
        bl = [box[0], box[1]+box[3]]
        curr_box = {"points": [tl, tr, br, bl]}
        box_dict["boxes"].append(curr_box)
    return box_dict


def main():
    args = parse_args()

    for filename in os.listdir(args.input_dir):
        if filename[0] == '.':
            # skip hidden files
            continue

        try:
            print("Detecting boxes for {}{}".format(args.input_dir, filename))

            original_image = cv2.imread(args.input_dir+filename)
            input_image = original_image.copy()

            # find the binarization Threshold of the image
            binarization_threshold = detect_threshold(input_image)

            # convert image to grayscale
            h = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

            # Thresholding: convert grayscale to binary image
            _, b_image = cv2.threshold(h, binarization_threshold, 255, 1)

            # Pre-processing to determine form areas
            edges_highlight_image = find_all_forms(b_image)
            
            # use morphological operators to prevent unclosed boxes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            dilated_image = cv2.morphologyEx(edges_highlight_image, cv2.MORPH_CLOSE, kernel)

            # remove margin
            margin_removed_image = flood_fill_margin(dilated_image)

            # detect boxes in all forms
            boxes = find_boxes(margin_removed_image)

            # format output box location and Draw box on original image
            boxes_json = convert_boxes_to_json(boxes)
            boxes_highlight_image = draw_boxes(original_image, boxes)

            # write to output files
            filename = filename.split('.')[0]  # remove filename extension
            with open(args.output_dir+filename+'.json', 'w') as outfile:
                json.dump(boxes_json, outfile)

            cv2.imwrite(args.output_dir + filename + '.jpg', boxes_highlight_image)

        except:
            print("Encounter error during detecting boxes for {}{}".format(args.input_dir, filename))

def parse_args():
    parser = argparse.ArgumentParser(description='Python program for box detection')
    parser.add_argument('input_dir', type=str, help='input directory')
    parser.add_argument('output_dir', type=str, help='output directory')
    args = parser.parse_args()
    args.input_dir = args.input_dir.strip('/') + '/'
    args.output_dir = args.output_dir.strip('/') + '/'
    if not os.path.isdir(args.input_dir) or not os.path.isdir(args.output_dir):
        parser.error("input dir and output dir should already exist!")
    return args


if __name__ == "__main__":
    main()
