import statsFuncs.trigonometry as trig
import utils
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

"""Finds blob in images.

Args:
    centers (array): array with centers of tags found in frame
    frames_path (str): filepath to the target directory that will contain the extract frames
    total_frames (int): total number of frames in videos
"""


def find_blob(frame_impath, correct_ids, angle, draw=False, verbose=False, progress_bar=True, show=False):

    frame, tag_ids = utils.detect_tags(frame_impath, progress_bar)

    # check if at least the number of ids was found:
    if len(frame[0]) < len(correct_ids):
        center_blob = 'Error'
        top_left_blob = 'Error'
        bottom_right_blob = 'Error'
        return center_blob, top_left_blob, bottom_right_blob

    marker_ids = utils.attribute(frame, 'id')
    if verbose:
        print('Total number of markers: ' + str(len(marker_ids)))
    good_frames = 0
    # filter markers that we don't have in the video
    if not len(frame[0]) == len(correct_ids):

        # find the wrong ones
        for tag_idx in range(len(frame[0])):
            if not frame[0][tag_idx]['id'] in correct_ids:
                frame_cleaned = [ii for ii in frame[0] if not (ii['id'] == frame[0][tag_idx]['id'])]

    if 'frame_cleaned' not in locals():  # if there wasn't any mistaken tag
        frame_cleaned = frame
        good_frames = good_frames + 1

    if (good_frames == 1) & (verbose is True):
        print('all correct ids detected')

    # attains the centers of each QR code of the frame; not sorted in any order
    centers = utils.attribute(frame_cleaned, 'centroid')
    # attains the corners for each of the QR codes in the frame; not sorted in any order
    corners = utils.attribute(frame_cleaned, 'verts')

    center_blob, top_left_blob, bottom_right_blob, screen_center = find_blob_coordinates(frame_impath, centers, corners, angle, draw,
                                                                          verbose, show)

    return center_blob, top_left_blob, bottom_right_blob, screen_center


def find_blob_coordinates(frame_impath, centers, corners, angle, draw, verbose, show):
    centroid1_hoz = centers[1]
    centroid2_hoz = centers[6]
    centroid1_vert = trig.midpoint(centers[9], centers[8])
    centroid2_vert = trig.midpoint(centers[3], centers[4])

    L1 = trig.line(centroid1_hoz, centroid2_hoz)
    L2 = trig.line(centroid1_vert, centroid2_vert)

    R = trig.intersection(L1, L2)

    if verbose:
        if R:
            print("Intersection detected:", R)
        else:
            print("No single intersection point detected")

    intersection_point = (int(R[0]), int(R[1]))

    start_point_hoz = (int(centroid1_hoz[0]), int(centroid1_hoz[1]))
    end_point_hoz = (int(centroid2_hoz[0]), int(centroid2_hoz[1]))
    start_point_vert = (int(centroid1_vert[0]), int(centroid1_vert[1]))
    end_point_vert = (int(centroid2_vert[0]), int(centroid2_vert[1]))

    # parameters:
    radius_circle = 65
    size_rectangle_blob = 80

    center_blob = trig.point_in_circle(angle, intersection_point, radius_circle)
    top_left_blob, bottom_right_blob = trig.rect_corners_from_center(center_blob, size_rectangle_blob,
                                                                     size_rectangle_blob)

    if draw:
        draw_blob(frame_impath, corners, start_point_hoz, end_point_hoz, start_point_vert, end_point_vert,
                  intersection_point,
                  radius_circle, top_left_blob, bottom_right_blob, verbose, show)

    return center_blob, top_left_blob, bottom_right_blob, intersection_point


def draw_blob(frame_impath, corners, start_point_hoz, end_point_hoz, start_point_vert, end_point_vert, intersection,
              radius_circle, top_left, bottom_right, verbose, show):
    # colors:
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)

    # parameters:
    thickness = 2

    image = cv2.imread(frame_impath)

    image = utils.make_mask_image(image, corners)
    image = cv2.line(image, start_point_hoz, end_point_hoz, red, thickness)  # horizontal line
    image = cv2.line(image, start_point_vert, end_point_vert, red, thickness)  # vertical line
    image = cv2.circle(image, intersection, 2, green, 8)  # point in the center of the screen
    image = cv2.circle(image, intersection, radius_circle, blue, 2)  # draw circle

    # draw rectangle around blob
    image = cv2.rectangle(image, top_left, bottom_right, blue, thickness)
    #draw circle using rectangle coordinates
    #center of rectange
    rectangle_center = (int((top_left[0] + bottom_right[0])/2), int((top_left[1] + bottom_right[1])/2))
    #diameter of circle
    circle_radius = int((-top_left[0] + bottom_right[0])/2)
    image = cv2.circle(image, rectangle_center, circle_radius, green)
    if show:
        plt.figure(figsize=(20, 20))
        plt.imshow(image)

    path = frame_impath[0:frame_impath.find('frame')]
    frame_name = frame_impath[frame_impath.find('frames/') + 7:]

    path_to_save = path + 'processed/'

    try:
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
            if verbose:
                print("Successfully created the directory %s " % path_to_save)
        else:
            if verbose:
                print("Directory already exists %s " % path_to_save)
    except OSError:
        print("Creation of the directory %s failed" % path_to_save)

    try:
        if verbose:
            print(path_to_save + 'processed_' + frame_name)
        cv2.imwrite(path_to_save + 'processed_' + frame_name, image)
    except OSError:
        print('couldnt save file, check paths and editing permission')
