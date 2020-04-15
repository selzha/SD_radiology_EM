import sys
import time
import cv2
import numpy as np
from pupil_apriltags import Detector, Detection
from collections import defaultdict
import os
from typing import List, Dict, Tuple, Any

def print_progress_bar (iteration, total, prefix ='', suffix ='', decimals = 1, length = 100, fill ='█', printEnd ="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    # Print New Line on Complete
    if iteration == total:
        print()


def detect_tags(frames_path: str, aperture=11, visualize=False) -> Tuple[List[List[Dict[str, Any]]], Dict[int, int]]:
    """Detect all tags (Apriltags3) found in a folder of PNG files and return (1) a list of tag objects
    for preprocessing and (2) a dictionary containing the frequency that each tag ID appeared
    Args:n
        frames_path (str): path to the directory containing PNG images
        aperture (int):
        visualize (bool):
    Returns:
        frames (List[Dict[str, Any]]): list of objects containing id (int), centroid (np.array[int]) and corners (np.array[int])
        tag_ds (Dict[int, int]): dictionary mapping tag IDs to frequency of tag across all images
    """
    # Initialize variables
    frames = []
    tag_ids = defaultdict(int)
    at_detector = Detector()

    # Sort by index in.../frame<index>.png
    all_images = sorted(glob(f'{frames_path}/*.png'), key=lambda f: int(os.path.basename(f)[5:-4]))

    # Deleted last image after out of range error popped up
    # TODO: but not analyzing last 2 frames?
    # Feb 21: commented out deleting last frame, altho this was important when you have all frames in folder
    print(len(all_images))
    if len(all_images) > 1:
        all_images = all_images[:-1]

    num_images = len(all_images)
    print_progress_bar(0, num_images, prefix='Progress:', suffix='Complete', length=50)

    # Iterate thru all PNG images in frames_path
    for i, img_path in enumerate(all_images):
        # Create a grayscale 2D NumPy array for Detector.detect()
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if type(img) == np.ndarray:
            tags_in_framex = []
            for tag in at_detector.detect(img):
                # Increment frequency
                tag_ids[tag.tag_id] += 1
                # r = np.roll(np.float32(img), tag.homography + 1, axis=0)
                # Add to frames in following format - feel free to adjust
                tags_in_framex.append({
                    'id': tag.tag_id,
                    'id_confidence': tag.decision_margin,
                    'soft_id': tag.tag_id,
                    'perimeter': 100, #cv2.arcLength(img, closed=True),
                    'centroid': tag.center,
                    'verts': tag.corners,
                    'frames_since_true_detection': 0
                })

                # {'id': msg, 'id_confidence': id_confidence, 'verts': r.tolist(), 'soft_id': soft_msg,
                #  'perimeter': cv2.arcLength(r, closed=True), 'centroid': centroid.tolist(),
                #  "frames_since_true_detection": 0}
            frames.append(tags_in_framex)
        time.sleep(0.01)
        print_progress_bar(i + 1, num_images, prefix='Progress:', suffix='Complete', length=50)

    return frames, dict(tag_ids)


def extract_frames(video_path: str, frames_path: str) -> None:
    """Convert a video (mp4 or similar) into a series of individual PNG frames.
    Make sure to create a directory to store the frames before running this function.
    Args:
        video_path (str): filepath to the video being converted
        frames_path (str): filepath to the target directory that will contain the extract frames
    """
    video = cv2.VideoCapture(video_path)
    count = 0
    success = 1

    # Basically just using OpenCV's tools
    while success:
        success, frame = video.read()
        cv2.imwrite(f'{frames_path}/frame{count}.png', frame)
        count += 1

    # Optional print statement
    print(f'Extracted {count} frames from {video_path}.')