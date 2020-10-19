# Huge shoutout to https://github.com/CMU-Perceptual-Computing-Lab/openpose
# Huge shoutout to https://tinyurl.com/y2vuqgxv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import cv2
import os

# download one of the output folders with json files from google drive
# put in the path to that data here
file_name = 'fast'
dataPath = '/home/carmelo/Documents/pose/keypointData/' + file_name + '/'

column_names = ['x', 'y', 'acc']
path_to_json = dataPath
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
json_files = sorted(json_files)
count = 0
# width, height = get_vid_properties()
body_keypoints_df = pd.DataFrame()
left_knee_df = pd.DataFrame()
# Loop through all json files in output directory
# Each file is a frame in the video
# If multiple people are detected - choose the most centered high confidence points
for file in json_files:
    temp_df = json.load(open(path_to_json + file))
    temp = []
    for k, v in temp_df['part_candidates'][0].items():
        # Single point detected
        if len(v) < 4:
            temp.append(v)
        # Multiple points detected. could be problematic in future. hack for now
        elif len(v) > 4:
            highest_confidence_idx = (np.argmax(v[2::3]) * 3 + 3) - 1
            # yeet = v[highest_confidence_idx-2:highest_confidence_idx]
            print('Keeping estimates with highest confidence YEET', end='\r')
            temp.append(v[highest_confidence_idx - 2:highest_confidence_idx])
        else:
            # No detection - record zeros
            temp.append([0, 0, 0])
    temp_df = pd.DataFrame(temp)
    temp_df = temp_df.fillna(-1)
    try:
        prev_temp_df = temp_df
        body_keypoints_df = body_keypoints_df.append(temp_df)
    except Exception:
        print('bad point set at: ', file)

body_keypoints_df.columns = column_names
body_keypoints_df.reset_index()
d = np.array(body_keypoints_df)
d = np.reshape(d, (len(d) // 25, 25, 3))


# plot right/left knee/ankle. it looks decent, but has some behavior that isnt right.
# from visually inspecting the video, it is clear that right/left sides of body are
# being confused at some points. lets fix that
def example_plot():
    fig, (ax1) = plt.subplots(1, 1)
    ax1.plot(d[:, 14, 0], d[:, 14, 1], 'k', label='LAnkel')
    ax1.plot(d[:, 11, 0], d[:, 11, 1], 'r', label='RAnkel')
    ax1.plot(d[:, 13, 0], d[:, 13, 1], 'b', label='LKnee')
    ax1.plot(d[:, 10, 0], d[:, 10, 1], 'g', label='RKnee')
    ax1.invert_yaxis()
    ax1.legend()
    ax1.set_ylabel('y', rotation=0)
    ax1.set_xlabel('x')
    fig.tight_layout()
    plt.show()


def make_video(d):
    def draw_lines(r, connections):
        img = np.ones((frame_height, frame_width, 3)).astype('uint8')
        for c in connections:
            pt1 = tuple(r[c[0]][:2])
            pt2 = tuple(r[c[1]][:2])
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
        cv2.imshow('', img)
        cv2.waitKey(1)
        return img

    connections = [(17, 15), (15, 0), (0, 16), (16, 18), (0, 1), (1, 2), (2, 3), (3, 4), (1, 5),
                   (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (10, 11), (11, 24), (11, 22), (22, 23),
                   (8, 12), (12, 13), (13, 14), (14, 21), (14, 19), (19, 20)]

    frame_width, frame_height = 3840, 850
    print('Hard coded video dims')
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(file_name + '.avi', codec, 10, (frame_width, frame_height))
    img = np.ones((frame_height, frame_width, 3)).astype('uint8')
    for k in range(len(d[:, 0])):
        dd = d[k, :].astype('int')
        frame = draw_lines(dd, connections)
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()


def make_centered_video(d):
    def draw_lines(r, connections):
        img = np.ones((frame_height, frame_width, 3)).astype('uint8')
        for c in connections:
            pt1 = tuple(r[c[0]][:2])
            pt2 = tuple(r[c[1]][:2])
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
        cv2.imshow('', img)
        cv2.waitKey(1)
        return img

    connections = [(17, 15), (15, 0), (0, 16), (16, 18), (0, 1), (1, 2), (2, 3), (3, 4), (1, 5),
                   (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (10, 11), (11, 24), (11, 22), (22, 23),
                   (8, 12), (12, 13), (13, 14), (14, 21), (14, 19), (19, 20)]

    frame_width, frame_height = 600, 850
    print('Hard coded video dims')
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(file_name + '_centered.avi', codec, 10, (frame_width, frame_height))
    img = np.ones((frame_height, frame_width, 3)).astype('uint8')
    for k in range(len(d[:, 0])):
        dd = d[k, :].astype('int')
        dd[:, 0] = dd[:, 0] - dd[8, 0] + 300
        frame = draw_lines(dd, connections)
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()


make_centered_video(d)
