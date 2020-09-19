# Huge shoutout to https://github.com/CMU-Perceptual-Computing-Lab/openpose
# Huge shoutout to https://tinyurl.com/y2vuqgxv
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import os
import pandas as pd

# download one of the output folders with json files from google drive
# put in the path to that data here
dataPath = 'C:\\Users\\carme\\Desktop\\TheProverbialCode\\run\\keypointData\\lateral2018\\'
# also need the video used to create the data (jk not yet)
# cap = cv2.VideoCapture(dataPath + '\\lateral.mov')
# def get_vid_properties():
#     width = int(cap.get(3))  # float
#     height = int(cap.get(4))  # float
#     cap.release()
#     return width, height


column_names = ['x', 'y', 'acc']
path_to_json = dataPath
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
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
            temp.append(v[:3])
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
fig, (ax1) = plt.subplots(1, 1)
ax1.plot(d[:, 14, 0], d[:, 14, 1], 'k', label='LAnkel')
ax1.plot(d[:, 11, 0], d[:, 11, 1], 'r', label='RAnkel')
ax1.plot(d[:, 13, 0], d[:, 13, 1], 'k', label='LKnee')
ax1.plot(d[:, 10, 0], d[:, 10, 1], 'r', label='RKnee')
ax1.invert_yaxis()
ax1.legend()
ax1.set_ylabel('y', rotation=0)
ax1.set_xlabel('x')
fig.tight_layout()
plt.show()

connections = [(17, 15), (15, 0), (0, 16), (16, 18), (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6),
               (6, 7), (1, 8), (8, 9), (9, 10), (10, 11), (11, 24), (11, 22), (22, 23), (8, 12),
               (12, 13), (13, 14), (14, 21), (14, 19), (19, 20)]


def moving_average(ma, data):
    csum = np.cumsum(np.insert(data, np.ones(ma + 1), 0))
    mavg = (csum[ma:] - csum[:-ma]) / float(ma)
    mavg = np.delete(mavg, 0)
    mavg[0] = mavg[1]
    return mavg


def interp_lazy(y1, y0):
    return 2 * y1 - y0


def brute_avg(data):
    return np.mean(data, axis=0)


# plot knee keypoints
d = np.array(body_keypoints_df)
d = np.reshape(d, (len(d) // 25, 25, 3))
d_og = d.copy()
plt.plot(d[:, 10, 0], 'k', label='LKnee')
plt.plot(d[:, 13, 0], 'r', label='RKnee')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Original Data')
plt.show()

# Find anomalous points and try to replace with symmetric point (opposite side of body)
for k in range(5, len(d[:, :, 0])):
    if k == 62:
        print('yeet')
    avgR = brute_avg(d[k - 5:k - 1, 10, 0])
    avgL = brute_avg(d[k - 5:k - 1, 13, 0])
    R = d[k, 10, 0]
    L = d[k, 13, 0]
    adR = abs(avgR - R) / R
    adL = abs(avgL - L) / L
    if adR > 0.15:
        adRnew = abs(avgR - L) / L
        if adRnew < 0.15:
            d[k, 10, 0], d[k, 10, 1] = d_og[k, 13, 0], d_og[k, 13, 1]
        else:
            d[k, 10, 0], d[k, 10, 1] = interp_lazy(d[k - 2, 10, 0], d[k - 1, 10, 0]), interp_lazy(
                d[k - 2, 10, 1], d[k - 1, 10, 1])
    if adL > 0.15:
        adLnew = abs(avgL - R) / R
        if adLnew < 0.15:
            d[k, 13, 0], d[k, 13, 1] = d_og[k, 10, 0], d_og[k, 10, 1]
        else:
            d[k, 13, 0], d[k, 13, 1] = interp_lazy(d[k - 2, 13, 0], d[k - 1, 13, 0]), interp_lazy(
                d[k - 2, 13, 1], d[k - 1, 13, 1])

plt.plot(d[:, 10, 0], 'k', label='LKnee')
plt.plot(d[:, 13, 0], 'r', label='RKnee')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Fixed Data')
plt.show()

fig, (ax1) = plt.subplots(1, 1)
ax1.plot(d[:, 13, 0], d[:, 13, 1], 'k', label='LKnee')
ax1.plot(d[:, 10, 0], d[:, 10, 1], 'r', label='RKnee')
ax1.invert_yaxis()
ax1.legend()
ax1.set_ylabel('y', rotation=0)
ax1.set_xlabel('x')
fig.tight_layout()
plt.show()
