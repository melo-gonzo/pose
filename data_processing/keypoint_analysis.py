# Huge shoutout to https://github.com/CMU-Perceptual-Computing-Lab/openpose
# Huge shoutout to https://tinyurl.com/y2vuqgxv
# Huge s/o to https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import cv2
import os

# download one of the output folders with json files from google drive
# put in the path to that data here
file_name = 'slow'
# video_name = '3_.avi'
data_path = '/home/carmelo/Documents/pose/keypoint_data/' + file_name + '/'
# video_path = '/home/carmelo/Documents/pose/videos/' + video_name
video_path = '/home/carmelo/Documents/pose/keypoint_data/slow/lateral_9_17_2020.mov'


def get_keypoint_data(data_path):
    column_names = ['x', 'y', 'acc']
    path_to_json = data_path
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
    return d


# plot right/left knee/ankle. it looks decent, but has some behavior that isnt right.
# from visually inspecting the video, it is clear that right/left sides of body are
# being confused at some points. lets fix that
def example_plot(d):
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
    w, h = 3840, 850
    print('Hard coded video dims')
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('/home/carmelo/Documents/pose/videos/' + file_name + '.avi', codec, 10, (w, h))
    img = np.ones((h, w, 3)).astype('uint8')
    for k in range(len(d[:, 0])):
        dd = d[k, :].astype('int')
        frame = draw_lines(dd, img)
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()


def draw_lines(r, img):
    connections = [(17, 15), (15, 0), (0, 16), (16, 18), (0, 1), (1, 2), (2, 3), (3, 4), (1, 5),
                   (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (10, 11), (11, 24), (11, 22), (22, 23),
                   (8, 12), (12, 13), (13, 14), (14, 21), (14, 19), (19, 20)]
    for c in connections:
        pt1 = tuple(r[c[0]][:2])
        pt2 = tuple(r[c[1]][:2])
        cv2.line(img, pt1, pt2, (0, 255, 0), 2)
    return img


def make_centered_video(d):
    out_w, out_h = 400, 500
    print('Hard coded video dims')
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(file_name + '_centered.avi', codec, 10, (out_w, out_h))
    img = np.ones((out_h, out_w, 3)).astype('uint8')
    for k in range(len(d[:, 0])):
        dd = d[k, :].astype('int')
        dd[:, 0] = dd[:, 0] - dd[8, 0] + 300
        d[k, :] = dd
        frame = draw_lines(dd, img)
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()


def get_angle(v1, v2):
    print('angle in degrees')
    angles = np.arctan2(np.array([v2[1], v1[1]]), np.array([v2[0], v1[0]])) * 180 / np.pi
    return np.round(angles[0] - angles[1], 1)


def center_data(d):
    for k in range(len(d[:, 0])):
        dd = d[k, :]
        dd[:, 0] = dd[:, 0] - dd[8, 0] + 300
        d[k, :, 0] = dd[:, 0]
    return d


def calculate_angles_mpii(d):
    vert = (0, -1)
    # horiz = (1, 0)
    angle_dict = {'Head-Vert': [],
                  'Head-Spine': [],
                  'RHumerus-Spine': [],
                  'LHumerus-Spine': [],
                  'RHumerus-RRadius': [],
                  'LHumerus-LRadius': [],
                  'Spine-Vert': [],
                  'Spine-LFemur': [],
                  'Spine-RFemur': [],
                  'LFemur-Vert': [],
                  'RFemur-Vert': [],
                  'LFemur-LTibia': [],
                  'RFemur-RTibia': [],
                  'LTibia-Vert': [],
                  'RTibia-Vert': []}
    frames = len(d.shape)
    n_frames = d.shape[0] if frames >= 3 else 1
    for k in range(n_frames):
        try:
            dv = d[k, :, :]
        except Exception:
            dv = d
        head = (dv[0, 0] - dv[1, 0], dv[0, 1] - dv[1, 1])
        spine = (dv[1, 0] - dv[14, 0], dv[1, 1] - dv[14, 1])
        lhumerus = (dv[5, 0] - dv[6, 0], dv[5, 1] - dv[6, 1])
        rhumerus = (dv[2, 0] - dv[3, 0], dv[2, 1] - dv[3, 1])
        lradius = (dv[6, 0] - dv[7, 0], dv[6, 1] - dv[7, 1])
        rradius = (dv[3, 0] - dv[4, 0], dv[3, 1] - dv[4, 1])
        lfemur = (dv[11, 0] - dv[12, 0], dv[11, 1] - dv[12, 1])
        rfemur = (dv[8, 0] - dv[9, 0], dv[8, 1] - dv[9, 1])
        ltibia = (dv[12, 0] - dv[13, 0], dv[12, 1] - dv[13, 1])
        rtibia = (dv[9, 0] - dv[10, 0], dv[9, 1] - dv[10, 1])
        angle_dict['Head-Vert'].append(get_angle(head, vert))
        angle_dict['Head-Spine'].append(get_angle(head, spine))
        angle_dict['RHumerus-Spine'].append(get_angle(rhumerus, spine))
        angle_dict['LHumerus-Spine'].append(get_angle(lhumerus, spine))
        angle_dict['RHumerus-RRadius'].append(get_angle(rhumerus, rradius))
        angle_dict['LHumerus-LRadius'].append(get_angle(lhumerus, lradius))
        angle_dict['Spine-Vert'].append(get_angle(spine, vert))
        angle_dict['Spine-LFemur'].append(get_angle(spine, lfemur))
        angle_dict['Spine-RFemur'].append(get_angle(spine, rfemur))
        angle_dict['LFemur-Vert'].append(get_angle(lfemur, vert))
        angle_dict['RFemur-Vert'].append(get_angle(rfemur, vert))
        angle_dict['LFemur-LTibia'].append(get_angle(lfemur, ltibia))
        angle_dict['RFemur-RTibia'].append(get_angle(rfemur, rtibia))
        angle_dict['LTibia-Vert'].append(get_angle(ltibia, vert))
        angle_dict['RTibia-Vert'].append(get_angle(rtibia, vert))
    return angle_dict


def correct_tilt(d):
    # correct tilt based on right ankle data
    x = d[:, 11, 0]
    y = d[:, 11, 1]
    peaks = find_peaks(y, distance=20)[0]
    fit = np.polyfit(peaks, y[peaks], 1)
    d[:, :, 1] = np.array([d[idx, :, 1] - fit[0] * idx for idx in range(len(d[:, 0, 0]))])
    return d


def crop_and_center_video(video_path, d):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_w, out_h = 400, 600
    print('Hard coded video dims')
    print('Using 0th height')
    all_frames = np.zeros((out_h, out_w, 3, n_frames)).astype('uint8')
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(file_name + '_centered.avi', codec, fps, (out_w, out_h))
    for k in range(167):
        print(k, end='\r')
        ret, frame = cap.read()
        frame = np.pad(frame, ((out_h, out_h), (out_w, out_w), (0, 0)), mode='constant')
        xc, yc = d[k, 8, 0] + out_w, d[0, 8, 1] + out_h
        tile = frame[yc - out_h // 2:yc + out_h // 2, xc - out_w // 2:xc + out_w // 2, :]
        all_frames[:, :, :, k] = tile
        d[k, :, 0] = d[k, :, 0] - d[k, 8, 0] + out_w // 2
        d[k, :, 1] = d[k, :, 1] - (d[0, 8, 1] - d[k, 8, 1])
        draw_lines(d[k, :], tile)
        cv2.imshow('', tile)
        cv2.waitKey(1)
        out.write(tile)
    out.release()
    cv2.destroyAllWindows()
    return all_frames, d


def inference(media_path):
    protoFile = '/home/carmelo/Documents/pose/openpose/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt'
    weightsFile = "/home/carmelo/Documents/pose/openpose/models/pose/mpi/pose_iter_160000.caffemodel"
    connections = [(0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 14), (14, 11), (14, 8), (8, 9), (9, 10),
                   (11, 12), (12, 13)]
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    file_type = media_path.split('.')[-1]
    photo_types = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
    video_types = ['mov', 'avi', 'mp4', 'MOV', 'AVI', 'MP4']
    video = True if file_type in video_types else False
    photo = True if file_type in photo_types else False
    thresh = 0.1
    if video:
        frame = points = angles = None
        print('Do video pipeline :)')
    elif photo:
        frame = cv2.imread(media_path)
        in_h = frame.shape[0]
        in_w = frame.shape[1]
        scale_percent = (1024 / in_h)
        frame = cv2.resize(frame,
                           (int(in_w * scale_percent), int(in_h * scale_percent)))  # , interpolation=cv2.INTER_AREA)
        in_h = frame.shape[0]
        in_w = frame.shape[1]
        inWidth = 480
        inHeight = 480
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
        print(output.shape)
        out_h = output.shape[2]
        out_w = output.shape[3]
        points = []
        x_data, y_data = [], []
        # Iterate through the returned output and store the data
        # A bit of a hack for right now, should be cleaned up
        for i in range(15):
            probMap = output[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            x = (in_w * point[0]) / out_w
            y = (in_h * point[1]) / out_h
            if prob > thresh:
                points.append((int(x), int(y)))
                x_data.append(x)
                y_data.append(y)
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            else:
                points.append((None, None))
                x_data.append(None)
                y_data.append(None)
        for pair in connections:
            partA = pair[0]
            partB = pair[1]
            if points[partA][0] is not None and points[partB][0] is not None:
                cv2.line(frame, points[partA], points[partB], (0, 255, 0), 3)
        angles = calculate_angles_mpii(np.array(points))
        # cv2.imshow('inference', frame)
        # cv2.imwrite('/home/carmelo/Documents/pose/videos/bike0_inference.png', frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        frame = points = angles = None
        print('File type not supported!: ' + media_path)
    return frame, points, angles


# d = get_keypoint_data(data_path).astype('int')
# frames, d_centered = crop_and_center_video(video_path, d)
# d_corrected = correct_tilt(d_centered)
# # make_centered_video(d)
# angle_dict = calculate_angles(d_corrected)
# for key in list(angle_dict.keys()):
#     plt.plot(np.abs(angle_dict[key]), label=key)
# plt.legend()

# media_path = '/home/carmelo/Documents/pose/videos/bike0.jpg'
# frame, points, angles = inference(media_path)
# b = np.array(points)
# a = calculate_angles_mpii(b)
# print(a)
