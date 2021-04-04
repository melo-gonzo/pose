from matplotlib import cm
import numpy as np
import traceback
import time
import cv2
import os


def get_model_data(model='body_25'):
    model_path = '/home/carmelo/Documents/pose/models/pose/'

    protoFile = model_path + '/' + model + '/' + 'pose_deploy.prototxt'
    weightsFile = model_path + '/' + model + '/' + 'pose_iter_584000.caffemodel'
    body_parts = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9, "RKnee": 10,
                  "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14, "REye": 15,
                  "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19, "LSmallToe": 20, "LHeel": 21, "RBigToe": 22,
                  "RSmallToe": 23, "RHeel": 24, "Head": 25}
    pairs = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"], ["RElbow", "RWrist"],
             ["LShoulder", "LElbow"], ["LElbow", "LWrist"], ["Neck", "MidHip"], ["MidHip", "RHip"],
             ["MidHip", "LHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["RAnkle", "RHeel"], ["RAnkle", "RBigToe"],
             ["RBigToe", "RSmallToe"], ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["LAnkle", "LHeel"],
             ["LAnkle", "LBigToe"], ["LBigToe", "LSmallToe"], ["Neck", "Nose"], ["Nose", "REye"], ["REye", "REar"],
             ["Nose", "LEye"], ["LEye", "LEar"]]
    angle_pairs = [[["Head", "Neck"], ["Neck", "MidHip"]],
                   [["Head", "Neck"], ["Neck", "RShoulder"]],
                   [["Head", "Neck"], ["Neck", "LShoulder"]],
                   [["Neck", "RShoulder"], ["RShoulder", "RElbow"]],
                   [["RShoulder", "RElbow"], ["RElbow", "RWrist"]],
                   [["Neck", "LShoulder"], ["LShoulder", "LElbow"]],
                   [["LShoulder", "LElbow"], ["LElbow", "LWrist"]],
                   [["RHip", "RKnee"], ["RKnee", "RAnkle"]],
                   [["LHip", "LKnee"], ["LKnee", "LAnkle"]],
                   [["Head", "Neck"], ["Vert"]],
                   [["Neck", "RShoulder"], ["Vert"]],
                   [["RShoulder", "RElbow"], ["Vert"]],
                   [["RElbow", "RWrist"], ["Vert"]],
                   [["Neck", "LShoulder"], ["Vert"]],
                   [["LShoulder", "LElbow"], ["Vert"]],
                   [["LElbow", "LWrist"], ["Vert"]],
                   [["Neck", "MidHip"], ["Vert"]],
                   [["RHip", "RKnee"], ["Vert"]],
                   [["RKnee", "RAnkle"], ["Vert"]],
                   [["LHip", "LKnee"], ["Vert"]],
                   [["LKnee", "LAnkle"], ["Vert"]],
                   [["LHeel", "LBigToe"], ["Horiz"]],
                   [["RHeel", "RBigToe"], ["Horiz"]]]

    angle_names = ['-'.join(ap[0]) + ':' + '-'.join(ap[1]) for ap in angle_pairs]
    angle_dict = dict(zip(angle_names, [[] for _ in range(len(angle_names))]))
    colors = [cm.jet(k, bytes=True)[0:3] for k in np.arange(0, 256, int(265 / (len(body_parts))))]
    colors = [(int(color[0]), int(color[1]), int(color[2])) for color in colors]
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net, body_parts, pairs, angle_dict, angle_pairs, colors


def get_angle(v1, v2):
    angles = np.arctan2(np.array([v2[1], v1[1]]), np.array([v2[0], v1[0]])) * 180 / np.pi
    return np.round(angles[0] - angles[1], 1)


def calculate_angles(d, angle_names, angle_dict, body_parts):
    vert = (0, -1)
    horiz = (1, 0)
    keys = ['-'.join(ap[0]) + ':' + '-'.join(ap[1]) for ap in angle_names]
    for idx, key in enumerate(keys):
        vectors = angle_names[idx]
        try:
            if "Vert" in vectors[0]:
                v1 = np.array(vert)
                v2 = np.array(d[body_parts[vectors[1][0]]]) - np.array(d[body_parts[vectors[1][1]]])
            elif "Vert" in vectors[1]:
                v1 = np.array(d[body_parts[vectors[0][0]]]) - np.array(d[body_parts[vectors[0][1]]])
                v2 = np.array(vert)
            elif "Horiz" in vectors[0]:
                v1 = np.array(horiz)
                v2 = np.array(d[body_parts[vectors[1][0]]]) - np.array(d[body_parts[vectors[1][1]]])
            elif "Horiz" in vectors[1]:
                v1 = np.array(d[body_parts[vectors[0][0]]]) - np.array(d[body_parts[vectors[0][1]]])
                v2 = np.array(horiz)
            else:
                v1 = np.array(d[body_parts[vectors[0][0]]]) - np.array(d[body_parts[vectors[0][1]]])
                v2 = np.array(d[body_parts[vectors[1][0]]]) - np.array(d[body_parts[vectors[1][1]]])
            angle = get_angle(v1, v2)
        except TypeError:
            angle = 10000
        angle_dict[key].append(angle)
    return angle_dict


def get_head(p, model):
    if model == 'body_25':
        hp = [0, 15, 16, 17, 18]
    else:
        hp = [0, 14, 15, 16, 17]
    hx, hy = [], []
    for h in hp:
        if p[h] is not None:
            hx.append(p[h][0])
            hy.append(p[h][1])
    try:
        head = (int(np.mean(hx)), int(np.mean(hy)))
    except ValueError:
        head = None
    return head


def add_angle_text(frame, angle_dict):
    y_start = frame.shape[1]
    text = "{:<50} {:<10}".format('Pair', 'Angle') + '\n'
    keys = angle_dict.keys()
    for key in keys:
        try:
            key_value = str(np.abs(angle_dict[key][-1]))
        except IndexError:
            key_value = str('')
        f = "{:<50} {:<10}".format(key, key_value)
        text = text + f + '\n'
    frame = np.hstack((frame, np.zeros((frame.shape[0], 700, 3)).astype('uint8')))
    angle_text = text.split('\n')
    for idx, line in enumerate(angle_text):
        line = line.split()
        locations = [y_start + 10, y_start + 600]
        for ii, l in enumerate(line):
            cv2.putText(frame, l, (locations[ii], (idx * 35) + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, lineType=cv2.LINE_AA)
    return frame


def crop_and_center_video(video_path, d, flip=False):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_w, out_h = 500, 700
    print('Hard coded video dims')
    print('Using 0th height')
    all_frames = np.zeros((out_h, out_w, 3, n_frames)).astype('uint8')
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(video_path.split('.')[0] + '_centered.avi', codec, fps, (out_w, out_h))
    for k in range(d.shape[0]):
        print(k, end='\r')
        ret, frame = cap.read()
        frame = np.pad(frame, ((out_h, out_h), (out_w, out_w), (0, 0)), mode='constant')
        frame = cv2.flip(frame, flipCode=-1) if flip else frame
        xc, yc = int(d[k, 8, 0] + out_w), int(d[0, 8, 1] + out_h)
        tile = frame[yc - out_h // 2:yc + out_h // 2, xc - out_w // 2:xc + out_w // 2, :]
        all_frames[:, :, :, k] = tile
        d[k, :, 0] = d[k, :, 0] - d[k, 8, 0] + out_w // 2
        d[k, :, 1] = d[k, :, 1] - (d[0, 8, 1] - d[k, 8, 1])
        # draw_lines(d[k, :], tile)
        cv2.imshow('', tile)
        cv2.waitKey(10)
        out.write(tile)
    out.release()
    cv2.destroyAllWindows()
    return all_frames, d


def crop_and_center_linear_video(video_path, d, flip=False):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_w, out_h = 500, 700
    print('Hard coded video dims')
    print('Using 0th height')
    all_frames = np.zeros((out_h, out_w, 3, n_frames)).astype('uint8')
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(video_path.split('.')[0] + '_centered.avi', codec, fps, (out_w, out_h))
    k = 0
    xc, yc = int(d[k, 8, 0] + out_w), int(d[0, 8, 1] + out_h)
    xc_end = int(d[-1, 8, 0] + out_w)
    pix_per_frame = int((xc_end - xc) / n_frames)
    for k in range(d.shape[0]):
        print(k, end='\r')
        ret, frame = cap.read()
        frame = np.pad(frame, ((out_h, out_h), (out_w, out_w), (0, 0)), mode='constant')
        frame = cv2.flip(frame, flipCode=-1) if flip else frame
        xc = xc + pix_per_frame
        tile = frame[yc - out_h // 2:yc + out_h // 2, xc - out_w // 2:xc + out_w // 2, :]
        all_frames[:, :, :, k] = tile
        d[k, :, 0] = d[k, :, 0] - d[k, 8, 0] + out_w // 2
        d[k, :, 1] = d[k, :, 1] - (d[0, 8, 1] - d[k, 8, 1])
        cv2.imshow('', tile)
        cv2.waitKey(1)
        out.write(tile)
    out.release()
    cv2.destroyAllWindows()
    return all_frames, d


def trim_video(video_path, start=0, end=-1):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end == -1:
        end = n_frames
    # else:
    #     end = n_frames - end
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(video_path.split('.')[0] + '_.avi', codec, fps, (out_w, out_h))
    ret = True
    current_frame = 0
    while ret:
        ret, frame = cap.read()
        cv2.putText(frame, str(current_frame), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, lineType=cv2.LINE_AA)
        if current_frame > start and current_frame < end:
            out.write(frame)
        current_frame += 1
        try:
            cv2.imshow('', cv2.resize(frame, (1024, 576)))
            cv2.waitKey(1)
        except Exception:
            pass
    out.release()
    return


def plot_angles_from_keypoints(angle_dict):
    # Middle
    # Right Top
    # Left Toop
    # Right Bottom
    # Left Bottom
    middle_ = ['Head-Neck:Neck-MidHip', 'Head-Neck:Vert', 'Neck-MidHip:Vert']
    right_top = ['Head-Neck:Neck-RShoulder', 'Neck-RShoulder:RShoulder-RElbow', 'RShoulder-RElbow:RElbow-RWrist',
                 'RShoulder-RElbow:Vert', 'RElbow-RWrist:Vert']
    left_top = ['Head-Neck:Neck-RShoulder', 'Neck-LShoulder:LShoulder-LElbow', 'LShoulder-LElbow:LElbow-LWrist',
                'LShoulder-LElbow:Vert', 'LElbow-LWrist:Vert']
    right_bottom = ['RHip-RKnee:RKnee-RAnkle', 'RHip-RKnee:Vert', 'RKnee-RAnkle:Vert', 'RHeel-RBigToe:Horiz']
    left_bottom = ['LHip-LKnee:LKnee-LAnkle', 'LHip-LKnee:Vert', 'LKnee-LAnkle:Vert', 'LHeel-LBigToe:Horiz']
    fig, (ax_mid, ax_rtop, ax_ltop, ax_rbottom, ax_lbottom) = plt.subplots(5, 1)
    for idx, p in enumerate(middle_):
        ax_mid.plot(angle_dict[p], label=middle_[idx])
        ax_mid.legend()

    for idx, p in enumerate(right_top):
        ax_rtop.plot(angle_dict[p], label=right_top[idx])
        ax_rtop.legend()

    for idx, p in enumerate(left_top):
        ax_ltop.plot(angle_dict[p], label=left_top[idx])
        ax_ltop.legend()

    for idx, p in enumerate(right_bottom):
        ax_rbottom.plot(angle_dict[p], label=right_bottom[idx])
        ax_rbottom.legend()

    for idx, p in enumerate(left_bottom):
        ax_lbottom.plot(angle_dict[p], label=left_bottom[idx])
        ax_lbottom.legend()


def print_keys(angle_dict):
    for k in angle_dict.keys():
        print(k)


def plot_specific_angles(angle_dict, plots=[]):
plots = [['RHip-RKnee:Vert', 'LHip-LKnee:Vert'], ['RHeel-RBigToe:Horiz', 'LHeel-LBigToe:Horiz']]
fig, ax = plt.subplots(len(plots), 1, sharex=True)
for idx, sub in enumerate(plots):
    for line in sub:
        ax[idx].plot(np.abs(np.array((angle_dict[line]))), label=line)
    ax[idx].legend()


# def compare