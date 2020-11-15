from matplotlib import cm
import numpy as np
import traceback
import time
import cv2
import os


def get_model_data(model='body_25'):
    model_path = '/home/carmelo/Documents/pose/models/pose/'
    if model == 'mpi':
        protoFile = model_path + '/' + model + '/' + 'pose_deploy_linevec.prototxt'
        weightsFile = model_path + '/' + model + '/' + 'pose_iter_160000.caffemodel'
        body_parts = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                      "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                      "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                      "Background": 15}
        pairs = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                 ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                 ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                 ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]
        angle_pairs = [[["Head", "Neck"], ["Neck", "Chest"]],
                       [["Head", "Neck"], ["Neck", "RShoulder"]],
                       [["Neck", "RShoulder"], ["RShoulder", "RElbow"]],
                       [["RShoulder", "RElbow"], ["RElbow", "RWrist"]],
                       [["Neck", "LShoulder"], ["LShoulder", "LElbow"]],
                       [["LShoulder", "LElbow"], ["LElbow", "LWrist"]],
                       [["Head", "Neck"], ["Neck", "Chest"]],
                       [["RHip", "RKnee"], ["RKnee", "RAnkle"]],
                       [["LHip", "LKnee"], ["LKnee", "LAnkle"]],
                       [["Head", "Neck"], ["Vert"]],
                       [["Neck", "RShoulder"], ["Vert"]],
                       [["RShoulder", "RElbow"], ["Vert"]],
                       [["RElbow", "RWrist"], ["Vert"]],
                       [["Neck", "LShoulder"], ["Vert"]],
                       [["LShoulder", "LElbow"], ["Vert"]],
                       [["LElbow", "LWrist"], ["Vert"]],
                       [["Neck", "Chest"], ["Vert"]],
                       [["RHip", "RKnee"], ["Vert"]],
                       [["RKnee", "RAnkle"], ["Vert"]],
                       [["LHip", "LKnee"], ["Vert"]],
                       [["LKnee", "LAnkle"], ["Vert"]]]
    elif model == 'body_25':
        protoFile = model_path + '/' + model + '/' + 'pose_deploy.prototxt'
        weightsFile = model_path + '/' + model + '/' + 'pose_iter_584000.caffemodel'
        body_parts = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                      "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9, "RKnee": 10,
                      "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14, "REye": 15,
                      "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19, "LSmallToe": 20, "LHeel": 21, "RBigToe": 22,
                      "RSmallToe": 23, "RHeel": 24, "Background": 25, "Head": 26}
        pairs = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"], ["RElbow", "RWrist"],
                 ["LShoulder", "LElbow"], ["LElbow", "LWrist"], ["Neck", "MidHip"], ["MidHip", "RHip"],
                 ["MidHip", "LHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["RAnkle", "RHeel"], ["RAnkle", "RBigToe"],
                 ["RBigToe", "RSmallToe"], ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["LAnkle", "LHeel"],
                 ["LAnkle", "LBigToe"], ["LBigToe", "LSmallToe"], ["Neck", "Nose"], ["Nose", "REye"], ["REye", "REar"],
                 ["Nose", "LEye"], ["LEye", "LEar"]]
        angle_pairs = [[["Head", "Neck"], ["Neck", "MidHip"]],
                       [["Head", "Neck"], ["Neck", "RShoulder"]],
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
                       [["LKnee", "LAnkle"], ["Vert"]]]
    elif model == 'coco':
        protoFile = model_path + '/' + model + '/' + 'pose_deploy_linevec.prototxt'
        weightsFile = model_path + '/' + model + '/' + 'pose_iter_440000.caffemodel'
        body_parts = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                      "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                      "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                      "LEye": 15, "REar": 16, "LEar": 17, "Background": 18, "Head": 19}

        pairs = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                 ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                 ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                 ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                 ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]
        angle_pairs = [[["Head", "Neck"], ["Neck", "RShoulder"]],
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
                       [["RHip", "RKnee"], ["Vert"]],
                       [["RKnee", "RAnkle"], ["Vert"]],
                       [["LHip", "LKnee"], ["Vert"]],
                       [["LKnee", "LAnkle"], ["Vert"]]]
    else:
        body_parts = pairs = colors = None
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


def photo_inference(media_path, model, net, body_parts, pairs, angle_dict, angle_names, colors):
    thresh = 0.1
    frame = cv2.imread(media_path)
    net_w = int(16 * (1 + (384 * (frame.shape[1] / frame.shape[0])) // 16))
    net_h = 384
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (net_w, net_h), (125, 125, 125), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()
    out_h = output.shape[2]
    out_w = output.shape[3]
    points = []
    frame = cv2.resize(frame, (int(frame.shape[1] * (1024 / frame.shape[0])), 1024), interpolation=cv2.INTER_AREA)
    in_h = frame.shape[0]
    in_w = frame.shape[1]
    for i in range(len(body_parts)):
        prob_map = output[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(prob_map)
        x = (in_w * point[0]) / out_w
        y = (in_h * point[1]) / out_h
        points.append((int(x), int(y)) if conf > thresh else None)
    if model == 'body_25' or model == 'coco': points[-1] = get_head(points, model)
    for idx, pair in enumerate(pairs):
        part_a = pair[0]
        part_b = pair[1]
        assert (part_a in body_parts)
        assert (part_b in body_parts)
        idx_a = body_parts[part_a]
        idx_b = body_parts[part_b]
        if points[idx_a] and points[idx_b]:
            cv2.line(frame, points[idx_a], points[idx_b], colors[idx], 3)
            cv2.circle(frame, points[idx_a], 5, colors[idx], thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[idx_b], 5, colors[idx], thickness=-1, lineType=cv2.FILLED)
    angle_dict = calculate_angles(points, angle_names, angle_dict, body_parts)
    frame = add_angle_text(frame, angle_dict)
    photo_file_name = ''.join(media_path.split('.')[:-1]) + '_inference' + '.' + media_path.split('.')[-1]
    cv2.imwrite(photo_file_name, frame)
    # cv2.imshow('inference', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return photo_file_name


def video_inference(media_path, model, net, body_parts, pairs, angle_dict, angle_names, colors):
    thresh = 0.1
    cap = cv2.VideoCapture(media_path)
    if media_path == 0:
        media_path = 'Webcam '+ time.strftime("%Y-%m-%dT%H%M%S", time.localtime())
    ret, frame = cap.read()
    f_ = cv2.resize(frame, (int(frame.shape[1] * (1024 / frame.shape[0])), 1024), interpolation=cv2.INTER_AREA)
    in_h = f_.shape[0]
    in_w = f_.shape[1]
    f_ = add_angle_text(f_, angle_dict)
    video_h = f_.shape[0]
    video_w = f_.shape[1]
    net_w = int(16 * (1 + (384 * (frame.shape[1] / frame.shape[0])) // 16))
    net_h = 384
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if n_frames/fps > 10:
        ret = False
    video_file_name = ''.join(media_path.split('.')[:-1]) + '_inference.avi'
    video_results = cv2.VideoWriter(video_file_name, codec, fps, (video_w, video_h))
    frame_number = 0
    while ret:
        frame_number += 1
        print(100*frame_number/n_frames, end='\r')
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (net_w, net_h), (125, 125, 125), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
        out_h = output.shape[2]
        out_w = output.shape[3]
        points = []
        frame = cv2.resize(frame, (int(frame.shape[1] * (1024 / frame.shape[0])), 1024), interpolation=cv2.INTER_AREA)
        for i in range(len(body_parts)):
            prob_map = output[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(prob_map)
            x = (in_w * point[0]) / out_w
            y = (in_h * point[1]) / out_h
            points.append((int(x), int(y)) if conf > thresh else None)
        if model == 'body_25' or model == 'coco': points[-1] = get_head(points, model)
        for idx, pair in enumerate(pairs):
            part_a = pair[0]
            part_b = pair[1]
            assert (part_a in body_parts)
            assert (part_b in body_parts)
            idx_a = body_parts[part_a]
            idx_b = body_parts[part_b]
            if points[idx_a] and points[idx_b]:
                cv2.line(frame, points[idx_a], points[idx_b], colors[idx], 3)
                cv2.circle(frame, points[idx_a], 5, colors[idx], thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[idx_b], 5, colors[idx], thickness=-1, lineType=cv2.FILLED)
        angle_dict = calculate_angles(points, angle_names, angle_dict, body_parts)
        frame = add_angle_text(frame, angle_dict)
        # cv2.imshow('', frame)
        # cv2.waitKey(1)
        video_results.write(frame)
        ret, frame = cap.read()
    video_results.release()
    cv2.destroyAllWindows()
    return video_file_name


def inference(media_path, model='body_25'):
    file_name = media_path.split('/')
    file_type = file_name[-1].split('.')[-1]
    net, body_parts, pairs, angle_dict, angle_names, colors = get_model_data(model=model)
    photo_types = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
    video_types = ['mov', 'avi', 'mp4', 'MOV', 'AVI', 'MP4']
    video = True if file_type in video_types else False
    photo = True if file_type in photo_types else False
    if video:
        file_name =  video_inference(media_path, model, net, body_parts, pairs, angle_dict, angle_names, colors)
    elif photo:
            file_name = photo_inference(media_path, model, net, body_parts, pairs, angle_dict, angle_names, colors)
    else:
        print('File type not supported: ' + media_path)
        file_name = None
    return file_name


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
            cv2.putText(frame, l, (locations[ii], (idx * 45) + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, lineType=cv2.LINE_AA)
    return frame


def do_batch_inference():
    # media_path = '/home/carmelo/Documents/pose/data_processing/test_images/'
    media_path = '/home/carmelo/Desktop/'
    images = os.listdir(media_path)
    for image in images:
        try:
            inference(media_path + image)
        except Exception:
            print(traceback.format_exc())



# do_batch_inference()
