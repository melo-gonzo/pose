from utils import *
import sys

sys.path.append('/home/carmelo/Projects/Pose/openpose/build/python')
from openpose import pyopenpose as op


def photo_inference(media_path, model, body_parts, pairs, kp_dict, angle_dict, angle_names, colors, flip=False):
    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "/home/carmelo/Projects/Pose/models/"
    params["net_resolution"] = "-1x256"
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    # Process Image
    datum = op.Datum()
    frame = cv2.imread(media_path)
    frame = cv2.flip(frame, flipCode=-1) if flip else frame
    scalee = 1024
    frame = cv2.resize(frame, (int(frame.shape[1] * (scalee / frame.shape[0])), scalee), interpolation=cv2.INTER_AREA)
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    # Display Image
    try:
        output = np.array(datum.poseKeypoints[0])
    except Exception:
        output = np.zeros((25, 2))
    in_h = frame.shape[0]
    in_w = frame.shape[1]
    out_h = datum.netOutputSize.y
    out_w = datum.netOutputSize.x
    points = [(int(part[0] * (in_w / out_w)), int(part[1] * (in_h / out_h))) for part in output]
    if model == 'body_25' or model == 'coco':
        points.append(get_head(points, model))
    for idx, val in enumerate(kp_dict.keys()):
        kp_dict[val].append(points[idx])
    for idx, pair in enumerate(pairs[:-1]):
        part_a = pair[0]
        part_b = pair[1]
        assert (part_a in body_parts)
        assert (part_b in body_parts)
        idx_a = body_parts[part_a]
        idx_b = body_parts[part_b]
        if points[idx_a] and points[idx_b]:
            if points[idx_a] != (0, 0) and points[idx_b] != (0, 0):
                cv2.line(frame, points[idx_a], points[idx_b], colors[idx], 3)
            if points[idx_a] != (0, 0):
                cv2.circle(frame, points[idx_a], 5, colors[idx], thickness=-1, lineType=cv2.FILLED)
            if points[idx_b] != (0, 0):
                cv2.circle(frame, points[idx_b], 5, colors[idx], thickness=-1, lineType=cv2.FILLED)
    angle_dict = calculate_angles(points, angle_names, angle_dict, body_parts)
    frame = add_angle_text(frame, angle_dict)
    photo_file_name = ''.join(media_path.split('.')[:-1]) + '_inference' + '.' + media_path.split('.')[-1]
    cv2.imwrite(photo_file_name, frame)
    # cv2.imshow('inference', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return photo_file_name, kp_dict, angle_dict


def video_inference(media_path, model, body_parts, pairs, kp_dict, angle_dict, angle_names, colors, flip=False,
                    rotate=False):
    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "/home/carmelo/Projects/Pose/models/"
    params["net_resolution"] = "-1x480"  # 256, 288, 320, 400, 480
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    # Process Image
    datum = op.Datum()
    save_video = True
    cap = cv2.VideoCapture(media_path)
    if media_path == 0:
        media_path = 'Webcam ' + time.strftime("%Y-%m-%dT%H%M%S", time.localtime())
    ret, frame = cap.read()
    frame = cv2.flip(frame, flipCode=-1) if flip else frame
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) if rotate else frame
    scalee = 1024
    f_ = cv2.resize(frame, (int(frame.shape[1] * (scalee / frame.shape[0])), scalee), interpolation=cv2.INTER_AREA)
    in_h = f_.shape[0]
    in_w = f_.shape[1]
    f_ = add_angle_text(f_, angle_dict)
    video_h = f_.shape[0]
    video_w = f_.shape[1]
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps = 5
    # if n_frames / fps > 10:
    #     ret = False
    video_file_name = ''.join(media_path.split('.')[:-1]) + '_inference.avi'
    if save_video:
        video_results = cv2.VideoWriter(video_file_name, codec, fps, (video_w, video_h))
    frame_number = 0
    kps = np.zeros((n_frames, 25, 2))
    while ret:
        frame_number += 1
        print(100 * frame_number / n_frames, end='\r')
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        # Display Image
        try:
            output = np.array(datum.poseKeypoints[0])
        except Exception:
            output = np.zeros((25, 2))
        out_h = datum.netOutputSize.y
        out_w = datum.netOutputSize.x
        frame = cv2.resize(frame, (int(frame.shape[1] * (scalee / frame.shape[0])), scalee),
                           interpolation=cv2.INTER_AREA)
        points = [(int(part[0] * (in_w / out_w)), int(part[1] * (in_h / out_h))) for part in output]
        if model == 'body_25' or model == 'coco':
            points.append(get_head(points, model))
        for idx, val in enumerate(kp_dict.keys()):
            kp_dict[val].append(points[idx])
        for idx, pair in enumerate(pairs[:-1]):
            part_a = pair[0]
            part_b = pair[1]
            assert (part_a in body_parts)
            assert (part_b in body_parts)
            idx_a = body_parts[part_a]
            idx_b = body_parts[part_b]
            if points[idx_a] and points[idx_b]:
                if points[idx_a] != (0, 0) and points[idx_b] != (0, 0):
                    cv2.line(frame, points[idx_a], points[idx_b], colors[idx], 3)
                if points[idx_a] != (0, 0):
                    cv2.circle(frame, points[idx_a], 5, colors[idx], thickness=-1, lineType=cv2.FILLED)
                if points[idx_b] != (0, 0):
                    cv2.circle(frame, points[idx_b], 5, colors[idx], thickness=-1, lineType=cv2.FILLED)
        angle_dict = calculate_angles(points, angle_names, angle_dict, body_parts)
        frame = add_angle_text(frame, angle_dict)
        if save_video: video_results.write(frame)
        frame = cv2.resize(frame, (1800, (int(frame.shape[0] * (1800 / frame.shape[1])))), interpolation=cv2.INTER_AREA)
        cv2.imshow('', frame)
        cv2.waitKey(1)
        ret, frame = cap.read()
        frame = cv2.flip(frame, flipCode=-1) if flip else frame
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) if rotate else frame
    if save_video: video_results.release()
    cv2.destroyAllWindows()
    return video_file_name, angle_dict, kps


def inference(media_path, model='body_25', flip=False, rotate=False):
    if media_path != 0:
        file_name = media_path.split('/')
        file_type = file_name[-1].split('.')[-1]
    else:
        file_type = 'avi'
    body_parts, pairs, kp_dict, angle_dict, angle_pairs, colors = get_model_data(model=model)
    photo_types = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
    video_types = ['mov', 'avi', 'mp4', 'MOV', 'AVI', 'MP4']
    video = True if file_type in video_types else False
    photo = True if file_type in photo_types else False
    if video:
        file_name, angle_dict, kps = video_inference(media_path, model, body_parts, pairs, kp_dict, angle_dict, angle_pairs, colors,
                                    flip=flip, rotate=rotate)
    elif photo:
        file_name, angle_dict, kps  = photo_inference(media_path, model, body_parts, pairs, kp_dict, angle_dict, angle_pairs, colors,
                                    flip=flip)
    else:
        print('File type not supported: ' + media_path)
        file_name = None
    return file_name, angle_dict, kps


def do_batch_inference():
    media_path = '/home/carmelo/Projects/pose/data_processing/test_images/'
    images = os.listdir(media_path)
    for image in images:
        try:
            inference(media_path + image)
        except Exception:
            print(traceback.format_exc())


def get_centered_video(video_path, flip=True):
    og_path = video_path.split('.')[0]
    _, angle_dict, kps = inference(video_path, flip=flip)
    crop_and_center_linear_video(video_path, kps, flip=flip)
    # _, angle_dict, kps = inference(og_path + '_centered.avi', flip=flip)
    # crop_and_center_video(video_path, kps, flip=flip)


# file_path = '/home/carmelo/Desktop/iphone/'
# files = [file for file in os.listdir(file_path) if '.MOV' in file]
# for file in files:
#     full_file_path = file_path + file
#     _, kp_dict1, angle_dict1 = inference(full_file_path, flip=False, rotate=True)

# file_path = '/home/carmelo/Projects/Pose/videos/tiger.mov'
# get_centered_video(file_path, flip=False)
# trim_video(file_path, start=30)
# _, kp_dict1, angle_dict1 = inference(file_path, flip=True, rotate=True)


k = 1

_, angle_dict_1, kps_1 = inference('/home/carmelo/Projects/Pose/videos/GJGolfPhotos/bob4.jpg', flip=False)
_, angle_dict_2, kps_2 = inference('/home/carmelo/Projects/Pose/videos/GJGolfPhotos/reg4.jpg', flip=False)
_, angle_dict_3, kps_3 = inference('/home/carmelo/Projects/Pose/videos/GJGolfPhotos/tig4.jpg', flip=False)

# frame = skeleton(angle_dict_1, color=(255,0,0))
# frame = skeleton(angle_dict_2, color=(0,0,255), frame=frame)
# frame = skeleton(angle_dict_3, color=(0,255,0), frame=frame)

# cv2.imwrite('/home/carmelo/Projects/Pose/videos/GJGolfPhotos/StartSwing.png', frame)


# fig, ax = plt.subplots(1,1)
# frame = skeleton(kp_dict1)
# frame = skeleton(kp_dict2, frame=frame)
# cv2.imshow('', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# _, angle_dict_1, kps_1 = inference('/home/carmelo/Documents/pose/videos/cuboid1_centered.avi', flip=False)
# _, angle_dict_2, kps_2 = inference('/home/carmelo/Documents/pose/videos/IMG_5552__centered.avi', flip=False)
# kk = 1
# trim_video('/home/carmelo/Documents/pose/videos/IMG_5552.mov', start=33, end=195)
# get_centered_video('/home/carmelo/Documents/pose/videos/cuboid1.mov')
