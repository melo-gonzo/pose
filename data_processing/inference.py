# Imprt Libraries
from scipy.spatial import distance as dist
import numpy as np
import pandas as pd
import cv2

# Necessary Paths
protoFile = '/home/carmelo/Documents/pose/openpose/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt'
weightsFile = "/home/carmelo/Documents/pose/openpose/models/pose/mpi/pose_iter_160000.caffemodel"

video_path = '/home/carmelo/Documents/pose/videos/lateral.mov'
csv_path = '/home/carmelo/Documents/pose/videos/tester.csv'

# Load the model and the weights
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# Store the input video specifics
cap = cv2.VideoCapture(video_path)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
ok, frame = cap.read()
(frameHeight, frameWidth) = frame.shape[:2]
h = 500
w = int((h / frameHeight) * frameWidth)

# Dimensions for inputing into the model
inHeight = 640  # frameHeight
inWidth = 640  # frameWidth #386

data = []
previous_x, previous_y = list(np.zeros(16)), list(np.zeros(16))

# Define the output
out_path = '/home/carmelo/Documents/pose/keypointData/inference.avi'
output = cv2.VideoWriter(out_path, 0, fps, (w, h))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = None
(f_h, f_w) = (h, w)
zeros = None

# There are 15 points in the skeleton
pairs = [[0, 1],  # head
         [1, 2], [1, 5],  # sholders
         [2, 3], [3, 4], [5, 6], [6, 7],  # arms
         [1, 14], [14, 11], [14, 8],  # hips
         [8, 9], [9, 10], [11, 12], [12, 13]]  # legs

# probability threshold fro prediction of the coordinates
thresh = 0.1

circle_color, line_color = (0, 255, 255), (0, 255, 0)

# Start the iteration
while True:
    ok, frame = cap.read()

    if ok != True:
        break

    frame = cv2.resize(frame, (w, h), cv2.INTER_AREA)
    frame_copy = np.copy(frame)

    # Input the frame into the model
    inpBlob = cv2.dnn.blobFromImage(frame_copy, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]

    points = []
    x_data, y_data = [], []

    # Iterate through the returned output and store the data
    for i in range(15):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        x = (w * point[0]) / W
        y = (h * point[1]) / H

        if prob > thresh:
            points.append((int(x), int(y)))
            x_data.append(x)
            y_data.append(y)
        else:
            points.append((0, 0))
            x_data.append(previous_x[i])
            y_data.append(previous_y[i])

    for i in range(len(points)):
        cv2.circle(frame_copy, (points[i][0], points[i][1]), 2, circle_color, -1)

    for pair in pairs:
        partA = pair[0]
        partB = pair[1]
        cv2.line(frame_copy, points[partA], points[partB], line_color, 1, lineType=cv2.LINE_AA)

    if writer is None:
        writer = cv2.VideoWriter(out_path, fourcc, fps, (f_w, f_h), True)
        zeros = np.zeros((f_h, f_w), dtype="uint8")

    writer.write(cv2.resize(frame_copy, (f_w, f_h)))

    cv2.imshow('frame', frame_copy)

    data.append(x_data + y_data)
    previous_x, previous_y = x_data, y_data

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# Save the output data from the video in CSV format
df = pd.DataFrame(data)
df.to_csv(csv_path, index=False)
print('save complete')

cap.release()
cv2.destroyAllWindows()
