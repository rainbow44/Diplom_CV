import cv2
import numpy as np

threshold = 0.5  # threshold value
camera_port = 1  # camera id

nms_threshold = 0.5
cap = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

with open('coco.names', 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=threshold)
    print(classIds, bbox)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])  # converting a np array into a list
    confs = list(map(float, confs))
    indices = cv2.dnn.NMSBoxes(bbox, confs, threshold, nms_threshold)  # NonMaximumSupression

    for i in indices:
        i = i[0]

        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)
        cv2.putText(img, classNames[classIds[i][0] - 1], (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("Output", img)
    cv2.waitKey(1)

