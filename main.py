import cv2
import numpy as np
from tkinter import *
from tkinter import ttk



def main_function():

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
            cv2.putText(img, str(round(confs[i][0] * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Output", img)
        cv2.waitKey(1)
root = Tk()
root.title("Диплом")
root.resizable(0, 0)
root.rowconfigure([0, 1, 2], minsize=100)
root.columnconfigure(0, minsize=250)
frame = ttk.Frame(root, padding=10)
frame.pack()
root.geometry("250x150")

PicDetection = ttk.Button(frame, text="Picture detection", command=root.destroy, )
PicDetection.grid(column=0, row=0, sticky='EW', ipadx=10, ipady=10)
CamDetection = ttk.Button(frame, text="Camera detection", command=main_function)
CamDetection.grid(column=0, row=1, sticky='EW', ipadx=10, ipady=10)
QuitButton = ttk.Button(frame, text="Quit", command=root.destroy)
QuitButton.grid(column=0, row=2, sticky='EW', ipadx=10, ipady=10)

root.mainloop()