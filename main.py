import cv2
import numpy as np
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import messagebox


def main():
    threshold = 0.5  # threshold value


    with open('coco.names', 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    root = Tk()
    root.title("Диплом")
    root.resizable(0, 0)
    root.rowconfigure([0, 1, 2], minsize=100)
    root.columnconfigure(0, minsize=250)
    frame = ttk.Frame(root, padding=10)
    frame.pack()
    root.geometry("250x150")

    def pic_select_window():

        def pic_detection(imgurl):

            img = cv2.imread(imgurl)


            net = cv2.dnn_DetectionModel(weightsPath, configPath)
            net.setInputSize(320, 320)
            net.setInputScale(1.0 / 127.5)
            net.setInputMean((127.5, 127.5, 127.5))
            net.setInputSwapRB(True)

            classIds, confs, bbox = net.detect(img, confThreshold=threshold)

            if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
                    cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence, 2)), (box[0] + 10, box[1] + 50),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)



                cv2.imshow("Output", img)
                cv2.waitKey(1)
            elif len(classIds) == 0:
                messagebox.showerror(title='Error', message='No objects detected!')

        newWindow = Toplevel()
        newWindow.title('Picture Detection')
        newWindow.geometry("250x100")
        newWindow.resizable(0, 0)
        newWindow.columnconfigure(0, weight=0)
        newWindow.columnconfigure(1, weight=3)

        def get_file():
            filename = fd.askopenfilename(filetypes=[
                ('image', '.jpeg'),
                ('image', '.jpg'),
                ('image', '.png')
            ])
            TextArea.config(state=NORMAL)
            TextArea.delete(0, END)
            TextArea.insert(0, filename)
            TextArea.config(state=DISABLED)

        def pic_detection_init():
            try:
                pic_detection(TextArea.get())
            except:
                messagebox.showerror(title='Error', message='Incorrect filepath')


        FileLabel = ttk.Label(newWindow, text="Picture:")
        FileLabel.grid(column=0, row=0, sticky=W, padx=5, pady=5)
        TextArea = ttk.Entry(newWindow, state=DISABLED)
        TextArea.grid(column=1, row=0, sticky=EW, padx=5, pady=5)
        ChooseFileButton = ttk.Button(newWindow, text="Choose a picture", command=get_file)
        ChooseFileButton.grid(column=1, row=1, sticky=EW, padx=5, pady=5)
        GoButton = ttk.Button(newWindow, text="Go!", command=pic_detection_init)
        GoButton.grid(column=1, row=2, sticky=EW, padx=5, pady=5)

        newWindow.mainloop()

    def cam_detection():
        camera_port = 0  # camera id
        nms_threshold = 0.5
        cap = cv2.VideoCapture(camera_port)
        cap.set(3, 640)
        cap.set(4, 480)

        net = cv2.dnn_DetectionModel(weightsPath, configPath)
        net.setInputSize(320, 320)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)

        while True:
            _, img = cap.read()
            classIds, confs, bbox = net.detect(img, confThreshold=threshold)
            bbox = list(bbox)
            confs = list(np.array(confs).reshape(1, -1)[0])  # converting a np array into a list
            confs = list(map(float, confs))
            indices = cv2.dnn.NMSBoxes(bbox, confs, threshold, nms_threshold)  # NonMaximumSuppression

            for i in indices:
                i = i[0]

                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classIds[i][0] - 1], (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                cv2.putText(img, str(round(confs[i], 2)), (box[0] + 10, box[1] + 50),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

            cv2.imshow("Output", img)
            cv2.waitKey(1)

    PicDetection = ttk.Button(frame, text="Picture detection", command=pic_select_window)
    PicDetection.grid(column=1, row=0, sticky='EW', ipadx=10, ipady=10)
    CamDetection = ttk.Button(frame, text="Camera detection", command=cam_detection)
    CamDetection.grid(column=1, row=1, sticky='EW', ipadx=10, ipady=10)
    QuitButton = ttk.Button(frame, text="Quit", command=root.destroy)
    QuitButton.grid(column=1, row=2, sticky='EW', ipadx=10, ipady=10)

    root.mainloop()


main()
