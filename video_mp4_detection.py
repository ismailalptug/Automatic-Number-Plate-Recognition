from multiprocessing import Process, Manager

import cv2
import numpy as np
import time
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

net = cv2.dnn.readNet("yolov3_training_final.weights", "yolov3_testing.cfg")
classes = ["Plaka"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


class DisplayTask(object):

    def __init__(self, name="frame",show=True):

        self.win_name = name
        self.win_show = show

    def show(self, img,name=None,waitKey=25):
        if self.win_show:
            if name:
                cv2.imshow(name, img)
            else:
                cv2.imshow(self.win_name, img)

            if waitKey:
                cv2.waitKey(waitKey) & 0xFF

    def close(self):
        cv2.destroyAllWindows()

def readPlate(currentFrame,dt):
 while True:
  try:
    if currentFrame["img"] is None: continue

    frame= currentFrame["img"][::1]


    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)


            mask = np.zeros(frame.shape[0:2], dtype="uint8")
            cv2.rectangle(mask, (x + int(w * 0.13), y), (x + w - int(w * 0.08), y + h), 255, -1)

            (x, y) = np.where(mask == 255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = frame[x1:x2, y1:y2]
            cropped_image = cv2.resize(cropped_image, None, fx=1 / 0.2, fy=1 / 0.2)

            plate = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            plate = cv2.adaptiveThreshold(plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 11)
            kernel_size = 3
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            plate = cv2.erode(plate, kernel, iterations=1)

            plate = cv2.medianBlur(plate, kernel_size)

            dt.show( plate,"Plaka1" )

            text = pytesseract.image_to_string(plate, lang='eng',
                                               config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')

            list(text)
            text = text.split()
            print(text)


  except Exception as e:
      print(e)
      currentFrame["img"] = None
      time.sleep(3)


def capture(currentFrame,dt):

    cap = cv2.VideoCapture("./1.mp4")
    while True:
        _,  frame = cap.read()
        currentFrame["img"]= frame
        dt.show( frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()

currentFrame = {}
if __name__ == '__main__':
    with Manager() as manager:
        currentFrame = manager.dict()
        dt = DisplayTask("Plaka", True)
        p1 = Process(target=capture, args=(currentFrame,dt,))
        p2 = Process(target=readPlate, args=(currentFrame,dt,))
        p1.start()
        p2.start()
        p1.join()
        p2.join()
