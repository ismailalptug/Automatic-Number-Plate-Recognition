import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

net = cv2.dnn.readNet("yolov3_training_final.weights", "yolov3_testing.cfg")
classes = ["Plaka"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
font = cv2.FONT_HERSHEY_PLAIN

while True:

    _, frame = cap.read()
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
                # Object detected
                # print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            # cv2.putText(img, label, (x + int(w * 0.9) + 20, y + int(h/2) + 10), font, 1.5, color, 2)

            mask = np.zeros(frame.shape[0:2], dtype="uint8")
            cv2.rectangle(mask, (x + int(w * 0.11), y), (x + w, y + h), 255, -1)
            #cv2.show(mask,"Rectangular Mask" )
            masked = cv2.bitwise_and(frame, frame, mask=mask)
            # cv2.imshow("Mask Applied to Image", masked)
            (x, y) = np.where(mask == 255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = frame[x1:x2, y1:y2]
            cropped_image = cv2.resize(cropped_image, None, fx=1 / 0.2, fy=1 / 0.2)
            #plate = cv2.bitwise_not(cropped_image)
            plate = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            plate = cv2.adaptiveThreshold(plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 11)
            kernel_size = 3
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            plate = cv2.erode(plate, kernel, iterations=1)
            # plate = cv2.morphologyEx(plate, cv2.MORPH_CLOSE, kernel)
            plate = cv2.medianBlur(plate, kernel_size)
            # cv2.imshow("Plaka", plate)
            # print(pytesseract.image_to_string(plate, lang='eng',
            # config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3'))
            text = pytesseract.image_to_string(plate, lang='eng',
                                               config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')

            list(text)
            text = text.split()
            print(text)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            if (text == []):
                cv2.putText(frame, label, (x + int(w * 0.9) + 30, y + int(h / 2) + 10), font, 1.5, color, 2)
            else:
                cv2.putText(frame, text[0], (x + int(w * 0.9) + 30, y + int(h / 2) + 10), font, 1.5, color, 2)

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()




