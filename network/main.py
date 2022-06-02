import cv2
import numpy as np


def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers


def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=1, fy=1)
    height, width, channels = img.shape
    return img, height, width, channels


def start_webcam():
    cap = cv2.VideoCapture(0)
    return cap


def display_blob(blob):
    for b in blob:
        for n, imgb in enumerate(b):
            cv2.imshow(str(n), imgb)


def detect_objects(img, net, output_layers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00754, size=(1024, 1024), mean=(0, 0, 0),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def get_img_with_weapons(boxes, confs, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.2, 0.25)
    font = cv2.FONT_ITALIC
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            weapon_class = str(classes[class_ids[i]])
            percentage = str(confs[i] * 100)[:4] + "%"
            color = (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, "Gun" + " " + percentage, (x, y - 5), font, 0.8, color, 1)
    img = cv2.resize(img, (800, 600))
    cv2.imshow("Pistol Detector", img)
    cv2.imwrite("hello.jpg", img)
    return img


def image_detect(img_path):
    model, classes, output_layers = load_yolo()
    image, height, width, channels = load_image(img_path)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    img = get_img_with_weapons(boxes, confs, class_ids, classes, image)
    while True:
        cv2.waitKey(1)


def webcam_detect():
    model, classes, output_layers = load_yolo()
    cap = start_webcam()
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        get_img_with_weapons(boxes, confs, class_ids, classes, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()


def start_video(video_path):
    model, classes, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        get_img_with_weapons(boxes, confs, class_ids, classes, frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    image_detect("images/img_10.png")
