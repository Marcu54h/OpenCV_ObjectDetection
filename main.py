import cv2

#img = cv2.imread('IMG_0818.JPG')

cap = cv2.VideoCapture(0)

cap.set(3, 1920)
cap.set(4, 1080)

class_names = []

class_file = "classnames.txt"

with open(class_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')
    
config_path = 'ssd_inception_v2_coco_2017_11_17.pbtxt'
weights_path = '.\\ssd_mobilenet_v3_large_coco_2020_01_14\\frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weights_path, config_path)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.6)
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            class_name = str(classId)
            
            try:
                class_name = class_names[classId-1] + " - " + str(int(confidence * 100)) + "%"
            except IndexError:
                print("Index out of range ->", classId) 
            finally:
                cv2.putText(img, class_name, (box[0] + 10, box[1] + 40), cv2.FONT_HERSHEY_COMPLEX,
                            2, (0, 255, 0), thickness=2)

        cv2.imshow("Output", img)
        cv2.waitKey(1)