import numpy as np
import cv2
import glob


paths = glob.glob("images/*.jpg")

# Modele yolo v4
net = cv2.dnn.readNet("dnn_model/yolov4-obj_last.weights", "dnn_model/yolov4-obj.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416,416), # plus grand est l'image, meilleur est la precision mais le processus est lent
                     scale=1/255)
# Liste des differentes categories d'objets que le modele de reseau (yolo) detecte pour Coco
classes = []
with open("dnn_model/classes.txt","r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)
print(classes)
count = 0

# Appliquons le detecteur sur les images
for path in paths:
    img = cv2.imread(path)
    count += 1
    class_ids, scores, bboxes =model.detect(img, nmsThreshold=0.4, confThreshold=0.5)
    for class_id, score, bboxe in zip(class_ids, scores, bboxes):
        x, y, w, h = bboxe
        class_name = classes[class_id]
        cx = int(x + w/2)
        cy = int(y + h/2)

        cv2.putText(img, class_name, (x-3,y), cv2.FONT_HERSHEY_PLAIN, 2, (200,0,50), 2)
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.circle(img, (cx,cy), 3, (0,0,255), -1)
    cv2.imshow("image", img)
    key = cv2.waitKey(0)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
