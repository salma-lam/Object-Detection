import cv2
import numpy as np

# Chargement des classes YOLO
classes = None
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Chargement du modèle YOLO et des poids pré-entraînés
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Chargement des couches de sortie
layer_names = net.getLayerNames() 
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Chargement de l'image
image = cv2.imread('image.jpg')
height, width, channels = image.shape

# Prétraitement de l'image
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Initialisation de listes pour les boîtes englobantes, les confidences et les classes détectées
boxes = []
confidences = []
class_ids = []

# Analyse des détections
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Coordonnées de la boîte englobante
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Coordonnées de la boîte englobante (coin supérieur gauche)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Suppression des détections multiples et application de la non-maxima suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Affichage des détections
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y + 30), font, 3, color, 3)

# Affichage de l'image avec les détections
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
