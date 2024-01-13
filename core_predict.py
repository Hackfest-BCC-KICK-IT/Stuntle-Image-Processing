import cv2
import numpy as np

path_yolov_weights = "./yolov/yolov3.weights"
path_yolov_cfg = "./yolov/yolov3.cfg"
path_yolov_names = "./yolov/coco.names"

net = cv2.dnn.readNet(path_yolov_weights, path_yolov_cfg)

classes = []
with open(path_yolov_names, "r") as f:
    classes = [line.strip() for line in f]

classes = []
with open(path_yolov_names, "r") as f:
    classes = [line.strip() for line in f]

class CorePredict:
    def predict(self, image, age):
        height, width, _ = image.shape

        # Load model YOLOv3
        layer_names = net.getUnconnectedOutLayersNames()

        # Pre-processing image and detect human
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(layer_names)

        # Find human object detection
        conf_threshold = 0.5
        class_ids = []
        boxes = []
        confidences = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold and classes[class_id] == "person":
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        heights = []

        # Estimated human height
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            match age:
                case 0:
                    height_cm = (h / 7) * 50
                case 1,2:
                    height_cm = (h / 7) * 65
                case 3,4:
                    height_cm = (h / 7) * 100
                case 5,6:
                    height_cm = (h / 7) * 115
                case 7,8:
                    height_cm = (h / 7) * 125
                case 9,10:
                    height_cm = (h / 7) * 140
                case 11,12:
                    height_cm = (h / 7) * 150
                case 13,14:
                    height_cm = (h / 7) * 160
                case 15,16,17:
                    height_cm = (h / 7) * 165
                case _:
                    if(age > 0):
                        height_cm = (h / 7) * 170
                    else:
                        return {
                            "height": None,
                            "message": "invalid age or image, please validate your input",
                            "success": False
                        }
            heights.append(height_cm)
        
        sum = 0.0
        for height in heights:
            sum += height
        
        avg = sum / len(heights)
        return {
            "height": avg,
            "message": "success calculate height data",
            "success": True
        }