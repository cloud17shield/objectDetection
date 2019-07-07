import numpy as np
import imutils
import cv2
from imageai.Detection import ObjectDetection
import tensorflow as tf

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath('yolo.h5')
detector.loadModel(detection_speed="flash")
graph = tf.get_default_graph()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    # print(type(frame))
    with graph.as_default():
        # img_array = np.array(frame)
        detected_image_array, detections = detector.detectObjectsFromImage(input_type="array",
                                                                           input_image=frame,
                                                                           output_type="array")
        # image_really = Image.fromarray(detected_image_array.astype('uint8')).convert('RGB')
        cv2.imshow("Frame", detected_image_array)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
cv2.destroyAllWindows()
cap.stop()
