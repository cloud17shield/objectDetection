import numpy as np
import imutils
import cv2
from imageai.Detection import ObjectDetection
import tensorflow as tf

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath('yolo.h5')
detector.loadModel()
graph = tf.get_default_graph()
with graph.as_default():
    frame = cv2.imread("/Users/ranxin/Downloads/knife/woman-3409529__480.jpg")
    frame = imutils.resize(frame, width=600)
    detected_image_array, detections = detector.detectObjectsFromImage(input_type="array",
                                                                       input_image=frame,
                                                                       output_type="array")
    cv2.imwrite("obj2.jpg", detected_image_array)
