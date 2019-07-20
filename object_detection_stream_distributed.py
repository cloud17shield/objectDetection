from kafka import KafkaProducer

from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

from imutils import face_utils
from scipy.spatial import distance
import numpy as np
import imutils
import dlib
import cv2
from imageai.Detection import ObjectDetection
import os
from PIL import Image
import tensorflow as tf
import time

conf = SparkConf().setAppName("object detection streaming").setMaster("yarn")
conf.set("spark.scheduler.mode", "FAIR")
sc = SparkContext(conf=conf)
sc.setLocalProperty("spark.scheduler.pool", "pool3")
ssc = StreamingContext(sc, 0.5)
sql_sc = SQLContext(sc)
input_topic = 'input'
output_topic = 'output3'
brokers = "G01-01:2181,G01-02:2181,G01-03:2181,G01-04:2181,G01-05:2181,G01-06:2181,G01-07:2181,G01-08:2181," \
          "G01-09:2181,G01-10:2181,G01-11:2181,G01-12:2181,G01-13:2181,G01-14:2181,G01-15:2181,G01-16:2181"


def my_decoder(s):
    return s


kafkaStream = KafkaUtils.createStream(ssc, brokers, 'test-consumer-group-3', {input_topic: 15},
                                      valueDecoder=my_decoder)
producer = KafkaProducer(bootstrap_servers='G01-01:9092', compression_type='gzip', batch_size=163840,
                         buffer_memory=33554432, max_request_size=20485760)

detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()  # !!!tiny
detector.setModelPath('/home/hduser/yolo-tiny.h5')
detector.loadModel(detection_speed="flash")
custom = detector.CustomObjects(person=True, bottle=True, knife=True, cell_phone=True, fork=True)
graph = tf.get_default_graph()
