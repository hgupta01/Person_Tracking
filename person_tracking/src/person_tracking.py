#!/home/ros-indigo/rospythonenv/bin/python

import cv2
import numpy as np
import time
import tensorflow as tf
import os

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

class DetectorAPI:
    def __init__(self, path_to_ckpt, path_to_deepsort):
        self.path_to_ckpt = path_to_ckpt
        self.path_to_deepsort = path_to_deepsort

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Deep sort
        max_cosine_distance = 0.3
        nn_budget = None
        self.nms_max_overlap = 1.0
        self.threshold = 0.7

        self.encoder = gdet.create_box_encoder(self.path_to_deepsort, batch_size=1)    
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        scores = scores[0].tolist() 
        classes = [int(x) for x in classes[0].tolist()]
        num = int(num[0])

        im_height, im_width,_ = image.shape
        boxes_list = []
        for i in range(boxes.shape[1]):
            y1 = int(boxes[0,i,0] * im_height)
            x1 = int(boxes[0,i,1]*im_width)
            y2 = int(boxes[0,i,2] * im_height)
            x2 = int(boxes[0,i,3]*im_width)
            w = int(x2 - x1)
            h = int(y2 - y1)
            x = x1
            y = y1
            boxes_list.append([x, y, w, h])
        

        boxes = []
        for i in range(len(boxes_list)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > self.threshold:
                boxes.append(boxes_list[i])

        features = self.encoder(image,boxes)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)

        track_box = []
        track_id = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            track_box.append(track.to_tlbr())
            track_id.append(track.track_id)
        
        detect_box = []
        for det in detections:
            detect_box.append(det.to_tlbr())
        
        end_time = time.time()
        print("Elapsed Time:", end_time-start_time)

        return track_box, track_id, detect_box
        
    def close(self):
        self.sess.close()
        self.default_graph.close()
