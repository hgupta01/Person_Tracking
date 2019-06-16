#!/home/ros-indigo/rospythonenv/bin/python

import rospy
import sys
import cv2
from cv_bridge import CvBridge, CvBridgeError
from person_tracking import DetectorAPI
from sensor_msgs.msg import Image

class tracking:
    def __init__(self): #, path_to_ckpt, path_to_deepsort
        '''Initialize ros publisher, ros subscriber'''

        path_to_ckpt = rospy.get_param("/tracker/path_to_ckpt")
        path_to_deepsort = rospy.get_param("/tracker/path_to_deepsort")

        # rospy.

        # path_to_ckpt = "/home/ros-indigo/catkin_ws/tf_models/ssd/frozen_inference_graph.pb"
        # path_to_deepsort = "/home/ros-indigo/catkin_ws/tf_models/deep_sort_det/mars-small128.pb"
        
        self.bridge = CvBridge()
        self.detectorapi = DetectorAPI(path_to_ckpt, path_to_deepsort)

        # subscribed Topic
        self.subscriber = rospy.Subscriber("camImage", Image, self.callback)

    def callback(self, data):
        '''Callback function of subscribed topic. 
        Here tracking is performed'''

        #### direct conversion to CV2 ####
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            track_boxes, track_ids, detect_boxes = self.detectorapi.processFrame(cv_image)
            for i in range(len(track_boxes)):
                bbox = track_boxes[i]
                track_id = track_ids[i]
                cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                cv2.putText(cv_image, str(track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

            for bbox in detect_boxes:
                cv2.rectangle(cv_image,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

        except CvBridgeError as e:
            print(e)
        
        cv2.imshow('cv_img', cv_image)
        cv2.waitKey(2)

def main(args):
    '''Initializes and cleanup ros node'''
    ic = tracking()
    rospy.init_node('tracking')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
