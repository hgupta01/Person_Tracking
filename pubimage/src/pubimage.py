#!/home/ros-indigo/rospythonenv/bin/python

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def image_publisher():
    pub = rospy.Publisher('camImage', Image, queue_size=1)
    rospy.init_node('pubimage')
    rate = rospy.Rate(5)

    cap = cv2.VideoCapture(0)
    bridge = CvBridge()
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        img_msg = bridge.cv2_to_imgmsg(frame, "bgr8")
        pub.publish(img_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        image_publisher()
    except rospy.ROSInterruptException:
        pass
