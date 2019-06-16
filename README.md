# Person Tracking
In this repository two ros packages are given 
1. pubimage: package to publish images from webcam.
2. person_tracking : package to track people using pretrained tensorflow models and deepsort algorithm.

## person_tracking Package
The object(person) detection is done using the pretrained Tensorflow models available at [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)) trained on COCO dataset. 

For tracking and recognition of the detected objects/person, [**deepsort app**](https://github.com/nwojke/deep_sort) is used. Deepsort App uses a pretrained tf-model "mars-small128.pb" (link provided at their [github page](https://github.com/nwojke/deep_sort)).

## pubimage Package
Package to publish unaltered frames from webcam to rostopic **camImage**.

## Using in Project
For using these packages, just copy then into src folder of ROS project, build the project using *catkin_make* and run the launch file. Two changes have to be made before using the packages.
1. Changing the python environment in following nodes (replace *#!/home/ros-indigo/rospythonenv/bin/python* with your python virtualenv)
    * *person_tracking/src/person_tracking.py*
    * *person_tracking/src/tracking.py*
    * *pubimage/src/pubimage.py*
2. Provide the path to the tensorflow models in launch file p*erson_tracking/launch/tracker_launch.launch*.
