#!/usr/bin/python

import rospy
from cv_bridge import CvBridge
from bin_picking.srv import GetDepthImage


class Camera:
    def __init__(self):
        self.bridge = CvBridge()

    def getDepthImage(self):
        get_depth_image_srv = rospy.ServiceProxy('depth_image', GetDepthImage)
        image_msg = get_depth_image_srv()
        return self.bridge.imgmsg_to_cv2(image_msg.depth_image, 'mono8')
