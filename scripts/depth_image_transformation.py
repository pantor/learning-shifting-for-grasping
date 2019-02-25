#!/usr/bin/python3

import cv2
import os

from depth_image import DepthImage

file_path = os.path.dirname(os.path.realpath(__file__))
image = cv2.imread(file_path + '/documents/side-overview.png', 0)

image = DepthImage(image, 2000.0, 0.15, 0.35).translate((0.0, 0.0, 0.05))
# helper.drawAroundBin(image)
# image = DepthImage(image, 2000.0, 0.15, 0.35).rotateX(-0.3, (0.0, 0.25))

# cv2.imwrite(file_path + 'side-calculated.png', image)

cv2.imshow('transformed image', image)
cv2.waitKey()
