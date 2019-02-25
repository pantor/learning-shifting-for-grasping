#!/usr/bin/python3

import cv2
import numpy as np
import os
import yaml


config = yaml.load(open(os.path.dirname(os.path.realpath(__file__)) + '/../config.yaml', 'r'))

bin_rect = [config['bin_data']['top_left'], config['bin_data']['bottom_right']]
background_color = [config['bin_data']['height']] * 3
gripper_classes = config['gripper_classes']
pixel_size = config['ensenso']['pixel_size']
min_depth = config['ensenso']['min_depth']
max_depth = config['ensenso']['max_depth']


def getStats(data):
    return {
        'total': len(data),
        'reward': data['reward'].mean(),
        'collision': data['collision'].mean(),
        'prob': data['action_prob'].mean(),
        'prob_error': abs(data['action_prob'] - data['reward']).mean(),
        'ratio': float(1.0 / data['reward'].mean() - 1.0) if data['reward'].sum() else 1,
        'methods': data['action_method'].value_counts().to_dict(),
    }


def getGripperClass(pos):
    return np.argmin(np.abs(np.array(gripper_classes) - pos))  # Find index of nearest element


def addPointsInt(x, y):
    return (int(round(x[0] + y[0])), int(round(x[1] + y[1])))


def crop(image, size_output, vec=(0, 0)):
    margin_x = int(round((image.shape[0] - size_output[0]) / 2 + vec[1]))
    margin_y = int(round((image.shape[1] - size_output[1]) / 2 + vec[0]))
    return image[margin_x:margin_x + int(round(size_output[0])), margin_y:margin_y + int(round(size_output[1]))]


def rotateVector(vec, a):  # [rad]
    return (vec[0] * np.cos(a) - vec[1] * np.sin(a), vec[0] * np.sin(a) + vec[1] * np.cos(a))


def getTransformation(x, y, a, center):  # [rad]
    rot_mat = cv2.getRotationMatrix2D((round(center[0] - x), round(center[1] - y)), a * 180.0 / np.pi, 1.0)  # [deg]
    rot_mat[0][2] += x
    rot_mat[1][2] += y
    return rot_mat


def getAreaOfInterest(image, grasp, size_cropped=None, size_result=None, border_color=background_color):  # [rad]
    size_input = (image.shape[1], image.shape[0])
    center_image = (size_input[0] / 2, size_input[1] / 2)
    trans = getTransformation(pixel_size * grasp.y, pixel_size * grasp.x, -grasp.a, center_image)
    result = cv2.warpAffine(image, trans, size_input, borderValue=border_color)
    if size_cropped:
        result = crop(result, size_output=size_cropped)
    if size_result:
        result = cv2.resize(result, size_result)
    return result


def drawAroundBin(image, color=background_color, draw_lines=False, bin=bin_rect):
    center_image = (image.shape[1] / 2, image.shape[0] / 2)
    bin_px = ((pixel_size * bin[0][0], pixel_size * bin[0][1]), (pixel_size * bin[1][0], pixel_size * bin[1][1]))
    point_1 = addPointsInt(center_image, bin_px[0])
    point_2 = addPointsInt(center_image, bin_px[1])

    if draw_lines:
        color = (255, 255, 255)
        cv2.line(image, (point_1[0], 0), (point_1[0], image.shape[0]), color, 1)
        cv2.line(image, (point_2[0], 0), (point_2[0], image.shape[0]), color, 1)
        cv2.line(image, (0, point_1[1]), (image.shape[1], point_1[1]), color, 1)
        cv2.line(image, (0, point_2[1]), (image.shape[1], point_2[1]), color, 1)
    else:
        cv2.rectangle(image, (0, 0), (point_1[0], image.shape[0]), color, -1)
        cv2.rectangle(image, (image.shape[1], 0), (point_2[0], image.shape[0]), color, -1)
        cv2.rectangle(image, (0, 0), (image.shape[1], point_1[1]), color, -1)
        cv2.rectangle(image, (0, point_2[1]), (image.shape[1], image.shape[0]), color, -1)


def drawLine(image, grasp, pt1, pt2, color, thickness=1):
    center = (image.shape[1] / 2 - pixel_size * grasp.y, image.shape[0] / 2 - pixel_size * grasp.x)
    pt1_rot = rotateVector(pt1, -grasp.a)
    pt2_rot = rotateVector(pt2, -grasp.a)
    cv2.line(image, addPointsInt(center, pt1_rot), addPointsInt(center, pt2_rot), color, thickness, lineType=cv2.LINE_AA)


def drawRotatedRect(image, grasp, size, color, thickness=1):
    vecs = [(-size[1] / 2, size[0] / 2), (size[1] / 2, size[0] / 2), (size[1] / 2, -size[0] / 2), (-size[1] / 2, -size[0] / 2)]
    for i in range(len(vecs)):
        drawLine(image, grasp, vecs[i % len(vecs)], vecs[(i + 1) % len(vecs)], color, thickness)


def drawPose(image, grasp):
    gripper_px = pixel_size * (grasp.d + 0.001)
    color_rect = (255, 0, 0)  # Blue
    color_lines = (0, 0, 255)  # Red
    color_direction = (0, 255, 0)  # Green

    drawRotatedRect(image, grasp, (200, 200), color_rect, thickness=2)  # Cropped input of CNN
    drawLine(image, grasp, (0, -90), (0, -100), color_rect, thickness=2)
    drawLine(image, grasp, (gripper_px / 2, pixel_size * 0.012), (gripper_px / 2, -pixel_size * 0.012), color_lines)
    drawLine(image, grasp, (-gripper_px / 2, pixel_size * 0.012), (-gripper_px / 2, -pixel_size * 0.012), color_lines)
    drawLine(image, grasp, (gripper_px / 2, 0), (-gripper_px / 2, 0), color_lines)
    drawLine(image, grasp, (0, pixel_size * 0.006), (0, -pixel_size * 0.006), color_lines)

    if not isinstance(grasp.z, str) and np.isfinite(grasp.z):
        x_prime = pixel_size * np.tan(grasp.b) * grasp.z
        y_prime = pixel_size * np.tan(grasp.c) * grasp.z
        drawLine(image, grasp, (y_prime, x_prime), (0, 0), color_direction)


def imageDifference(image1, image2):
    kernel = np.ones((5, 5), np.uint8)
    diff = np.zeros(image1.shape, np.uint8)
    diff[(image1 > image2 + 5) & (image1 > 0)] = 255
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel, iterations=2)
    return diff


def inpaint(image):
    mask = np.zeros(image.shape, np.uint8)
    mask[(image == 0)] = 255
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_NS) # INPAINT_TELEA, INPAINT_NS
