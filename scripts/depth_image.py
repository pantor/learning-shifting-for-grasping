#!/usr/bin/python3

import cv2
import numpy as np


class DepthImage:
    def __init__(self, image, pixel_size, depth_min, depth_max):
        self.image = image
        self.pixel_size = pixel_size
        self.depth_min = depth_min
        self.depth_max = depth_max

    def depthFromValue(self, value):
        value = value.astype(float) if isinstance(value, np.ndarray) else float(value)
        return self.depth_max + value / 255 * (self.depth_min - self.depth_max)

    def valueFromDepth(self, d):
        return np.maximum(np.minimum(np.round(255 * (d - self.depth_max) / (self.depth_min - self.depth_max)), 255), 0)

    def translate(self, translation):
        M = np.float32([[1, 0, self.pixel_size * translation[0]], [0, 1, self.pixel_size * translation[1]]])
        result = cv2.warpAffine(self.image, M, (self.image.shape[1], self.image.shape[0]))
        translation_brightness = int(round(255.0 / (self.depth_max - self.depth_min) * translation[2]))

        translated = result[:, :].astype(float) + translation_brightness
        result[:, :] = np.maximum(np.minimum(translated[:, :], 255), 0)
        return result

    def rotateX(self, angle, vector):
        result = np.zeros(self.image.shape, np.uint8)

        j = np.arange(self.image.shape[1])
        for i in range(self.image.shape[0]):
            y = (i - self.image.shape[0] / 2) / self.pixel_size + vector[0]

            d = self.depthFromValue(self.image[i]) - vector[1]

            y_new = y * np.cos(angle) - d * np.sin(angle)
            d_new = y * np.sin(angle) + d * np.cos(angle)
            i_new = np.maximum(np.minimum(np.round((y_new - vector[0]) * self.pixel_size + self.image.shape[0] / 2), self.image.shape[0] - 1), 0).astype(int)
            value_new = np.maximum(self.valueFromDepth(d_new + vector[1]), result[i_new, j]).astype(np.uint8)

            mask = np.zeros(self.image.shape, np.bool)
            mask[i_new, j] = (self.image[i] != 0)  # Don't change value if image is black and depth unknown
            np.putmask(result, mask, value_new)

        return result

    # Not tested
    def rotateY(self, angle, vector):
        result = np.zeros(self.image.shape, np.uint8)

        i = np.arange(self.image.shape[0])
        for j in range(self.image.shape[1]):
            x = (j - self.image.shape[1] / 2) / self.pixel_size + vector[0]

            d = self.depthFromValue(self.image[:, j]) - vector[1]

            x_new = x * np.cos(angle) - d * np.sin(angle)
            d_new = x * np.sin(angle) + d * np.cos(angle)
            j_new = np.maximum(np.minimum(np.round((x_new - vector[0]) * self.pixel_size + self.image.shape[1] / 2), self.image.shape[1] - 1), 0).astype(int)
            value_new = np.maximum(self.valueFromDepth(d_new + vector[1], self.depth_min, self.depth_max), result[i, j_new]).astype(np.uint8)

            mask = np.zeros(self.image.shape, np.bool)
            mask[i, j_new] = (self.image[i, j_new] != 0)  # Don't change value if image is black and depth unknown
            np.putmask(result, mask, value_new)

            # j_new = max(min(int(round((x_new - vector[0]) * pixel_size + image.shape[1] / 2)), image.shape[1] - 1), 0)
            # result[i][j_new] = max(valueFromDepth(d_new + vector[1], depth_min, depth_max), result[i][j_new])

        return result
