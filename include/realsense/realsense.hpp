// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#pragma once

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <map>

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

#include <realsense/config.hpp>


class texture {
    GLuint gl_handle {0};
    rs2_stream stream {RS2_STREAM_ANY};

public:
    void upload(const rs2::video_frame& frame) {
        if (!frame) {
            return;
        }

        if (!gl_handle) {
            glGenTextures(1, &gl_handle);
        }

        auto format = frame.get_profile().format();
        auto width = frame.get_width();
        auto height = frame.get_height();
        stream = frame.get_profile().stream_type();

        glBindTexture(GL_TEXTURE_2D, gl_handle);

        switch (format) {
        case RS2_FORMAT_RGB8:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, frame.get_data());
            break;
        case RS2_FORMAT_RGBA8:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, frame.get_data());
            break;
        case RS2_FORMAT_Y8:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, frame.get_data());
            break;
        case RS2_FORMAT_Z16:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_SHORT, frame.get_data());
            break;
        default:
            throw std::runtime_error("The requested format is not supported by this demo!");
        }

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    GLuint get_gl_handle() {
        return gl_handle;
    }
};

class window {
    GLFWwindow *win;

public:
    window(int width, int height, const char* title) {
        glfwInit();
        glfwWindowHint(GLFW_VISIBLE, 0);
        win = glfwCreateWindow(width, height, title, nullptr, nullptr);
        glfwMakeContextCurrent(win);
    }

    ~window() {
        glfwDestroyWindow(win);
        glfwTerminate();
    }
};


class Realsense {
    double min_depth {0.22}; // [m]
    double max_depth {0.41}; // [m]

    texture tex;

    rs2::pointcloud pc;
    rs2::points points;

    rs2::pipeline pipe;

    template<typename T, typename U>
    T clampLimits(U value) {
        return std::min<U>(std::max<U>(value, std::numeric_limits<T>::min()), std::numeric_limits<T>::max());
    }


    cv::Mat draw_pointcloud(float width, float height, bool draw_texture, texture& tex, rs2::points& points);

public:
    Realsense(RealsenseConfig config);
    ~Realsense();
  
    cv::Mat takeDepthImage();
    std::pair<cv::Mat, cv::Mat> takeImages();
};


