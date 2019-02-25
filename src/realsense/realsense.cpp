#include <realsense/realsense.hpp>


Realsense::Realsense(RealsenseConfig config) {
    pipe.start();
    takeImages();
}

Realsense::~Realsense() {
    pipe.stop();
}

cv::Mat Realsense::draw_pointcloud(float width, float height, bool draw_texture, texture& tex, rs2::points& points) {
    cv::Mat result = cv::Mat::zeros(cv::Size(height, width), CV_8UC3);

    if (!points) {
        return result;
    }

    // Only realsense
    // std::array<float, 3> camera_position {0.03, 0.01, 0.02};

    // Realsense + Ensenso
    std::array<float, 3> camera_position {0.03, -0.051, 0.01};

    glLoadIdentity();
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    
    float alpha = 0.0005 / 2;
    glOrtho(-alpha * width, alpha * width, -alpha * height, alpha * height, 0.01f, 10.0f);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    gluLookAt(camera_position[0], camera_position[1], camera_position[2], 0, 0, 1, 0, -1, 0);

    glPointSize(width / 640);
    if (draw_texture) {
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, tex.get_gl_handle());

        float tex_border_color[] = { 0.8f, 0.8f, 0.8f, 0.8f };
        
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, tex_border_color);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 0x812F); // GL_CLAMP_TO_EDGE
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 0x812F); // GL_CLAMP_TO_EDGE
    }

    glEnable(GL_POINT_SMOOTH);
    glBegin(GL_POINTS);
    {
        auto vertices = points.get_vertices(); // get vertices
        auto tex_coords = points.get_texture_coordinates(); // and texture coordinates

        for (int i = 0; i < points.size(); i++) {
            if (vertices[i].z) {
                if (!draw_texture) {
                    float depth = std::abs(vertices[i].z - camera_position[2]);
                    float c = std::max(std::min((depth - max_depth) / (min_depth - max_depth), 1.0), 0.0);
                    glColor3f(c, c, c);
                }

                glVertex3fv(vertices[i]);

                if (draw_texture) {
                    glTexCoord2fv(tex_coords[i]);
                }
            }
        }
    }
    glEnd();

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glPopAttrib();

    glPixelStorei(GL_PACK_ALIGNMENT, (result.step & 3) ? 1 : 4);
    glPixelStorei(GL_PACK_ROW_LENGTH, result.step / result.elemSize());
    glReadPixels(0, 0, result.cols, result.rows, GL_BGR, GL_UNSIGNED_BYTE, result.data);

    cv::flip(result, result, 1); // Horizontal flip
    if (!draw_texture) {
        cv::cvtColor(result, result, cv::COLOR_RGB2GRAY);
        result.convertTo(result, CV_16U);
        result *= 255;
    }

    return result;
}

cv::Mat Realsense::takeDepthImage() {
    auto frames = pipe.wait_for_frames();
    auto depth = frames.get_depth_frame();

    points = pc.calculate(depth);

    return draw_pointcloud(752, 480, false, tex, points);
}

std::pair<cv::Mat, cv::Mat> Realsense::takeImages() {
    auto frames = pipe.wait_for_frames();
    auto depth = frames.get_depth_frame();
    auto color = frames.get_color_frame();

    points = pc.calculate(depth);
    
    pc.map_to(color);
    tex.upload(color); 

    cv::Mat depth_image = draw_pointcloud(752, 480, false, tex, points);
    cv::Mat color_image = draw_pointcloud(752, 480, true, tex, points);

    return std::make_pair(depth_image, color_image);
}