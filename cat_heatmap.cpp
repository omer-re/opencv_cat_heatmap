#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

const std::string WEIGHTS_PATH = "../yolo_files/yolov3.weights";
const std::string CONFIG_PATH = "../yolo_files/yolov3.cfg";
const std::string NAMES_PATH = "../yolo_files/coco.names";

struct Detection {
    int classId;
    float confidence;
    cv::Rect box;
};

int main() {
    cv::dnn::Net net = cv::dnn::readNet(WEIGHTS_PATH, CONFIG_PATH);
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video capture" << std::endl;
        return -1;
    }

    cv::Mat frame;
    std::vector<std::string> classes;
    std::ifstream classNamesFile(NAMES_PATH);
    std::string line;
    while (getline(classNamesFile, line)) {
        classes.push_back(line);
    }

    int frame_count = 0;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        frame_count++;
        if (frame_count % 8 != 1) continue;

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Mat markers = cv::Mat::zeros(gray.size(), CV_32SC1);
        cv::Mat blob = cv::dnn::blobFromImage(frame, 0.00392, cv::Size(416, 416), cv::Scalar(), true, false);
        net.setInput(blob);
        std::vector<cv::Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());
        std::vector<Detection> detections;

        for (auto &out : outs) {
            for (int i = 0; i < out.rows; i++) {
                float confidence = out.at<float>(i, 4);
                if (confidence > 0.5) {
                    float* data = &out.at<float>(i, 5);
                    int bestClassId = std::max_element(data, data + 80) - data;
                    float bestConf = data[bestClassId];
                    if (bestConf > 0.5) {
                        int centerX = (int)(out.at<float>(i, 0) * frame.cols);
                        int centerY = (int)(out.at<float>(i, 1) * frame.rows);
                        int width = (int)(out.at<float>(i, 2) * frame.cols);
                        int height = (int)(out.at<float>(i, 3) * frame.rows);
                        cv::Rect box(centerX - width / 2, centerY - height / 2, width, height);
                        detections.push_back({bestClassId, bestConf, box});
                        markers.at<int>(centerY, centerX) = i + 1;  // Seed for watershed
                    }
                }
            }
        }

        // Perform the watershed algorithm
        cv::watershed(frame, markers);

        // Visualize the result
        frame.convertTo(frame, CV_8U);
        for (int i = 0; i < markers.rows; i++) {
            for (int j = 0; j < markers.cols; j++) {
                int index = markers.at<int>(i, j);
                if (index == -1) {  // Boundary between segments
                    frame.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 255, 0);
                } else if (index > 0 && index <= detections.size()) {
                    if (detections[index-1].classId == 0 || detections[index-1].classId == 16) {  // Person or Cat
                        frame.at<cv::Vec3b>(i, j) *= 0.5;
                        frame.at<cv::Vec3b>(i, j) += cv::Vec3b(128, 128, 0) * 0.5;  // Highlight detected segments
                    }
                }
            }
        }

        // Add labels
        for (const auto& det : detections) {
            if (det.classId == 0 || det.classId == 16) {
                cv::putText(frame, classes[det.classId], cv::Point(det.box.x, det.box.y - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
            }
        }

        cv::imshow("Segmented Frame", frame);
        int keyboard = cv::waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
