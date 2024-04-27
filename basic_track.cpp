#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <deque>

int main() {
    cv::VideoCapture cap(0);  // Use the default camera
    if (!cap.isOpened()) {
        std::cerr << "Error opening video capture" << std::endl;
        return -1;
    }

    cv::Mat old_frame, old_gray;
    std::vector<cv::Point2f> p0, p1;

    // Capture a frame to initialize
    cap >> old_frame;
    cv::cvtColor(old_frame, old_gray, cv::COLOR_BGR2GRAY);
    cv::goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7);

    cv::Ptr<cv::BackgroundSubtractorMOG2> pBackSub;
    pBackSub = cv::createBackgroundSubtractorMOG2();

    // Maps to store trails of each point
    std::map<int, std::deque<cv::Point2f>> trails;
    cv::Mat mask = cv::Mat::zeros(old_frame.size(), old_frame.type());

    int frame_count = 0;
    while (true) {
        cv::Mat frame, frame_gray;
        cap >> frame;
        if (frame.empty()) break;
        frame_count++;

        // Skip two frames
        if (frame_count % 3 != 0) {
            continue;
        }

        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

        // Background subtraction and contour detection
        cv::Mat fgMask;
        pBackSub->apply(frame, fgMask);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Draw contours and fill them with semi-transparent yellow
        cv::Mat overlay;
        frame.copyTo(overlay);
        for (const auto& contour : contours) {
            cv::drawContours(overlay, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(0, 255, 255, 128), -1);
            cv::drawContours(frame, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(0, 255, 255), 2);
        }
        cv::addWeighted(overlay, 0.2, frame, 0.6, 0, frame);

        // Calculate optical flow
        std::vector<unsigned char> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err);

        std::vector<cv::Point2f> good_new;
        for (size_t i = 0; i < p0.size(); i++) {
            if (status[i]) {
                good_new.push_back(p1[i]);
                trails[i].push_back(p1[i]);
                if (trails[i].size() > 10) {
                    trails[i].pop_front();
                }

                // Draw the trails
                for (size_t j = 1; j < trails[i].size(); j++) {
                    cv::line(frame, trails[i][j-1], trails[i][j], cv::Scalar(0, 255, 0), 2);
                }
            }
        }

        cv::imshow("Frame", frame);

        int keyboard = cv::waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;

        // Now update the previous frame and previous points
        old_gray = frame_gray.clone();
        p0 = good_new;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
