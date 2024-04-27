cmake_minimum_required(VERSION 3.27)
project(opencv_tutorials)

set(CMAKE_CXX_STANDARD 17)

# Ensure this points to the directory containing OpenCVConfig.cmake
set(OpenCV_DIR "C:/tools/opencv/build")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencv_tutorials cat_heatmap.cpp)

set(OPENCV_LINK_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

target_link_libraries(opencv_tutorials ${OpenCV_LIBS})