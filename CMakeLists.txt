cmake_minimum_required (VERSION 2.6)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

LINK_DIRECTORIES(/home/indu/mxnet/lib/)
add_executable (MultithreadTest MultiThreadTest.cpp)
include_directories(/home/indu/mxnet/include)
TARGET_LINK_LIBRARIES(MultithreadTest  mxnet opencv_imgcodecs opencv_objdetect opencv_features2d opencv_imgproc opencv_highgui opencv_core boost_thread boost_system pthread)
