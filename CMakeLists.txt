cmake_minimum_required (VERSION 2.6)

set (CMAKE_CXX_STANDARD 11)

add_executable (MultithreadTest MultiThreadTest.cpp)
include_directories(/home/indu/mxnet/include)
LINK_DIRECTORIES(/home/indu/mxnet/lib/)
TARGET_LINK_LIBRARIES(MultithreadTest mxnet opencv_objdetect opencv_features2d opencv_imgproc opencv_highgui opencv_core boost_thread boost_system)
