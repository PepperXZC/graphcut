cmake_minimum_required(VERSION 3.12)
project(c__)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 11)

add_executable(c__ main.cpp)

target_link_libraries(c__ ${OpenCV_LIBS})