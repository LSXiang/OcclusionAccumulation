cmake_minimum_required(VERSION 3.2)

project(OcclusionAccumulation)

set(CMAKE_CXX_STANDARD 11)

include_directories(include)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

aux_source_directory(src src_file)
add_library(${PROJECT_NAME} ${src_file})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

add_executable(example_1 demo/example_1.cc)
target_link_libraries(example_1 ${OpenCV_LIBS} ${PROJECT_NAME})