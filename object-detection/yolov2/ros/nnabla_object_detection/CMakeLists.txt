# Copyright 2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 2.8.3)
project(nnabla_object_detection)

# Options
option(WITH_CUDA "Build with CUDA extension" ON)
if (NOT WITH_CUDA)
  add_definitions(-DWITHOUT_CUDA)
endif()

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
  )

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

################################################
## Declare ROS messages, services and actions ##
################################################


###################################
## catkin specific configuration ##
###################################
catkin_package()

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ executable
add_executable(${PROJECT_NAME}_detection_node src/detection_node.cpp)

## Rename C++ executable without prefix
set_target_properties(${PROJECT_NAME}_detection_node PROPERTIES OUTPUT_NAME detection PREFIX "")

## Add cmake target dependencies of the executable
add_dependencies(${PROJECT_NAME}_detection_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_detection_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  nnabla
  nnabla_utils
  )
if(WITH_CUDA)
  target_link_libraries(${PROJECT_NAME}_detection_node
    nnabla_cuda
    )
endif()
