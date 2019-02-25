cmake_minimum_required(VERSION 3.5)
project(bin_picking VERSION 0.1.0 LANGUAGES CXX)


## Only build tests from subdirectory, then return
option(ONLY_BUILD_TESTS "Build test programs" OFF)


list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")


set(CMAKE_CXX_STANDARD 14)


find_package(cereal REQUIRED)
find_package(Ensenso 2.2 REQUIRED)
find_package(OpenCV 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(realsense2 REQUIRED)
find_package(yaml-cpp 0.6 REQUIRED)


include_directories(include)


if (ONLY_BUILD_TESTS)
  message("Only build tests from test subdirectory.")

  enable_testing()

  add_subdirectory(test)

  return()
endif ()


set(cpr_DIR "~/Documents/libs/cpr/")
set(Eigen3_DIRS "/usr/local/include/eigen3/")
set(PYTHON_EXECUTABLE "python3")

set(ENV{TENSORFLOW_SOURCE_DIR} "~/Documents/libs/tensorflow")
set(ENV{TENSORFLOW_BUILD_DIR} "~/Documents/libs/tensorflow_dist")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI} -O3")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}")


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(cpr REQUIRED)
find_package(CURL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Franka 0.5.0 REQUIRED)
find_package(TensorFlow 1.10 EXACT REQUIRED)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  actionlib
  std_msgs
  sensor_msgs
  geometry_msgs
  franka_msgs
  interactive_markers
  image_transport
  cv_bridge
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
)


TensorFlow_REQUIRE_C_LIBRARY()
TensorFlow_REQUIRE_SOURCE()

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   *.msg
# )

## Generate services in the 'srv' folder
add_service_files(
  FILES
  GetDepthImage.srv
  GetImages.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS
    include
#  LIBRARIES
  CATKIN_DEPENDS
    roscpp
    std_msgs
    sensor_msgs
    geometry_msgs
    interactive_markers
    franka_msgs
    moveit_core
    moveit_ros_planning
    moveit_ros_planning_interface
#  DEPENDS
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
add_library(ros INTERFACE)
target_include_directories(ros INTERFACE ${catkin_INCLUDE_DIRS})
target_link_libraries(ros INTERFACE ${catkin_LIBRARIES})


add_library(frankr src/frankr/robot.cpp src/frankr/gripper.cpp)
target_include_directories(frankr PUBLIC ${catkin_INCLUDE_DIRS} ${Eigen3_DIRS})
target_link_libraries(frankr PUBLIC ${catkin_LIBRARIES} Franka::Franka)


add_library(inference src/inference.cpp)
target_include_directories(inference PUBLIC $ENV{TENSORFLOW_BUILD_DIR}/includes/)
target_link_libraries(inference PRIVATE cereal ${OpenCV_LIBS} TensorFlow_DEP)


add_library(ensenso src/ensenso/ensenso.cpp)
target_include_directories(ensenso PUBLIC ${Eigen3_DIRS} ${ENSENSO_INCLUDE_DIRS})
target_link_libraries(ensenso PUBLIC ${ENSENSO_LIBRARIES} ${OpenCV_LIBS} ros)


add_library(realsense src/realsense/realsense.cpp)
target_include_directories(realsense PUBLIC ${Eigen3_DIRS} ${GLFW_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIR} ${realsense_INCLUDE_DIR})
target_link_libraries(realsense PUBLIC glfw ${GLFW_LIBRARIES} ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${realsense2_LIBRARY} ros)


add_executable(grasping src/grasping.cpp)
target_include_directories(grasping PRIVATE
  ${CPR_INCLUDE_DIRS}
  ${Eigen3_DIRS}
)
target_link_libraries(grasping PRIVATE
  ros
  ${OpenCV_LIBS}
  cereal
  cpr
  curl
  yaml-cpp
  frankr
  inference
  ensenso
  realsense
)


add_executable(pushing src/pushing.cpp)
target_include_directories(pushing PRIVATE
  ${CPR_INCLUDE_DIRS}
  ${Eigen3_DIRS}
)
target_link_libraries(pushing PRIVATE
  ros
  ${OpenCV_LIBS}
  cereal
  cpr
  curl
  yaml-cpp
  frankr
  inference
  ensenso
  realsense
)


add_executable(ensenso_node src/ensenso/ensenso_node.cpp)
target_link_libraries(ensenso_node PRIVATE
  ros
  ${OpenCV_LIBS}
  yaml-cpp
  ensenso
)


add_executable(realsense_node src/realsense/realsense_node.cpp)
target_link_libraries(realsense_node PRIVATE
  ros
  ${OpenCV_LIBS}
  yaml-cpp
  realsense
)

