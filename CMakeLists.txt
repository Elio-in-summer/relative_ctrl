cmake_minimum_required(VERSION 3.0.2)
project(relative_ctrl)

add_compile_options(-std=c++14)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")


find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp mavros_msgs 
)

catkin_package(
  INCLUDE_DIRS include ${EIGEN3_INCLUDE_DIR}
  LIBRARIES ${PROJECT_NAME} 
  CATKIN_DEPENDS roscpp mavros_msgs 
  DEPENDS EIGEN3 OpenCV
)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
)

# 这是给后面的exe加的dependency，要和后面同步地注释或者取消注释
# add_dependencies(${PROJECT_NAME}
#   ${catkin_EXPORTED_TARGETS}
# )


#Specify the executables for building 
add_executable(trans_relative_pose src/trans_relative_pose.cpp)
# Link against required libraries
target_link_libraries(trans_relative_pose ${catkin_LIBRARIES})

# #Specify the executables for building
# add_executable(rviz_planning src/rviz_planning.cpp)
# # Link against required libraries
# target_link_libraries(rviz_planning ${catkin_LIBRARIES})

#Specify the executables for building 
add_executable(ikfom_test src/ikfom_test.cpp)
# Link against required libraries
target_link_libraries(ikfom_test ${catkin_LIBRARIES} ${PYTHON_LIBRARIES})
target_include_directories(ikfom_test PRIVATE ${PYTHON_INCLUDE_DIRS})

#Specify the executables for building 
add_executable(ikfom_uav_test src/ikfom_uav_test.cpp)
# Link against required libraries
target_link_libraries(ikfom_uav_test ${catkin_LIBRARIES} ${PYTHON_LIBRARIES})
target_include_directories(ikfom_uav_test PRIVATE ${PYTHON_INCLUDE_DIRS})