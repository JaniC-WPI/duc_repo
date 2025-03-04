cmake_minimum_required(VERSION 3.0.2)
project(panda_test)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  rosbag
  cv_bridge
  image_transport
  message_generation
  sensor_msgs
)

find_package(Eigen3)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
add_service_files(
FILES
energyFuncMsg.srv
energyFuncMsg_plan.srv
franka_control_points.srv
franka_bin_img.srv
franka_cp_goal.srv
# dream_goal_img.srv
# dream_kp.srv
# dream_img.srv
dl_goal_img.srv
dl_plan_sim_img.srv
dl_img.srv
dl_sim_img.srv
vel_start.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs  # Or other packages containing msgs
 )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

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
#  INCLUDE_DIRS include
  LIBRARIES panda_test
#  CATKIN_DEPENDS roscpp rospy
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/panda_test.cpp
# )

# add_executable(franka_energy_func_node src/franka_compute_energy_functional.cpp)
# add_dependencies(franka_energy_func_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(franka_energy_func_node ${catkin_LIBRARIES})

# add_executable(franka_shape_energy_func_node src/franka_shape_compute_energy_functional.cpp)
# add_dependencies(franka_shape_energy_func_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(franka_shape_energy_func_node ${catkin_LIBRARIES})

# add_executable(servo_controller_node src/servo_controller.cpp)
# add_dependencies(servo_controller_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(servo_controller_node ${catkin_LIBRARIES})

# add_executable(vsbot_record_node src/recorder.cpp)
# add_dependencies(vsbot_record_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(vsbot_record_node ${catkin_LIBRARIES})

# add_executable(redundant_record_node src/redundant_recorder.cpp)
# add_dependencies(redundant_record_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(redundant_record_node ${catkin_LIBRARIES})

# add_executable(vsbot_vis_node src/visualization.cpp)
# add_dependencies(vsbot_vis_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(vsbot_vis_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

# add_executable(shape_servo_controller_node src/shape_servo_controller.cpp)
# add_dependencies(shape_servo_controller_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(shape_servo_controller_node ${catkin_LIBRARIES} ${ARMADILL_LIBRARIES})

# add_executable(time_recorder_node src/time_recorder.cpp)
# add_dependencies(time_recorder_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(time_recorder_node ${catkin_LIBRARIES})


add_executable(franka_joint_pub src/franka_joint_pub.cpp)
add_dependencies(franka_joint_pub ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(franka_joint_pub ${catkin_LIBRARIES})

# add_executable(franka_adaptive_vs_node src/franka_adaptive_vs.cpp)
# add_dependencies(franka_adaptive_vs_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(franka_adaptive_vs_node ${catkin_LIBRARIES})


add_executable(franka_shape_record_node src/franka_shape_vs_recorder.cpp)
add_dependencies(franka_shape_record_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(franka_shape_record_node ${catkin_LIBRARIES})

# add_executable(franka_shape_controller_node src/franka_shape_vs.cpp)
# add_dependencies(franka_shape_controller_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(franka_shape_controller_node ${catkin_LIBRARIES})

# add_executable(redundant_shape_controller_node src/redundant_shape_servo_controller.cpp)
# add_dependencies(redundant_shape_controller_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(redundant_shape_controller_node ${catkin_LIBRARIES})

add_executable(franka_dl_controller_node src/franka_dl_vs.cpp)
add_dependencies(franka_dl_controller_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(franka_dl_controller_node ${catkin_LIBRARIES})

add_executable(franka_sim_controller_node src/franka_sim_vs.cpp)
add_dependencies(franka_sim_controller_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(franka_sim_controller_node ${catkin_LIBRARIES})

add_executable(franka_dl_energy_func_node src/franka_dl_compute_energy_functional.cpp)
add_dependencies(franka_dl_energy_func_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(franka_dl_energy_func_node ${catkin_LIBRARIES})

add_executable(franka_sim_energy_func_node src/franka_sim_compute_energy_functional.cpp)
add_dependencies(franka_sim_energy_func_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(franka_sim_energy_func_node ${catkin_LIBRARIES})

add_executable(franka_dl_record_node src/franka_dl_vs_recorder.cpp)
add_dependencies(franka_dl_record_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(franka_dl_record_node ${catkin_LIBRARIES})

add_executable(franka_sim_record_node src/franka_sim_vs_recorder.cpp)
add_dependencies(franka_sim_record_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(franka_sim_record_node ${catkin_LIBRARIES})

# add_executable(franka_multiref_shape_vs_node src/franka_multiref_shape_vs.cpp)
# add_dependencies(franka_multiref_shape_vs_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(franka_multiref_shape_vs_node ${catkin_LIBRARIES})

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/panda_test_node.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
#   scripts/my_python_script
  # scripts/image_capture.py
  # scripts/python_skeleton/astar.py
  # scripts/python_skeleton/franka_bin_svc_server.py
  # scripts/python_skeleton/franka_cp_svc_server.py
  # scripts/python_skeleton/franka_goal_bin_svc.py
  # scripts/python_skeleton/franka_goal_cp_svc_rqst.py
  # scripts/python_skeleton/franka_goal_cp_svc.py
  # scripts/python_skeleton/franka_goal_img_pub.py
  # scripts/python_skeleton/franka_shape_plot.py

  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_panda_test.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)