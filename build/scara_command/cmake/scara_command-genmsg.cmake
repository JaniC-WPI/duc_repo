# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "scara_command: 0 messages, 11 services")

set(MSG_I_FLAGS "-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(scara_command_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinFK.srv" NAME_WE)
add_custom_target(_scara_command_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "scara_command" "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinFK.srv" ""
)

get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinIK.srv" NAME_WE)
add_custom_target(_scara_command_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "scara_command" "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinIK.srv" ""
)

get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinFK.srv" NAME_WE)
add_custom_target(_scara_command_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "scara_command" "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinFK.srv" ""
)

get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinIK.srv" NAME_WE)
add_custom_target(_scara_command_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "scara_command" "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinIK.srv" ""
)

get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SetJointRef.srv" NAME_WE)
add_custom_target(_scara_command_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "scara_command" "/home/jc-merlab/duc_repo/src/scara_command/srv/SetJointRef.srv" ""
)

get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianVel.srv" NAME_WE)
add_custom_target(_scara_command_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "scara_command" "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianVel.srv" ""
)

get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianPos.srv" NAME_WE)
add_custom_target(_scara_command_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "scara_command" "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianPos.srv" ""
)

get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelFK.srv" NAME_WE)
add_custom_target(_scara_command_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "scara_command" "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelFK.srv" ""
)

get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelIK.srv" NAME_WE)
add_custom_target(_scara_command_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "scara_command" "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelIK.srv" ""
)

get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraHomoMatrix.srv" NAME_WE)
add_custom_target(_scara_command_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "scara_command" "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraHomoMatrix.srv" "std_msgs/Float32MultiArray:std_msgs/MultiArrayDimension:std_msgs/MultiArrayLayout"
)

get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SwitchControl.srv" NAME_WE)
add_custom_target(_scara_command_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "scara_command" "/home/jc-merlab/duc_repo/src/scara_command/srv/SwitchControl.srv" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages

### Generating Services
_generate_srv_cpp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinFK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/scara_command
)
_generate_srv_cpp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinIK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/scara_command
)
_generate_srv_cpp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinFK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/scara_command
)
_generate_srv_cpp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinIK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/scara_command
)
_generate_srv_cpp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/SetJointRef.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/scara_command
)
_generate_srv_cpp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianVel.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/scara_command
)
_generate_srv_cpp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianPos.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/scara_command
)
_generate_srv_cpp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelFK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/scara_command
)
_generate_srv_cpp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelIK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/scara_command
)
_generate_srv_cpp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraHomoMatrix.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Float32MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/scara_command
)
_generate_srv_cpp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/SwitchControl.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/scara_command
)

### Generating Module File
_generate_module_cpp(scara_command
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/scara_command
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(scara_command_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(scara_command_generate_messages scara_command_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinFK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_cpp _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinIK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_cpp _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinFK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_cpp _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinIK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_cpp _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SetJointRef.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_cpp _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianVel.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_cpp _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianPos.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_cpp _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelFK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_cpp _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelIK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_cpp _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraHomoMatrix.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_cpp _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SwitchControl.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_cpp _scara_command_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(scara_command_gencpp)
add_dependencies(scara_command_gencpp scara_command_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS scara_command_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages

### Generating Services
_generate_srv_eus(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinFK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/scara_command
)
_generate_srv_eus(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinIK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/scara_command
)
_generate_srv_eus(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinFK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/scara_command
)
_generate_srv_eus(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinIK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/scara_command
)
_generate_srv_eus(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/SetJointRef.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/scara_command
)
_generate_srv_eus(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianVel.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/scara_command
)
_generate_srv_eus(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianPos.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/scara_command
)
_generate_srv_eus(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelFK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/scara_command
)
_generate_srv_eus(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelIK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/scara_command
)
_generate_srv_eus(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraHomoMatrix.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Float32MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/scara_command
)
_generate_srv_eus(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/SwitchControl.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/scara_command
)

### Generating Module File
_generate_module_eus(scara_command
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/scara_command
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(scara_command_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(scara_command_generate_messages scara_command_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinFK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_eus _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinIK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_eus _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinFK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_eus _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinIK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_eus _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SetJointRef.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_eus _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianVel.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_eus _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianPos.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_eus _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelFK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_eus _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelIK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_eus _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraHomoMatrix.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_eus _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SwitchControl.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_eus _scara_command_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(scara_command_geneus)
add_dependencies(scara_command_geneus scara_command_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS scara_command_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages

### Generating Services
_generate_srv_lisp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinFK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/scara_command
)
_generate_srv_lisp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinIK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/scara_command
)
_generate_srv_lisp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinFK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/scara_command
)
_generate_srv_lisp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinIK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/scara_command
)
_generate_srv_lisp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/SetJointRef.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/scara_command
)
_generate_srv_lisp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianVel.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/scara_command
)
_generate_srv_lisp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianPos.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/scara_command
)
_generate_srv_lisp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelFK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/scara_command
)
_generate_srv_lisp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelIK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/scara_command
)
_generate_srv_lisp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraHomoMatrix.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Float32MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/scara_command
)
_generate_srv_lisp(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/SwitchControl.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/scara_command
)

### Generating Module File
_generate_module_lisp(scara_command
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/scara_command
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(scara_command_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(scara_command_generate_messages scara_command_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinFK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_lisp _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinIK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_lisp _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinFK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_lisp _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinIK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_lisp _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SetJointRef.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_lisp _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianVel.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_lisp _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianPos.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_lisp _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelFK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_lisp _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelIK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_lisp _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraHomoMatrix.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_lisp _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SwitchControl.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_lisp _scara_command_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(scara_command_genlisp)
add_dependencies(scara_command_genlisp scara_command_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS scara_command_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages

### Generating Services
_generate_srv_nodejs(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinFK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/scara_command
)
_generate_srv_nodejs(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinIK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/scara_command
)
_generate_srv_nodejs(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinFK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/scara_command
)
_generate_srv_nodejs(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinIK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/scara_command
)
_generate_srv_nodejs(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/SetJointRef.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/scara_command
)
_generate_srv_nodejs(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianVel.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/scara_command
)
_generate_srv_nodejs(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianPos.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/scara_command
)
_generate_srv_nodejs(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelFK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/scara_command
)
_generate_srv_nodejs(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelIK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/scara_command
)
_generate_srv_nodejs(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraHomoMatrix.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Float32MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/scara_command
)
_generate_srv_nodejs(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/SwitchControl.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/scara_command
)

### Generating Module File
_generate_module_nodejs(scara_command
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/scara_command
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(scara_command_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(scara_command_generate_messages scara_command_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinFK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_nodejs _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinIK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_nodejs _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinFK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_nodejs _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinIK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_nodejs _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SetJointRef.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_nodejs _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianVel.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_nodejs _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianPos.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_nodejs _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelFK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_nodejs _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelIK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_nodejs _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraHomoMatrix.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_nodejs _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SwitchControl.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_nodejs _scara_command_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(scara_command_gennodejs)
add_dependencies(scara_command_gennodejs scara_command_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS scara_command_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages

### Generating Services
_generate_srv_py(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinFK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/scara_command
)
_generate_srv_py(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinIK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/scara_command
)
_generate_srv_py(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinFK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/scara_command
)
_generate_srv_py(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinIK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/scara_command
)
_generate_srv_py(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/SetJointRef.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/scara_command
)
_generate_srv_py(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianVel.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/scara_command
)
_generate_srv_py(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianPos.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/scara_command
)
_generate_srv_py(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelFK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/scara_command
)
_generate_srv_py(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelIK.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/scara_command
)
_generate_srv_py(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraHomoMatrix.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Float32MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/scara_command
)
_generate_srv_py(scara_command
  "/home/jc-merlab/duc_repo/src/scara_command/srv/SwitchControl.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/scara_command
)

### Generating Module File
_generate_module_py(scara_command
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/scara_command
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(scara_command_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(scara_command_generate_messages scara_command_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinFK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_py _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinIK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_py _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinFK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_py _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinIK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_py _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SetJointRef.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_py _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianVel.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_py _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianPos.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_py _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelFK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_py _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelIK.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_py _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraHomoMatrix.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_py _scara_command_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jc-merlab/duc_repo/src/scara_command/srv/SwitchControl.srv" NAME_WE)
add_dependencies(scara_command_generate_messages_py _scara_command_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(scara_command_genpy)
add_dependencies(scara_command_genpy scara_command_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS scara_command_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/scara_command)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/scara_command
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(scara_command_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/scara_command)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/scara_command
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(scara_command_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/scara_command)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/scara_command
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(scara_command_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/scara_command)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/scara_command
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(scara_command_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/scara_command)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/scara_command\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/scara_command
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(scara_command_generate_messages_py std_msgs_generate_messages_py)
endif()
