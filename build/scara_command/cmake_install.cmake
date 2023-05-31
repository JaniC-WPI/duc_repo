# Install script for directory: /home/jc-merlab/duc_repo/src/scara_command

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/jc-merlab/duc_repo/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/scara_command/srv" TYPE FILE FILES
    "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinFK.srv"
    "/home/jc-merlab/duc_repo/src/scara_command/srv/CheckKinIK.srv"
    "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinFK.srv"
    "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraKinIK.srv"
    "/home/jc-merlab/duc_repo/src/scara_command/srv/SetJointRef.srv"
    "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianVel.srv"
    "/home/jc-merlab/duc_repo/src/scara_command/srv/SetCartesianPos.srv"
    "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelFK.srv"
    "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraVelIK.srv"
    "/home/jc-merlab/duc_repo/src/scara_command/srv/ScaraHomoMatrix.srv"
    "/home/jc-merlab/duc_repo/src/scara_command/srv/SwitchControl.srv"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/scara_command/cmake" TYPE FILE FILES "/home/jc-merlab/duc_repo/build/scara_command/catkin_generated/installspace/scara_command-msg-paths.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/jc-merlab/duc_repo/devel/include/scara_command")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/jc-merlab/duc_repo/devel/share/roseus/ros/scara_command")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/home/jc-merlab/duc_repo/devel/share/common-lisp/ros/scara_command")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/home/jc-merlab/duc_repo/devel/share/gennodejs/ros/scara_command")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python3" -m compileall "/home/jc-merlab/duc_repo/devel/lib/python3/dist-packages/scara_command")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages" TYPE DIRECTORY FILES "/home/jc-merlab/duc_repo/devel/lib/python3/dist-packages/scara_command")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/jc-merlab/duc_repo/build/scara_command/catkin_generated/installspace/scara_command.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/scara_command/cmake" TYPE FILE FILES "/home/jc-merlab/duc_repo/build/scara_command/catkin_generated/installspace/scara_command-msg-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/scara_command/cmake" TYPE FILE FILES
    "/home/jc-merlab/duc_repo/build/scara_command/catkin_generated/installspace/scara_commandConfig.cmake"
    "/home/jc-merlab/duc_repo/build/scara_command/catkin_generated/installspace/scara_commandConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/scara_command" TYPE FILE FILES "/home/jc-merlab/duc_repo/src/scara_command/package.xml")
endif()

