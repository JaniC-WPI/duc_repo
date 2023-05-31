;; Auto-generated. Do not edit!


(when (boundp 'scara_command::ScaraHomoMatrix)
  (if (not (find-package "SCARA_COMMAND"))
    (make-package "SCARA_COMMAND"))
  (shadow 'ScaraHomoMatrix (find-package "SCARA_COMMAND")))
(unless (find-package "SCARA_COMMAND::SCARAHOMOMATRIX")
  (make-package "SCARA_COMMAND::SCARAHOMOMATRIX"))
(unless (find-package "SCARA_COMMAND::SCARAHOMOMATRIXREQUEST")
  (make-package "SCARA_COMMAND::SCARAHOMOMATRIXREQUEST"))
(unless (find-package "SCARA_COMMAND::SCARAHOMOMATRIXRESPONSE")
  (make-package "SCARA_COMMAND::SCARAHOMOMATRIXRESPONSE"))

(in-package "ROS")



(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass scara_command::ScaraHomoMatrixRequest
  :super ros::object
  :slots (_q1 _q2 _q3 ))

(defmethod scara_command::ScaraHomoMatrixRequest
  (:init
   (&key
    ((:q1 __q1) 0.0)
    ((:q2 __q2) 0.0)
    ((:q3 __q3) 0.0)
    )
   (send-super :init)
   (setq _q1 (float __q1))
   (setq _q2 (float __q2))
   (setq _q3 (float __q3))
   self)
  (:q1
   (&optional __q1)
   (if __q1 (setq _q1 __q1)) _q1)
  (:q2
   (&optional __q2)
   (if __q2 (setq _q2 __q2)) _q2)
  (:q3
   (&optional __q3)
   (if __q3 (setq _q3 __q3)) _q3)
  (:serialization-length
   ()
   (+
    ;; float32 _q1
    4
    ;; float32 _q2
    4
    ;; float32 _q3
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _q1
       (sys::poke _q1 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _q2
       (sys::poke _q2 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _q3
       (sys::poke _q3 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _q1
     (setq _q1 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _q2
     (setq _q2 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _q3
     (setq _q3 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(defclass scara_command::ScaraHomoMatrixResponse
  :super ros::object
  :slots (_A1 _A2 _A3 ))

(defmethod scara_command::ScaraHomoMatrixResponse
  (:init
   (&key
    ((:A1 __A1) (instance std_msgs::Float32MultiArray :init))
    ((:A2 __A2) (instance std_msgs::Float32MultiArray :init))
    ((:A3 __A3) (instance std_msgs::Float32MultiArray :init))
    )
   (send-super :init)
   (setq _A1 __A1)
   (setq _A2 __A2)
   (setq _A3 __A3)
   self)
  (:A1
   (&rest __A1)
   (if (keywordp (car __A1))
       (send* _A1 __A1)
     (progn
       (if __A1 (setq _A1 (car __A1)))
       _A1)))
  (:A2
   (&rest __A2)
   (if (keywordp (car __A2))
       (send* _A2 __A2)
     (progn
       (if __A2 (setq _A2 (car __A2)))
       _A2)))
  (:A3
   (&rest __A3)
   (if (keywordp (car __A3))
       (send* _A3 __A3)
     (progn
       (if __A3 (setq _A3 (car __A3)))
       _A3)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Float32MultiArray _A1
    (send _A1 :serialization-length)
    ;; std_msgs/Float32MultiArray _A2
    (send _A2 :serialization-length)
    ;; std_msgs/Float32MultiArray _A3
    (send _A3 :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Float32MultiArray _A1
       (send _A1 :serialize s)
     ;; std_msgs/Float32MultiArray _A2
       (send _A2 :serialize s)
     ;; std_msgs/Float32MultiArray _A3
       (send _A3 :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Float32MultiArray _A1
     (send _A1 :deserialize buf ptr-) (incf ptr- (send _A1 :serialization-length))
   ;; std_msgs/Float32MultiArray _A2
     (send _A2 :deserialize buf ptr-) (incf ptr- (send _A2 :serialization-length))
   ;; std_msgs/Float32MultiArray _A3
     (send _A3 :deserialize buf ptr-) (incf ptr- (send _A3 :serialization-length))
   ;;
   self)
  )

(defclass scara_command::ScaraHomoMatrix
  :super ros::object
  :slots ())

(setf (get scara_command::ScaraHomoMatrix :md5sum-) "b04a28e6794c58d88ef84b160d32201a")
(setf (get scara_command::ScaraHomoMatrix :datatype-) "scara_command/ScaraHomoMatrix")
(setf (get scara_command::ScaraHomoMatrix :request) scara_command::ScaraHomoMatrixRequest)
(setf (get scara_command::ScaraHomoMatrix :response) scara_command::ScaraHomoMatrixResponse)

(defmethod scara_command::ScaraHomoMatrixRequest
  (:response () (instance scara_command::ScaraHomoMatrixResponse :init)))

(setf (get scara_command::ScaraHomoMatrixRequest :md5sum-) "b04a28e6794c58d88ef84b160d32201a")
(setf (get scara_command::ScaraHomoMatrixRequest :datatype-) "scara_command/ScaraHomoMatrixRequest")
(setf (get scara_command::ScaraHomoMatrixRequest :definition-)
      "float32 q1
float32 q2
float32 q3
---
std_msgs/Float32MultiArray A1
std_msgs/Float32MultiArray A2
std_msgs/Float32MultiArray A3


================================================================================
MSG: std_msgs/Float32MultiArray
# Please look at the MultiArrayLayout message definition for
# documentation on all multiarrays.

MultiArrayLayout  layout        # specification of data layout
float32[]         data          # array of data


================================================================================
MSG: std_msgs/MultiArrayLayout
# The multiarray declares a generic multi-dimensional array of a
# particular data type.  Dimensions are ordered from outer most
# to inner most.

MultiArrayDimension[] dim # Array of dimension properties
uint32 data_offset        # padding elements at front of data

# Accessors should ALWAYS be written in terms of dimension stride
# and specified outer-most dimension first.
# 
# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]
#
# A standard, 3-channel 640x480 image with interleaved color channels
# would be specified as:
#
# dim[0].label  = \"height\"
# dim[0].size   = 480
# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)
# dim[1].label  = \"width\"
# dim[1].size   = 640
# dim[1].stride = 3*640 = 1920
# dim[2].label  = \"channel\"
# dim[2].size   = 3
# dim[2].stride = 3
#
# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.

================================================================================
MSG: std_msgs/MultiArrayDimension
string label   # label of given dimension
uint32 size    # size of given dimension (in type units)
")

(setf (get scara_command::ScaraHomoMatrixResponse :md5sum-) "b04a28e6794c58d88ef84b160d32201a")
(setf (get scara_command::ScaraHomoMatrixResponse :datatype-) "scara_command/ScaraHomoMatrixResponse")
(setf (get scara_command::ScaraHomoMatrixResponse :definition-)
      "float32 q1
float32 q2
float32 q3
---
std_msgs/Float32MultiArray A1
std_msgs/Float32MultiArray A2
std_msgs/Float32MultiArray A3


================================================================================
MSG: std_msgs/Float32MultiArray
# Please look at the MultiArrayLayout message definition for
# documentation on all multiarrays.

MultiArrayLayout  layout        # specification of data layout
float32[]         data          # array of data


================================================================================
MSG: std_msgs/MultiArrayLayout
# The multiarray declares a generic multi-dimensional array of a
# particular data type.  Dimensions are ordered from outer most
# to inner most.

MultiArrayDimension[] dim # Array of dimension properties
uint32 data_offset        # padding elements at front of data

# Accessors should ALWAYS be written in terms of dimension stride
# and specified outer-most dimension first.
# 
# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]
#
# A standard, 3-channel 640x480 image with interleaved color channels
# would be specified as:
#
# dim[0].label  = \"height\"
# dim[0].size   = 480
# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)
# dim[1].label  = \"width\"
# dim[1].size   = 640
# dim[1].stride = 3*640 = 1920
# dim[2].label  = \"channel\"
# dim[2].size   = 3
# dim[2].stride = 3
#
# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.

================================================================================
MSG: std_msgs/MultiArrayDimension
string label   # label of given dimension
uint32 size    # size of given dimension (in type units)
")



(provide :scara_command/ScaraHomoMatrix "b04a28e6794c58d88ef84b160d32201a")


