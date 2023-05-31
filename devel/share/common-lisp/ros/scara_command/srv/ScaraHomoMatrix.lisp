; Auto-generated. Do not edit!


(cl:in-package scara_command-srv)


;//! \htmlinclude ScaraHomoMatrix-request.msg.html

(cl:defclass <ScaraHomoMatrix-request> (roslisp-msg-protocol:ros-message)
  ((q1
    :reader q1
    :initarg :q1
    :type cl:float
    :initform 0.0)
   (q2
    :reader q2
    :initarg :q2
    :type cl:float
    :initform 0.0)
   (q3
    :reader q3
    :initarg :q3
    :type cl:float
    :initform 0.0))
)

(cl:defclass ScaraHomoMatrix-request (<ScaraHomoMatrix-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ScaraHomoMatrix-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ScaraHomoMatrix-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name scara_command-srv:<ScaraHomoMatrix-request> is deprecated: use scara_command-srv:ScaraHomoMatrix-request instead.")))

(cl:ensure-generic-function 'q1-val :lambda-list '(m))
(cl:defmethod q1-val ((m <ScaraHomoMatrix-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:q1-val is deprecated.  Use scara_command-srv:q1 instead.")
  (q1 m))

(cl:ensure-generic-function 'q2-val :lambda-list '(m))
(cl:defmethod q2-val ((m <ScaraHomoMatrix-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:q2-val is deprecated.  Use scara_command-srv:q2 instead.")
  (q2 m))

(cl:ensure-generic-function 'q3-val :lambda-list '(m))
(cl:defmethod q3-val ((m <ScaraHomoMatrix-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:q3-val is deprecated.  Use scara_command-srv:q3 instead.")
  (q3 m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ScaraHomoMatrix-request>) ostream)
  "Serializes a message object of type '<ScaraHomoMatrix-request>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'q1))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'q2))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'q3))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ScaraHomoMatrix-request>) istream)
  "Deserializes a message object of type '<ScaraHomoMatrix-request>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'q1) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'q2) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'q3) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ScaraHomoMatrix-request>)))
  "Returns string type for a service object of type '<ScaraHomoMatrix-request>"
  "scara_command/ScaraHomoMatrixRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ScaraHomoMatrix-request)))
  "Returns string type for a service object of type 'ScaraHomoMatrix-request"
  "scara_command/ScaraHomoMatrixRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ScaraHomoMatrix-request>)))
  "Returns md5sum for a message object of type '<ScaraHomoMatrix-request>"
  "b04a28e6794c58d88ef84b160d32201a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ScaraHomoMatrix-request)))
  "Returns md5sum for a message object of type 'ScaraHomoMatrix-request"
  "b04a28e6794c58d88ef84b160d32201a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ScaraHomoMatrix-request>)))
  "Returns full string definition for message of type '<ScaraHomoMatrix-request>"
  (cl:format cl:nil "float32 q1~%float32 q2~%float32 q3~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ScaraHomoMatrix-request)))
  "Returns full string definition for message of type 'ScaraHomoMatrix-request"
  (cl:format cl:nil "float32 q1~%float32 q2~%float32 q3~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ScaraHomoMatrix-request>))
  (cl:+ 0
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ScaraHomoMatrix-request>))
  "Converts a ROS message object to a list"
  (cl:list 'ScaraHomoMatrix-request
    (cl:cons ':q1 (q1 msg))
    (cl:cons ':q2 (q2 msg))
    (cl:cons ':q3 (q3 msg))
))
;//! \htmlinclude ScaraHomoMatrix-response.msg.html

(cl:defclass <ScaraHomoMatrix-response> (roslisp-msg-protocol:ros-message)
  ((A1
    :reader A1
    :initarg :A1
    :type std_msgs-msg:Float32MultiArray
    :initform (cl:make-instance 'std_msgs-msg:Float32MultiArray))
   (A2
    :reader A2
    :initarg :A2
    :type std_msgs-msg:Float32MultiArray
    :initform (cl:make-instance 'std_msgs-msg:Float32MultiArray))
   (A3
    :reader A3
    :initarg :A3
    :type std_msgs-msg:Float32MultiArray
    :initform (cl:make-instance 'std_msgs-msg:Float32MultiArray)))
)

(cl:defclass ScaraHomoMatrix-response (<ScaraHomoMatrix-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ScaraHomoMatrix-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ScaraHomoMatrix-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name scara_command-srv:<ScaraHomoMatrix-response> is deprecated: use scara_command-srv:ScaraHomoMatrix-response instead.")))

(cl:ensure-generic-function 'A1-val :lambda-list '(m))
(cl:defmethod A1-val ((m <ScaraHomoMatrix-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:A1-val is deprecated.  Use scara_command-srv:A1 instead.")
  (A1 m))

(cl:ensure-generic-function 'A2-val :lambda-list '(m))
(cl:defmethod A2-val ((m <ScaraHomoMatrix-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:A2-val is deprecated.  Use scara_command-srv:A2 instead.")
  (A2 m))

(cl:ensure-generic-function 'A3-val :lambda-list '(m))
(cl:defmethod A3-val ((m <ScaraHomoMatrix-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:A3-val is deprecated.  Use scara_command-srv:A3 instead.")
  (A3 m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ScaraHomoMatrix-response>) ostream)
  "Serializes a message object of type '<ScaraHomoMatrix-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'A1) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'A2) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'A3) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ScaraHomoMatrix-response>) istream)
  "Deserializes a message object of type '<ScaraHomoMatrix-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'A1) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'A2) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'A3) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ScaraHomoMatrix-response>)))
  "Returns string type for a service object of type '<ScaraHomoMatrix-response>"
  "scara_command/ScaraHomoMatrixResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ScaraHomoMatrix-response)))
  "Returns string type for a service object of type 'ScaraHomoMatrix-response"
  "scara_command/ScaraHomoMatrixResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ScaraHomoMatrix-response>)))
  "Returns md5sum for a message object of type '<ScaraHomoMatrix-response>"
  "b04a28e6794c58d88ef84b160d32201a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ScaraHomoMatrix-response)))
  "Returns md5sum for a message object of type 'ScaraHomoMatrix-response"
  "b04a28e6794c58d88ef84b160d32201a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ScaraHomoMatrix-response>)))
  "Returns full string definition for message of type '<ScaraHomoMatrix-response>"
  (cl:format cl:nil "std_msgs/Float32MultiArray A1~%std_msgs/Float32MultiArray A2~%std_msgs/Float32MultiArray A3~%~%~%================================================================================~%MSG: std_msgs/Float32MultiArray~%# Please look at the MultiArrayLayout message definition for~%# documentation on all multiarrays.~%~%MultiArrayLayout  layout        # specification of data layout~%float32[]         data          # array of data~%~%~%================================================================================~%MSG: std_msgs/MultiArrayLayout~%# The multiarray declares a generic multi-dimensional array of a~%# particular data type.  Dimensions are ordered from outer most~%# to inner most.~%~%MultiArrayDimension[] dim # Array of dimension properties~%uint32 data_offset        # padding elements at front of data~%~%# Accessors should ALWAYS be written in terms of dimension stride~%# and specified outer-most dimension first.~%# ~%# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]~%#~%# A standard, 3-channel 640x480 image with interleaved color channels~%# would be specified as:~%#~%# dim[0].label  = \"height\"~%# dim[0].size   = 480~%# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)~%# dim[1].label  = \"width\"~%# dim[1].size   = 640~%# dim[1].stride = 3*640 = 1920~%# dim[2].label  = \"channel\"~%# dim[2].size   = 3~%# dim[2].stride = 3~%#~%# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.~%~%================================================================================~%MSG: std_msgs/MultiArrayDimension~%string label   # label of given dimension~%uint32 size    # size of given dimension (in type units)~%uint32 stride  # stride of given dimension~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ScaraHomoMatrix-response)))
  "Returns full string definition for message of type 'ScaraHomoMatrix-response"
  (cl:format cl:nil "std_msgs/Float32MultiArray A1~%std_msgs/Float32MultiArray A2~%std_msgs/Float32MultiArray A3~%~%~%================================================================================~%MSG: std_msgs/Float32MultiArray~%# Please look at the MultiArrayLayout message definition for~%# documentation on all multiarrays.~%~%MultiArrayLayout  layout        # specification of data layout~%float32[]         data          # array of data~%~%~%================================================================================~%MSG: std_msgs/MultiArrayLayout~%# The multiarray declares a generic multi-dimensional array of a~%# particular data type.  Dimensions are ordered from outer most~%# to inner most.~%~%MultiArrayDimension[] dim # Array of dimension properties~%uint32 data_offset        # padding elements at front of data~%~%# Accessors should ALWAYS be written in terms of dimension stride~%# and specified outer-most dimension first.~%# ~%# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]~%#~%# A standard, 3-channel 640x480 image with interleaved color channels~%# would be specified as:~%#~%# dim[0].label  = \"height\"~%# dim[0].size   = 480~%# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)~%# dim[1].label  = \"width\"~%# dim[1].size   = 640~%# dim[1].stride = 3*640 = 1920~%# dim[2].label  = \"channel\"~%# dim[2].size   = 3~%# dim[2].stride = 3~%#~%# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.~%~%================================================================================~%MSG: std_msgs/MultiArrayDimension~%string label   # label of given dimension~%uint32 size    # size of given dimension (in type units)~%uint32 stride  # stride of given dimension~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ScaraHomoMatrix-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'A1))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'A2))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'A3))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ScaraHomoMatrix-response>))
  "Converts a ROS message object to a list"
  (cl:list 'ScaraHomoMatrix-response
    (cl:cons ':A1 (A1 msg))
    (cl:cons ':A2 (A2 msg))
    (cl:cons ':A3 (A3 msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'ScaraHomoMatrix)))
  'ScaraHomoMatrix-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'ScaraHomoMatrix)))
  'ScaraHomoMatrix-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ScaraHomoMatrix)))
  "Returns string type for a service object of type '<ScaraHomoMatrix>"
  "scara_command/ScaraHomoMatrix")