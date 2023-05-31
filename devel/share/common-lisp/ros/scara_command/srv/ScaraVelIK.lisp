; Auto-generated. Do not edit!


(cl:in-package scara_command-srv)


;//! \htmlinclude ScaraVelIK-request.msg.html

(cl:defclass <ScaraVelIK-request> (roslisp-msg-protocol:ros-message)
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
    :initform 0.0)
   (Vx
    :reader Vx
    :initarg :Vx
    :type cl:float
    :initform 0.0)
   (Vy
    :reader Vy
    :initarg :Vy
    :type cl:float
    :initform 0.0)
   (Vz
    :reader Vz
    :initarg :Vz
    :type cl:float
    :initform 0.0)
   (Wx
    :reader Wx
    :initarg :Wx
    :type cl:float
    :initform 0.0)
   (Wy
    :reader Wy
    :initarg :Wy
    :type cl:float
    :initform 0.0)
   (Wz
    :reader Wz
    :initarg :Wz
    :type cl:float
    :initform 0.0))
)

(cl:defclass ScaraVelIK-request (<ScaraVelIK-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ScaraVelIK-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ScaraVelIK-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name scara_command-srv:<ScaraVelIK-request> is deprecated: use scara_command-srv:ScaraVelIK-request instead.")))

(cl:ensure-generic-function 'q1-val :lambda-list '(m))
(cl:defmethod q1-val ((m <ScaraVelIK-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:q1-val is deprecated.  Use scara_command-srv:q1 instead.")
  (q1 m))

(cl:ensure-generic-function 'q2-val :lambda-list '(m))
(cl:defmethod q2-val ((m <ScaraVelIK-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:q2-val is deprecated.  Use scara_command-srv:q2 instead.")
  (q2 m))

(cl:ensure-generic-function 'q3-val :lambda-list '(m))
(cl:defmethod q3-val ((m <ScaraVelIK-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:q3-val is deprecated.  Use scara_command-srv:q3 instead.")
  (q3 m))

(cl:ensure-generic-function 'Vx-val :lambda-list '(m))
(cl:defmethod Vx-val ((m <ScaraVelIK-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:Vx-val is deprecated.  Use scara_command-srv:Vx instead.")
  (Vx m))

(cl:ensure-generic-function 'Vy-val :lambda-list '(m))
(cl:defmethod Vy-val ((m <ScaraVelIK-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:Vy-val is deprecated.  Use scara_command-srv:Vy instead.")
  (Vy m))

(cl:ensure-generic-function 'Vz-val :lambda-list '(m))
(cl:defmethod Vz-val ((m <ScaraVelIK-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:Vz-val is deprecated.  Use scara_command-srv:Vz instead.")
  (Vz m))

(cl:ensure-generic-function 'Wx-val :lambda-list '(m))
(cl:defmethod Wx-val ((m <ScaraVelIK-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:Wx-val is deprecated.  Use scara_command-srv:Wx instead.")
  (Wx m))

(cl:ensure-generic-function 'Wy-val :lambda-list '(m))
(cl:defmethod Wy-val ((m <ScaraVelIK-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:Wy-val is deprecated.  Use scara_command-srv:Wy instead.")
  (Wy m))

(cl:ensure-generic-function 'Wz-val :lambda-list '(m))
(cl:defmethod Wz-val ((m <ScaraVelIK-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:Wz-val is deprecated.  Use scara_command-srv:Wz instead.")
  (Wz m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ScaraVelIK-request>) ostream)
  "Serializes a message object of type '<ScaraVelIK-request>"
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
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'Vx))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'Vy))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'Vz))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'Wx))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'Wy))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'Wz))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ScaraVelIK-request>) istream)
  "Deserializes a message object of type '<ScaraVelIK-request>"
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
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'Vx) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'Vy) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'Vz) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'Wx) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'Wy) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'Wz) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ScaraVelIK-request>)))
  "Returns string type for a service object of type '<ScaraVelIK-request>"
  "scara_command/ScaraVelIKRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ScaraVelIK-request)))
  "Returns string type for a service object of type 'ScaraVelIK-request"
  "scara_command/ScaraVelIKRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ScaraVelIK-request>)))
  "Returns md5sum for a message object of type '<ScaraVelIK-request>"
  "273c8e596b5ffa32d13cf21237d3aaf2")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ScaraVelIK-request)))
  "Returns md5sum for a message object of type 'ScaraVelIK-request"
  "273c8e596b5ffa32d13cf21237d3aaf2")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ScaraVelIK-request>)))
  "Returns full string definition for message of type '<ScaraVelIK-request>"
  (cl:format cl:nil "float32 q1~%float32 q2~%float32 q3~%float32 Vx~%float32 Vy~%float32 Vz~%float32 Wx~%float32 Wy~%float32 Wz~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ScaraVelIK-request)))
  "Returns full string definition for message of type 'ScaraVelIK-request"
  (cl:format cl:nil "float32 q1~%float32 q2~%float32 q3~%float32 Vx~%float32 Vy~%float32 Vz~%float32 Wx~%float32 Wy~%float32 Wz~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ScaraVelIK-request>))
  (cl:+ 0
     4
     4
     4
     4
     4
     4
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ScaraVelIK-request>))
  "Converts a ROS message object to a list"
  (cl:list 'ScaraVelIK-request
    (cl:cons ':q1 (q1 msg))
    (cl:cons ':q2 (q2 msg))
    (cl:cons ':q3 (q3 msg))
    (cl:cons ':Vx (Vx msg))
    (cl:cons ':Vy (Vy msg))
    (cl:cons ':Vz (Vz msg))
    (cl:cons ':Wx (Wx msg))
    (cl:cons ':Wy (Wy msg))
    (cl:cons ':Wz (Wz msg))
))
;//! \htmlinclude ScaraVelIK-response.msg.html

(cl:defclass <ScaraVelIK-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil)
   (q1_dot
    :reader q1_dot
    :initarg :q1_dot
    :type cl:float
    :initform 0.0)
   (q2_dot
    :reader q2_dot
    :initarg :q2_dot
    :type cl:float
    :initform 0.0)
   (q3_dot
    :reader q3_dot
    :initarg :q3_dot
    :type cl:float
    :initform 0.0))
)

(cl:defclass ScaraVelIK-response (<ScaraVelIK-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ScaraVelIK-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ScaraVelIK-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name scara_command-srv:<ScaraVelIK-response> is deprecated: use scara_command-srv:ScaraVelIK-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <ScaraVelIK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:success-val is deprecated.  Use scara_command-srv:success instead.")
  (success m))

(cl:ensure-generic-function 'q1_dot-val :lambda-list '(m))
(cl:defmethod q1_dot-val ((m <ScaraVelIK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:q1_dot-val is deprecated.  Use scara_command-srv:q1_dot instead.")
  (q1_dot m))

(cl:ensure-generic-function 'q2_dot-val :lambda-list '(m))
(cl:defmethod q2_dot-val ((m <ScaraVelIK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:q2_dot-val is deprecated.  Use scara_command-srv:q2_dot instead.")
  (q2_dot m))

(cl:ensure-generic-function 'q3_dot-val :lambda-list '(m))
(cl:defmethod q3_dot-val ((m <ScaraVelIK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:q3_dot-val is deprecated.  Use scara_command-srv:q3_dot instead.")
  (q3_dot m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ScaraVelIK-response>) ostream)
  "Serializes a message object of type '<ScaraVelIK-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'q1_dot))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'q2_dot))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'q3_dot))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ScaraVelIK-response>) istream)
  "Deserializes a message object of type '<ScaraVelIK-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'q1_dot) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'q2_dot) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'q3_dot) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ScaraVelIK-response>)))
  "Returns string type for a service object of type '<ScaraVelIK-response>"
  "scara_command/ScaraVelIKResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ScaraVelIK-response)))
  "Returns string type for a service object of type 'ScaraVelIK-response"
  "scara_command/ScaraVelIKResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ScaraVelIK-response>)))
  "Returns md5sum for a message object of type '<ScaraVelIK-response>"
  "273c8e596b5ffa32d13cf21237d3aaf2")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ScaraVelIK-response)))
  "Returns md5sum for a message object of type 'ScaraVelIK-response"
  "273c8e596b5ffa32d13cf21237d3aaf2")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ScaraVelIK-response>)))
  "Returns full string definition for message of type '<ScaraVelIK-response>"
  (cl:format cl:nil "bool success~%float32 q1_dot~%float32 q2_dot~%float32 q3_dot~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ScaraVelIK-response)))
  "Returns full string definition for message of type 'ScaraVelIK-response"
  (cl:format cl:nil "bool success~%float32 q1_dot~%float32 q2_dot~%float32 q3_dot~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ScaraVelIK-response>))
  (cl:+ 0
     1
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ScaraVelIK-response>))
  "Converts a ROS message object to a list"
  (cl:list 'ScaraVelIK-response
    (cl:cons ':success (success msg))
    (cl:cons ':q1_dot (q1_dot msg))
    (cl:cons ':q2_dot (q2_dot msg))
    (cl:cons ':q3_dot (q3_dot msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'ScaraVelIK)))
  'ScaraVelIK-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'ScaraVelIK)))
  'ScaraVelIK-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ScaraVelIK)))
  "Returns string type for a service object of type '<ScaraVelIK>"
  "scara_command/ScaraVelIK")