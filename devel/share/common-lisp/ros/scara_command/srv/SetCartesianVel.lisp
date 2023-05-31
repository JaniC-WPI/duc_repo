; Auto-generated. Do not edit!


(cl:in-package scara_command-srv)


;//! \htmlinclude SetCartesianVel-request.msg.html

(cl:defclass <SetCartesianVel-request> (roslisp-msg-protocol:ros-message)
  ((Vx
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

(cl:defclass SetCartesianVel-request (<SetCartesianVel-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetCartesianVel-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetCartesianVel-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name scara_command-srv:<SetCartesianVel-request> is deprecated: use scara_command-srv:SetCartesianVel-request instead.")))

(cl:ensure-generic-function 'Vx-val :lambda-list '(m))
(cl:defmethod Vx-val ((m <SetCartesianVel-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:Vx-val is deprecated.  Use scara_command-srv:Vx instead.")
  (Vx m))

(cl:ensure-generic-function 'Vy-val :lambda-list '(m))
(cl:defmethod Vy-val ((m <SetCartesianVel-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:Vy-val is deprecated.  Use scara_command-srv:Vy instead.")
  (Vy m))

(cl:ensure-generic-function 'Vz-val :lambda-list '(m))
(cl:defmethod Vz-val ((m <SetCartesianVel-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:Vz-val is deprecated.  Use scara_command-srv:Vz instead.")
  (Vz m))

(cl:ensure-generic-function 'Wx-val :lambda-list '(m))
(cl:defmethod Wx-val ((m <SetCartesianVel-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:Wx-val is deprecated.  Use scara_command-srv:Wx instead.")
  (Wx m))

(cl:ensure-generic-function 'Wy-val :lambda-list '(m))
(cl:defmethod Wy-val ((m <SetCartesianVel-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:Wy-val is deprecated.  Use scara_command-srv:Wy instead.")
  (Wy m))

(cl:ensure-generic-function 'Wz-val :lambda-list '(m))
(cl:defmethod Wz-val ((m <SetCartesianVel-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:Wz-val is deprecated.  Use scara_command-srv:Wz instead.")
  (Wz m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetCartesianVel-request>) ostream)
  "Serializes a message object of type '<SetCartesianVel-request>"
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
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetCartesianVel-request>) istream)
  "Deserializes a message object of type '<SetCartesianVel-request>"
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetCartesianVel-request>)))
  "Returns string type for a service object of type '<SetCartesianVel-request>"
  "scara_command/SetCartesianVelRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetCartesianVel-request)))
  "Returns string type for a service object of type 'SetCartesianVel-request"
  "scara_command/SetCartesianVelRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetCartesianVel-request>)))
  "Returns md5sum for a message object of type '<SetCartesianVel-request>"
  "283707b7ad403b8c31a22ddc99890608")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetCartesianVel-request)))
  "Returns md5sum for a message object of type 'SetCartesianVel-request"
  "283707b7ad403b8c31a22ddc99890608")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetCartesianVel-request>)))
  "Returns full string definition for message of type '<SetCartesianVel-request>"
  (cl:format cl:nil "float32 Vx~%float32 Vy~%float32 Vz~%float32 Wx~%float32 Wy~%float32 Wz~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetCartesianVel-request)))
  "Returns full string definition for message of type 'SetCartesianVel-request"
  (cl:format cl:nil "float32 Vx~%float32 Vy~%float32 Vz~%float32 Wx~%float32 Wy~%float32 Wz~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetCartesianVel-request>))
  (cl:+ 0
     4
     4
     4
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetCartesianVel-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetCartesianVel-request
    (cl:cons ':Vx (Vx msg))
    (cl:cons ':Vy (Vy msg))
    (cl:cons ':Vz (Vz msg))
    (cl:cons ':Wx (Wx msg))
    (cl:cons ':Wy (Wy msg))
    (cl:cons ':Wz (Wz msg))
))
;//! \htmlinclude SetCartesianVel-response.msg.html

(cl:defclass <SetCartesianVel-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass SetCartesianVel-response (<SetCartesianVel-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetCartesianVel-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetCartesianVel-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name scara_command-srv:<SetCartesianVel-response> is deprecated: use scara_command-srv:SetCartesianVel-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <SetCartesianVel-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:success-val is deprecated.  Use scara_command-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetCartesianVel-response>) ostream)
  "Serializes a message object of type '<SetCartesianVel-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetCartesianVel-response>) istream)
  "Deserializes a message object of type '<SetCartesianVel-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetCartesianVel-response>)))
  "Returns string type for a service object of type '<SetCartesianVel-response>"
  "scara_command/SetCartesianVelResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetCartesianVel-response)))
  "Returns string type for a service object of type 'SetCartesianVel-response"
  "scara_command/SetCartesianVelResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetCartesianVel-response>)))
  "Returns md5sum for a message object of type '<SetCartesianVel-response>"
  "283707b7ad403b8c31a22ddc99890608")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetCartesianVel-response)))
  "Returns md5sum for a message object of type 'SetCartesianVel-response"
  "283707b7ad403b8c31a22ddc99890608")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetCartesianVel-response>)))
  "Returns full string definition for message of type '<SetCartesianVel-response>"
  (cl:format cl:nil "bool success~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetCartesianVel-response)))
  "Returns full string definition for message of type 'SetCartesianVel-response"
  (cl:format cl:nil "bool success~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetCartesianVel-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetCartesianVel-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetCartesianVel-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetCartesianVel)))
  'SetCartesianVel-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetCartesianVel)))
  'SetCartesianVel-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetCartesianVel)))
  "Returns string type for a service object of type '<SetCartesianVel>"
  "scara_command/SetCartesianVel")