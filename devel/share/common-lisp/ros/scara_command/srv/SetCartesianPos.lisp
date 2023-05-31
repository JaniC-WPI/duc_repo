; Auto-generated. Do not edit!


(cl:in-package scara_command-srv)


;//! \htmlinclude SetCartesianPos-request.msg.html

(cl:defclass <SetCartesianPos-request> (roslisp-msg-protocol:ros-message)
  ((x
    :reader x
    :initarg :x
    :type cl:float
    :initform 0.0)
   (y
    :reader y
    :initarg :y
    :type cl:float
    :initform 0.0)
   (z
    :reader z
    :initarg :z
    :type cl:float
    :initform 0.0))
)

(cl:defclass SetCartesianPos-request (<SetCartesianPos-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetCartesianPos-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetCartesianPos-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name scara_command-srv:<SetCartesianPos-request> is deprecated: use scara_command-srv:SetCartesianPos-request instead.")))

(cl:ensure-generic-function 'x-val :lambda-list '(m))
(cl:defmethod x-val ((m <SetCartesianPos-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:x-val is deprecated.  Use scara_command-srv:x instead.")
  (x m))

(cl:ensure-generic-function 'y-val :lambda-list '(m))
(cl:defmethod y-val ((m <SetCartesianPos-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:y-val is deprecated.  Use scara_command-srv:y instead.")
  (y m))

(cl:ensure-generic-function 'z-val :lambda-list '(m))
(cl:defmethod z-val ((m <SetCartesianPos-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:z-val is deprecated.  Use scara_command-srv:z instead.")
  (z m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetCartesianPos-request>) ostream)
  "Serializes a message object of type '<SetCartesianPos-request>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'x))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'y))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'z))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetCartesianPos-request>) istream)
  "Deserializes a message object of type '<SetCartesianPos-request>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'x) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'y) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'z) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetCartesianPos-request>)))
  "Returns string type for a service object of type '<SetCartesianPos-request>"
  "scara_command/SetCartesianPosRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetCartesianPos-request)))
  "Returns string type for a service object of type 'SetCartesianPos-request"
  "scara_command/SetCartesianPosRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetCartesianPos-request>)))
  "Returns md5sum for a message object of type '<SetCartesianPos-request>"
  "58d59f258ca9f2d2ba375d9428a7f1de")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetCartesianPos-request)))
  "Returns md5sum for a message object of type 'SetCartesianPos-request"
  "58d59f258ca9f2d2ba375d9428a7f1de")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetCartesianPos-request>)))
  "Returns full string definition for message of type '<SetCartesianPos-request>"
  (cl:format cl:nil "float32 x~%float32 y~%float32 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetCartesianPos-request)))
  "Returns full string definition for message of type 'SetCartesianPos-request"
  (cl:format cl:nil "float32 x~%float32 y~%float32 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetCartesianPos-request>))
  (cl:+ 0
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetCartesianPos-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetCartesianPos-request
    (cl:cons ':x (x msg))
    (cl:cons ':y (y msg))
    (cl:cons ':z (z msg))
))
;//! \htmlinclude SetCartesianPos-response.msg.html

(cl:defclass <SetCartesianPos-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass SetCartesianPos-response (<SetCartesianPos-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetCartesianPos-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetCartesianPos-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name scara_command-srv:<SetCartesianPos-response> is deprecated: use scara_command-srv:SetCartesianPos-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <SetCartesianPos-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:success-val is deprecated.  Use scara_command-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetCartesianPos-response>) ostream)
  "Serializes a message object of type '<SetCartesianPos-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetCartesianPos-response>) istream)
  "Deserializes a message object of type '<SetCartesianPos-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetCartesianPos-response>)))
  "Returns string type for a service object of type '<SetCartesianPos-response>"
  "scara_command/SetCartesianPosResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetCartesianPos-response)))
  "Returns string type for a service object of type 'SetCartesianPos-response"
  "scara_command/SetCartesianPosResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetCartesianPos-response>)))
  "Returns md5sum for a message object of type '<SetCartesianPos-response>"
  "58d59f258ca9f2d2ba375d9428a7f1de")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetCartesianPos-response)))
  "Returns md5sum for a message object of type 'SetCartesianPos-response"
  "58d59f258ca9f2d2ba375d9428a7f1de")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetCartesianPos-response>)))
  "Returns full string definition for message of type '<SetCartesianPos-response>"
  (cl:format cl:nil "bool success~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetCartesianPos-response)))
  "Returns full string definition for message of type 'SetCartesianPos-response"
  (cl:format cl:nil "bool success~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetCartesianPos-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetCartesianPos-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetCartesianPos-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetCartesianPos)))
  'SetCartesianPos-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetCartesianPos)))
  'SetCartesianPos-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetCartesianPos)))
  "Returns string type for a service object of type '<SetCartesianPos>"
  "scara_command/SetCartesianPos")