; Auto-generated. Do not edit!


(cl:in-package scara_command-srv)


;//! \htmlinclude CheckKinIK-request.msg.html

(cl:defclass <CheckKinIK-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass CheckKinIK-request (<CheckKinIK-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <CheckKinIK-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'CheckKinIK-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name scara_command-srv:<CheckKinIK-request> is deprecated: use scara_command-srv:CheckKinIK-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <CheckKinIK-request>) ostream)
  "Serializes a message object of type '<CheckKinIK-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <CheckKinIK-request>) istream)
  "Deserializes a message object of type '<CheckKinIK-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<CheckKinIK-request>)))
  "Returns string type for a service object of type '<CheckKinIK-request>"
  "scara_command/CheckKinIKRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'CheckKinIK-request)))
  "Returns string type for a service object of type 'CheckKinIK-request"
  "scara_command/CheckKinIKRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<CheckKinIK-request>)))
  "Returns md5sum for a message object of type '<CheckKinIK-request>"
  "36714075758334348b25a38f8ec94251")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'CheckKinIK-request)))
  "Returns md5sum for a message object of type 'CheckKinIK-request"
  "36714075758334348b25a38f8ec94251")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<CheckKinIK-request>)))
  "Returns full string definition for message of type '<CheckKinIK-request>"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'CheckKinIK-request)))
  "Returns full string definition for message of type 'CheckKinIK-request"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <CheckKinIK-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <CheckKinIK-request>))
  "Converts a ROS message object to a list"
  (cl:list 'CheckKinIK-request
))
;//! \htmlinclude CheckKinIK-response.msg.html

(cl:defclass <CheckKinIK-response> (roslisp-msg-protocol:ros-message)
  ((state_q1
    :reader state_q1
    :initarg :state_q1
    :type cl:float
    :initform 0.0)
   (state_q2
    :reader state_q2
    :initarg :state_q2
    :type cl:float
    :initform 0.0)
   (state_q3
    :reader state_q3
    :initarg :state_q3
    :type cl:float
    :initform 0.0)
   (ik_q1
    :reader ik_q1
    :initarg :ik_q1
    :type cl:float
    :initform 0.0)
   (ik_q2
    :reader ik_q2
    :initarg :ik_q2
    :type cl:float
    :initform 0.0)
   (ik_q3
    :reader ik_q3
    :initarg :ik_q3
    :type cl:float
    :initform 0.0)
   (correct
    :reader correct
    :initarg :correct
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass CheckKinIK-response (<CheckKinIK-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <CheckKinIK-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'CheckKinIK-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name scara_command-srv:<CheckKinIK-response> is deprecated: use scara_command-srv:CheckKinIK-response instead.")))

(cl:ensure-generic-function 'state_q1-val :lambda-list '(m))
(cl:defmethod state_q1-val ((m <CheckKinIK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:state_q1-val is deprecated.  Use scara_command-srv:state_q1 instead.")
  (state_q1 m))

(cl:ensure-generic-function 'state_q2-val :lambda-list '(m))
(cl:defmethod state_q2-val ((m <CheckKinIK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:state_q2-val is deprecated.  Use scara_command-srv:state_q2 instead.")
  (state_q2 m))

(cl:ensure-generic-function 'state_q3-val :lambda-list '(m))
(cl:defmethod state_q3-val ((m <CheckKinIK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:state_q3-val is deprecated.  Use scara_command-srv:state_q3 instead.")
  (state_q3 m))

(cl:ensure-generic-function 'ik_q1-val :lambda-list '(m))
(cl:defmethod ik_q1-val ((m <CheckKinIK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:ik_q1-val is deprecated.  Use scara_command-srv:ik_q1 instead.")
  (ik_q1 m))

(cl:ensure-generic-function 'ik_q2-val :lambda-list '(m))
(cl:defmethod ik_q2-val ((m <CheckKinIK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:ik_q2-val is deprecated.  Use scara_command-srv:ik_q2 instead.")
  (ik_q2 m))

(cl:ensure-generic-function 'ik_q3-val :lambda-list '(m))
(cl:defmethod ik_q3-val ((m <CheckKinIK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:ik_q3-val is deprecated.  Use scara_command-srv:ik_q3 instead.")
  (ik_q3 m))

(cl:ensure-generic-function 'correct-val :lambda-list '(m))
(cl:defmethod correct-val ((m <CheckKinIK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:correct-val is deprecated.  Use scara_command-srv:correct instead.")
  (correct m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <CheckKinIK-response>) ostream)
  "Serializes a message object of type '<CheckKinIK-response>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'state_q1))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'state_q2))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'state_q3))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'ik_q1))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'ik_q2))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'ik_q3))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'correct) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <CheckKinIK-response>) istream)
  "Deserializes a message object of type '<CheckKinIK-response>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'state_q1) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'state_q2) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'state_q3) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'ik_q1) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'ik_q2) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'ik_q3) (roslisp-utils:decode-single-float-bits bits)))
    (cl:setf (cl:slot-value msg 'correct) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<CheckKinIK-response>)))
  "Returns string type for a service object of type '<CheckKinIK-response>"
  "scara_command/CheckKinIKResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'CheckKinIK-response)))
  "Returns string type for a service object of type 'CheckKinIK-response"
  "scara_command/CheckKinIKResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<CheckKinIK-response>)))
  "Returns md5sum for a message object of type '<CheckKinIK-response>"
  "36714075758334348b25a38f8ec94251")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'CheckKinIK-response)))
  "Returns md5sum for a message object of type 'CheckKinIK-response"
  "36714075758334348b25a38f8ec94251")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<CheckKinIK-response>)))
  "Returns full string definition for message of type '<CheckKinIK-response>"
  (cl:format cl:nil "float32 state_q1~%float32 state_q2~%float32 state_q3~%float32 ik_q1~%float32 ik_q2~%float32 ik_q3~%bool correct~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'CheckKinIK-response)))
  "Returns full string definition for message of type 'CheckKinIK-response"
  (cl:format cl:nil "float32 state_q1~%float32 state_q2~%float32 state_q3~%float32 ik_q1~%float32 ik_q2~%float32 ik_q3~%bool correct~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <CheckKinIK-response>))
  (cl:+ 0
     4
     4
     4
     4
     4
     4
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <CheckKinIK-response>))
  "Converts a ROS message object to a list"
  (cl:list 'CheckKinIK-response
    (cl:cons ':state_q1 (state_q1 msg))
    (cl:cons ':state_q2 (state_q2 msg))
    (cl:cons ':state_q3 (state_q3 msg))
    (cl:cons ':ik_q1 (ik_q1 msg))
    (cl:cons ':ik_q2 (ik_q2 msg))
    (cl:cons ':ik_q3 (ik_q3 msg))
    (cl:cons ':correct (correct msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'CheckKinIK)))
  'CheckKinIK-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'CheckKinIK)))
  'CheckKinIK-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'CheckKinIK)))
  "Returns string type for a service object of type '<CheckKinIK>"
  "scara_command/CheckKinIK")