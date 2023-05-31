; Auto-generated. Do not edit!


(cl:in-package scara_command-srv)


;//! \htmlinclude SetJointRef-request.msg.html

(cl:defclass <SetJointRef-request> (roslisp-msg-protocol:ros-message)
  ((joint_name
    :reader joint_name
    :initarg :joint_name
    :type cl:string
    :initform "")
   (ref
    :reader ref
    :initarg :ref
    :type cl:float
    :initform 0.0))
)

(cl:defclass SetJointRef-request (<SetJointRef-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetJointRef-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetJointRef-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name scara_command-srv:<SetJointRef-request> is deprecated: use scara_command-srv:SetJointRef-request instead.")))

(cl:ensure-generic-function 'joint_name-val :lambda-list '(m))
(cl:defmethod joint_name-val ((m <SetJointRef-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:joint_name-val is deprecated.  Use scara_command-srv:joint_name instead.")
  (joint_name m))

(cl:ensure-generic-function 'ref-val :lambda-list '(m))
(cl:defmethod ref-val ((m <SetJointRef-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:ref-val is deprecated.  Use scara_command-srv:ref instead.")
  (ref m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetJointRef-request>) ostream)
  "Serializes a message object of type '<SetJointRef-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'joint_name))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'joint_name))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'ref))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetJointRef-request>) istream)
  "Deserializes a message object of type '<SetJointRef-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'joint_name) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'joint_name) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'ref) (roslisp-utils:decode-double-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetJointRef-request>)))
  "Returns string type for a service object of type '<SetJointRef-request>"
  "scara_command/SetJointRefRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetJointRef-request)))
  "Returns string type for a service object of type 'SetJointRef-request"
  "scara_command/SetJointRefRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetJointRef-request>)))
  "Returns md5sum for a message object of type '<SetJointRef-request>"
  "6f194d3e831f68db4bba0c86e04a9975")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetJointRef-request)))
  "Returns md5sum for a message object of type 'SetJointRef-request"
  "6f194d3e831f68db4bba0c86e04a9975")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetJointRef-request>)))
  "Returns full string definition for message of type '<SetJointRef-request>"
  (cl:format cl:nil "string joint_name~%float64 ref~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetJointRef-request)))
  "Returns full string definition for message of type 'SetJointRef-request"
  (cl:format cl:nil "string joint_name~%float64 ref~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetJointRef-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'joint_name))
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetJointRef-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetJointRef-request
    (cl:cons ':joint_name (joint_name msg))
    (cl:cons ':ref (ref msg))
))
;//! \htmlinclude SetJointRef-response.msg.html

(cl:defclass <SetJointRef-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass SetJointRef-response (<SetJointRef-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetJointRef-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetJointRef-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name scara_command-srv:<SetJointRef-response> is deprecated: use scara_command-srv:SetJointRef-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <SetJointRef-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:success-val is deprecated.  Use scara_command-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetJointRef-response>) ostream)
  "Serializes a message object of type '<SetJointRef-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetJointRef-response>) istream)
  "Deserializes a message object of type '<SetJointRef-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetJointRef-response>)))
  "Returns string type for a service object of type '<SetJointRef-response>"
  "scara_command/SetJointRefResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetJointRef-response)))
  "Returns string type for a service object of type 'SetJointRef-response"
  "scara_command/SetJointRefResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetJointRef-response>)))
  "Returns md5sum for a message object of type '<SetJointRef-response>"
  "6f194d3e831f68db4bba0c86e04a9975")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetJointRef-response)))
  "Returns md5sum for a message object of type 'SetJointRef-response"
  "6f194d3e831f68db4bba0c86e04a9975")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetJointRef-response>)))
  "Returns full string definition for message of type '<SetJointRef-response>"
  (cl:format cl:nil "bool success~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetJointRef-response)))
  "Returns full string definition for message of type 'SetJointRef-response"
  (cl:format cl:nil "bool success~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetJointRef-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetJointRef-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetJointRef-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetJointRef)))
  'SetJointRef-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetJointRef)))
  'SetJointRef-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetJointRef)))
  "Returns string type for a service object of type '<SetJointRef>"
  "scara_command/SetJointRef")