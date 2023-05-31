; Auto-generated. Do not edit!


(cl:in-package scara_command-srv)


;//! \htmlinclude SwitchControl-request.msg.html

(cl:defclass <SwitchControl-request> (roslisp-msg-protocol:ros-message)
  ((command
    :reader command
    :initarg :command
    :type cl:string
    :initform ""))
)

(cl:defclass SwitchControl-request (<SwitchControl-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SwitchControl-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SwitchControl-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name scara_command-srv:<SwitchControl-request> is deprecated: use scara_command-srv:SwitchControl-request instead.")))

(cl:ensure-generic-function 'command-val :lambda-list '(m))
(cl:defmethod command-val ((m <SwitchControl-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:command-val is deprecated.  Use scara_command-srv:command instead.")
  (command m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SwitchControl-request>) ostream)
  "Serializes a message object of type '<SwitchControl-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'command))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'command))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SwitchControl-request>) istream)
  "Deserializes a message object of type '<SwitchControl-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'command) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'command) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SwitchControl-request>)))
  "Returns string type for a service object of type '<SwitchControl-request>"
  "scara_command/SwitchControlRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SwitchControl-request)))
  "Returns string type for a service object of type 'SwitchControl-request"
  "scara_command/SwitchControlRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SwitchControl-request>)))
  "Returns md5sum for a message object of type '<SwitchControl-request>"
  "031d24522d462b2314fd1b6270670dd2")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SwitchControl-request)))
  "Returns md5sum for a message object of type 'SwitchControl-request"
  "031d24522d462b2314fd1b6270670dd2")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SwitchControl-request>)))
  "Returns full string definition for message of type '<SwitchControl-request>"
  (cl:format cl:nil "string command~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SwitchControl-request)))
  "Returns full string definition for message of type 'SwitchControl-request"
  (cl:format cl:nil "string command~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SwitchControl-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'command))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SwitchControl-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SwitchControl-request
    (cl:cons ':command (command msg))
))
;//! \htmlinclude SwitchControl-response.msg.html

(cl:defclass <SwitchControl-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass SwitchControl-response (<SwitchControl-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SwitchControl-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SwitchControl-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name scara_command-srv:<SwitchControl-response> is deprecated: use scara_command-srv:SwitchControl-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <SwitchControl-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:success-val is deprecated.  Use scara_command-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SwitchControl-response>) ostream)
  "Serializes a message object of type '<SwitchControl-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SwitchControl-response>) istream)
  "Deserializes a message object of type '<SwitchControl-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SwitchControl-response>)))
  "Returns string type for a service object of type '<SwitchControl-response>"
  "scara_command/SwitchControlResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SwitchControl-response)))
  "Returns string type for a service object of type 'SwitchControl-response"
  "scara_command/SwitchControlResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SwitchControl-response>)))
  "Returns md5sum for a message object of type '<SwitchControl-response>"
  "031d24522d462b2314fd1b6270670dd2")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SwitchControl-response)))
  "Returns md5sum for a message object of type 'SwitchControl-response"
  "031d24522d462b2314fd1b6270670dd2")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SwitchControl-response>)))
  "Returns full string definition for message of type '<SwitchControl-response>"
  (cl:format cl:nil "bool success~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SwitchControl-response)))
  "Returns full string definition for message of type 'SwitchControl-response"
  (cl:format cl:nil "bool success~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SwitchControl-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SwitchControl-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SwitchControl-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SwitchControl)))
  'SwitchControl-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SwitchControl)))
  'SwitchControl-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SwitchControl)))
  "Returns string type for a service object of type '<SwitchControl>"
  "scara_command/SwitchControl")