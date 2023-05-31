; Auto-generated. Do not edit!


(cl:in-package scara_command-srv)


;//! \htmlinclude CheckKinFK-request.msg.html

(cl:defclass <CheckKinFK-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass CheckKinFK-request (<CheckKinFK-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <CheckKinFK-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'CheckKinFK-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name scara_command-srv:<CheckKinFK-request> is deprecated: use scara_command-srv:CheckKinFK-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <CheckKinFK-request>) ostream)
  "Serializes a message object of type '<CheckKinFK-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <CheckKinFK-request>) istream)
  "Deserializes a message object of type '<CheckKinFK-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<CheckKinFK-request>)))
  "Returns string type for a service object of type '<CheckKinFK-request>"
  "scara_command/CheckKinFKRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'CheckKinFK-request)))
  "Returns string type for a service object of type 'CheckKinFK-request"
  "scara_command/CheckKinFKRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<CheckKinFK-request>)))
  "Returns md5sum for a message object of type '<CheckKinFK-request>"
  "6dd4bd6e62545926c2a7502c0ee7c4f1")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'CheckKinFK-request)))
  "Returns md5sum for a message object of type 'CheckKinFK-request"
  "6dd4bd6e62545926c2a7502c0ee7c4f1")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<CheckKinFK-request>)))
  "Returns full string definition for message of type '<CheckKinFK-request>"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'CheckKinFK-request)))
  "Returns full string definition for message of type 'CheckKinFK-request"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <CheckKinFK-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <CheckKinFK-request>))
  "Converts a ROS message object to a list"
  (cl:list 'CheckKinFK-request
))
;//! \htmlinclude CheckKinFK-response.msg.html

(cl:defclass <CheckKinFK-response> (roslisp-msg-protocol:ros-message)
  ((state_x
    :reader state_x
    :initarg :state_x
    :type cl:float
    :initform 0.0)
   (state_y
    :reader state_y
    :initarg :state_y
    :type cl:float
    :initform 0.0)
   (state_z
    :reader state_z
    :initarg :state_z
    :type cl:float
    :initform 0.0)
   (state_phi
    :reader state_phi
    :initarg :state_phi
    :type cl:float
    :initform 0.0)
   (state_theta
    :reader state_theta
    :initarg :state_theta
    :type cl:float
    :initform 0.0)
   (state_psi
    :reader state_psi
    :initarg :state_psi
    :type cl:float
    :initform 0.0)
   (fk_x
    :reader fk_x
    :initarg :fk_x
    :type cl:float
    :initform 0.0)
   (fk_y
    :reader fk_y
    :initarg :fk_y
    :type cl:float
    :initform 0.0)
   (fk_z
    :reader fk_z
    :initarg :fk_z
    :type cl:float
    :initform 0.0)
   (fk_phi
    :reader fk_phi
    :initarg :fk_phi
    :type cl:float
    :initform 0.0)
   (fk_theta
    :reader fk_theta
    :initarg :fk_theta
    :type cl:float
    :initform 0.0)
   (fk_psi
    :reader fk_psi
    :initarg :fk_psi
    :type cl:float
    :initform 0.0)
   (correct
    :reader correct
    :initarg :correct
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass CheckKinFK-response (<CheckKinFK-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <CheckKinFK-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'CheckKinFK-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name scara_command-srv:<CheckKinFK-response> is deprecated: use scara_command-srv:CheckKinFK-response instead.")))

(cl:ensure-generic-function 'state_x-val :lambda-list '(m))
(cl:defmethod state_x-val ((m <CheckKinFK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:state_x-val is deprecated.  Use scara_command-srv:state_x instead.")
  (state_x m))

(cl:ensure-generic-function 'state_y-val :lambda-list '(m))
(cl:defmethod state_y-val ((m <CheckKinFK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:state_y-val is deprecated.  Use scara_command-srv:state_y instead.")
  (state_y m))

(cl:ensure-generic-function 'state_z-val :lambda-list '(m))
(cl:defmethod state_z-val ((m <CheckKinFK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:state_z-val is deprecated.  Use scara_command-srv:state_z instead.")
  (state_z m))

(cl:ensure-generic-function 'state_phi-val :lambda-list '(m))
(cl:defmethod state_phi-val ((m <CheckKinFK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:state_phi-val is deprecated.  Use scara_command-srv:state_phi instead.")
  (state_phi m))

(cl:ensure-generic-function 'state_theta-val :lambda-list '(m))
(cl:defmethod state_theta-val ((m <CheckKinFK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:state_theta-val is deprecated.  Use scara_command-srv:state_theta instead.")
  (state_theta m))

(cl:ensure-generic-function 'state_psi-val :lambda-list '(m))
(cl:defmethod state_psi-val ((m <CheckKinFK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:state_psi-val is deprecated.  Use scara_command-srv:state_psi instead.")
  (state_psi m))

(cl:ensure-generic-function 'fk_x-val :lambda-list '(m))
(cl:defmethod fk_x-val ((m <CheckKinFK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:fk_x-val is deprecated.  Use scara_command-srv:fk_x instead.")
  (fk_x m))

(cl:ensure-generic-function 'fk_y-val :lambda-list '(m))
(cl:defmethod fk_y-val ((m <CheckKinFK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:fk_y-val is deprecated.  Use scara_command-srv:fk_y instead.")
  (fk_y m))

(cl:ensure-generic-function 'fk_z-val :lambda-list '(m))
(cl:defmethod fk_z-val ((m <CheckKinFK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:fk_z-val is deprecated.  Use scara_command-srv:fk_z instead.")
  (fk_z m))

(cl:ensure-generic-function 'fk_phi-val :lambda-list '(m))
(cl:defmethod fk_phi-val ((m <CheckKinFK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:fk_phi-val is deprecated.  Use scara_command-srv:fk_phi instead.")
  (fk_phi m))

(cl:ensure-generic-function 'fk_theta-val :lambda-list '(m))
(cl:defmethod fk_theta-val ((m <CheckKinFK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:fk_theta-val is deprecated.  Use scara_command-srv:fk_theta instead.")
  (fk_theta m))

(cl:ensure-generic-function 'fk_psi-val :lambda-list '(m))
(cl:defmethod fk_psi-val ((m <CheckKinFK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:fk_psi-val is deprecated.  Use scara_command-srv:fk_psi instead.")
  (fk_psi m))

(cl:ensure-generic-function 'correct-val :lambda-list '(m))
(cl:defmethod correct-val ((m <CheckKinFK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:correct-val is deprecated.  Use scara_command-srv:correct instead.")
  (correct m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <CheckKinFK-response>) ostream)
  "Serializes a message object of type '<CheckKinFK-response>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'state_x))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'state_y))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'state_z))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'state_phi))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'state_theta))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'state_psi))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'fk_x))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'fk_y))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'fk_z))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'fk_phi))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'fk_theta))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'fk_psi))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'correct) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <CheckKinFK-response>) istream)
  "Deserializes a message object of type '<CheckKinFK-response>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'state_x) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'state_y) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'state_z) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'state_phi) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'state_theta) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'state_psi) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'fk_x) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'fk_y) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'fk_z) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'fk_phi) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'fk_theta) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'fk_psi) (roslisp-utils:decode-single-float-bits bits)))
    (cl:setf (cl:slot-value msg 'correct) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<CheckKinFK-response>)))
  "Returns string type for a service object of type '<CheckKinFK-response>"
  "scara_command/CheckKinFKResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'CheckKinFK-response)))
  "Returns string type for a service object of type 'CheckKinFK-response"
  "scara_command/CheckKinFKResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<CheckKinFK-response>)))
  "Returns md5sum for a message object of type '<CheckKinFK-response>"
  "6dd4bd6e62545926c2a7502c0ee7c4f1")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'CheckKinFK-response)))
  "Returns md5sum for a message object of type 'CheckKinFK-response"
  "6dd4bd6e62545926c2a7502c0ee7c4f1")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<CheckKinFK-response>)))
  "Returns full string definition for message of type '<CheckKinFK-response>"
  (cl:format cl:nil "float32 state_x~%float32 state_y~%float32 state_z~%float32 state_phi~%float32 state_theta~%float32 state_psi~%float32 fk_x~%float32 fk_y~%float32 fk_z~%float32 fk_phi~%float32 fk_theta~%float32 fk_psi~%bool correct~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'CheckKinFK-response)))
  "Returns full string definition for message of type 'CheckKinFK-response"
  (cl:format cl:nil "float32 state_x~%float32 state_y~%float32 state_z~%float32 state_phi~%float32 state_theta~%float32 state_psi~%float32 fk_x~%float32 fk_y~%float32 fk_z~%float32 fk_phi~%float32 fk_theta~%float32 fk_psi~%bool correct~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <CheckKinFK-response>))
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
     4
     4
     4
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <CheckKinFK-response>))
  "Converts a ROS message object to a list"
  (cl:list 'CheckKinFK-response
    (cl:cons ':state_x (state_x msg))
    (cl:cons ':state_y (state_y msg))
    (cl:cons ':state_z (state_z msg))
    (cl:cons ':state_phi (state_phi msg))
    (cl:cons ':state_theta (state_theta msg))
    (cl:cons ':state_psi (state_psi msg))
    (cl:cons ':fk_x (fk_x msg))
    (cl:cons ':fk_y (fk_y msg))
    (cl:cons ':fk_z (fk_z msg))
    (cl:cons ':fk_phi (fk_phi msg))
    (cl:cons ':fk_theta (fk_theta msg))
    (cl:cons ':fk_psi (fk_psi msg))
    (cl:cons ':correct (correct msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'CheckKinFK)))
  'CheckKinFK-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'CheckKinFK)))
  'CheckKinFK-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'CheckKinFK)))
  "Returns string type for a service object of type '<CheckKinFK>"
  "scara_command/CheckKinFK")