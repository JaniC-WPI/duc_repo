; Auto-generated. Do not edit!


(cl:in-package scara_command-srv)


;//! \htmlinclude ScaraKinIK-request.msg.html

(cl:defclass <ScaraKinIK-request> (roslisp-msg-protocol:ros-message)
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
    :initform 0.0)
   (phi
    :reader phi
    :initarg :phi
    :type cl:float
    :initform 0.0)
   (theta
    :reader theta
    :initarg :theta
    :type cl:float
    :initform 0.0)
   (psi
    :reader psi
    :initarg :psi
    :type cl:float
    :initform 0.0))
)

(cl:defclass ScaraKinIK-request (<ScaraKinIK-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ScaraKinIK-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ScaraKinIK-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name scara_command-srv:<ScaraKinIK-request> is deprecated: use scara_command-srv:ScaraKinIK-request instead.")))

(cl:ensure-generic-function 'x-val :lambda-list '(m))
(cl:defmethod x-val ((m <ScaraKinIK-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:x-val is deprecated.  Use scara_command-srv:x instead.")
  (x m))

(cl:ensure-generic-function 'y-val :lambda-list '(m))
(cl:defmethod y-val ((m <ScaraKinIK-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:y-val is deprecated.  Use scara_command-srv:y instead.")
  (y m))

(cl:ensure-generic-function 'z-val :lambda-list '(m))
(cl:defmethod z-val ((m <ScaraKinIK-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:z-val is deprecated.  Use scara_command-srv:z instead.")
  (z m))

(cl:ensure-generic-function 'phi-val :lambda-list '(m))
(cl:defmethod phi-val ((m <ScaraKinIK-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:phi-val is deprecated.  Use scara_command-srv:phi instead.")
  (phi m))

(cl:ensure-generic-function 'theta-val :lambda-list '(m))
(cl:defmethod theta-val ((m <ScaraKinIK-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:theta-val is deprecated.  Use scara_command-srv:theta instead.")
  (theta m))

(cl:ensure-generic-function 'psi-val :lambda-list '(m))
(cl:defmethod psi-val ((m <ScaraKinIK-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:psi-val is deprecated.  Use scara_command-srv:psi instead.")
  (psi m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ScaraKinIK-request>) ostream)
  "Serializes a message object of type '<ScaraKinIK-request>"
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
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'phi))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'theta))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'psi))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ScaraKinIK-request>) istream)
  "Deserializes a message object of type '<ScaraKinIK-request>"
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
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'phi) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'theta) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'psi) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ScaraKinIK-request>)))
  "Returns string type for a service object of type '<ScaraKinIK-request>"
  "scara_command/ScaraKinIKRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ScaraKinIK-request)))
  "Returns string type for a service object of type 'ScaraKinIK-request"
  "scara_command/ScaraKinIKRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ScaraKinIK-request>)))
  "Returns md5sum for a message object of type '<ScaraKinIK-request>"
  "90f36ce76dbe958bc51489fea1b8cf59")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ScaraKinIK-request)))
  "Returns md5sum for a message object of type 'ScaraKinIK-request"
  "90f36ce76dbe958bc51489fea1b8cf59")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ScaraKinIK-request>)))
  "Returns full string definition for message of type '<ScaraKinIK-request>"
  (cl:format cl:nil "float32 x~%float32 y~%float32 z~%float32 phi~%float32 theta~%float32 psi~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ScaraKinIK-request)))
  "Returns full string definition for message of type 'ScaraKinIK-request"
  (cl:format cl:nil "float32 x~%float32 y~%float32 z~%float32 phi~%float32 theta~%float32 psi~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ScaraKinIK-request>))
  (cl:+ 0
     4
     4
     4
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ScaraKinIK-request>))
  "Converts a ROS message object to a list"
  (cl:list 'ScaraKinIK-request
    (cl:cons ':x (x msg))
    (cl:cons ':y (y msg))
    (cl:cons ':z (z msg))
    (cl:cons ':phi (phi msg))
    (cl:cons ':theta (theta msg))
    (cl:cons ':psi (psi msg))
))
;//! \htmlinclude ScaraKinIK-response.msg.html

(cl:defclass <ScaraKinIK-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil)
   (q1
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

(cl:defclass ScaraKinIK-response (<ScaraKinIK-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ScaraKinIK-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ScaraKinIK-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name scara_command-srv:<ScaraKinIK-response> is deprecated: use scara_command-srv:ScaraKinIK-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <ScaraKinIK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:success-val is deprecated.  Use scara_command-srv:success instead.")
  (success m))

(cl:ensure-generic-function 'q1-val :lambda-list '(m))
(cl:defmethod q1-val ((m <ScaraKinIK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:q1-val is deprecated.  Use scara_command-srv:q1 instead.")
  (q1 m))

(cl:ensure-generic-function 'q2-val :lambda-list '(m))
(cl:defmethod q2-val ((m <ScaraKinIK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:q2-val is deprecated.  Use scara_command-srv:q2 instead.")
  (q2 m))

(cl:ensure-generic-function 'q3-val :lambda-list '(m))
(cl:defmethod q3-val ((m <ScaraKinIK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:q3-val is deprecated.  Use scara_command-srv:q3 instead.")
  (q3 m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ScaraKinIK-response>) ostream)
  "Serializes a message object of type '<ScaraKinIK-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
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
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ScaraKinIK-response>) istream)
  "Deserializes a message object of type '<ScaraKinIK-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ScaraKinIK-response>)))
  "Returns string type for a service object of type '<ScaraKinIK-response>"
  "scara_command/ScaraKinIKResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ScaraKinIK-response)))
  "Returns string type for a service object of type 'ScaraKinIK-response"
  "scara_command/ScaraKinIKResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ScaraKinIK-response>)))
  "Returns md5sum for a message object of type '<ScaraKinIK-response>"
  "90f36ce76dbe958bc51489fea1b8cf59")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ScaraKinIK-response)))
  "Returns md5sum for a message object of type 'ScaraKinIK-response"
  "90f36ce76dbe958bc51489fea1b8cf59")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ScaraKinIK-response>)))
  "Returns full string definition for message of type '<ScaraKinIK-response>"
  (cl:format cl:nil "bool success~%float32 q1~%float32 q2~%float32 q3~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ScaraKinIK-response)))
  "Returns full string definition for message of type 'ScaraKinIK-response"
  (cl:format cl:nil "bool success~%float32 q1~%float32 q2~%float32 q3~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ScaraKinIK-response>))
  (cl:+ 0
     1
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ScaraKinIK-response>))
  "Converts a ROS message object to a list"
  (cl:list 'ScaraKinIK-response
    (cl:cons ':success (success msg))
    (cl:cons ':q1 (q1 msg))
    (cl:cons ':q2 (q2 msg))
    (cl:cons ':q3 (q3 msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'ScaraKinIK)))
  'ScaraKinIK-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'ScaraKinIK)))
  'ScaraKinIK-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ScaraKinIK)))
  "Returns string type for a service object of type '<ScaraKinIK>"
  "scara_command/ScaraKinIK")