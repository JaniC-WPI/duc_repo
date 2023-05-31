; Auto-generated. Do not edit!


(cl:in-package scara_command-srv)


;//! \htmlinclude ScaraKinFK-request.msg.html

(cl:defclass <ScaraKinFK-request> (roslisp-msg-protocol:ros-message)
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

(cl:defclass ScaraKinFK-request (<ScaraKinFK-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ScaraKinFK-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ScaraKinFK-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name scara_command-srv:<ScaraKinFK-request> is deprecated: use scara_command-srv:ScaraKinFK-request instead.")))

(cl:ensure-generic-function 'q1-val :lambda-list '(m))
(cl:defmethod q1-val ((m <ScaraKinFK-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:q1-val is deprecated.  Use scara_command-srv:q1 instead.")
  (q1 m))

(cl:ensure-generic-function 'q2-val :lambda-list '(m))
(cl:defmethod q2-val ((m <ScaraKinFK-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:q2-val is deprecated.  Use scara_command-srv:q2 instead.")
  (q2 m))

(cl:ensure-generic-function 'q3-val :lambda-list '(m))
(cl:defmethod q3-val ((m <ScaraKinFK-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:q3-val is deprecated.  Use scara_command-srv:q3 instead.")
  (q3 m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ScaraKinFK-request>) ostream)
  "Serializes a message object of type '<ScaraKinFK-request>"
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
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ScaraKinFK-request>) istream)
  "Deserializes a message object of type '<ScaraKinFK-request>"
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ScaraKinFK-request>)))
  "Returns string type for a service object of type '<ScaraKinFK-request>"
  "scara_command/ScaraKinFKRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ScaraKinFK-request)))
  "Returns string type for a service object of type 'ScaraKinFK-request"
  "scara_command/ScaraKinFKRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ScaraKinFK-request>)))
  "Returns md5sum for a message object of type '<ScaraKinFK-request>"
  "7861a4dbb7e9452dabc20f2eff991915")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ScaraKinFK-request)))
  "Returns md5sum for a message object of type 'ScaraKinFK-request"
  "7861a4dbb7e9452dabc20f2eff991915")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ScaraKinFK-request>)))
  "Returns full string definition for message of type '<ScaraKinFK-request>"
  (cl:format cl:nil "float32 q1~%float32 q2~%float32 q3~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ScaraKinFK-request)))
  "Returns full string definition for message of type 'ScaraKinFK-request"
  (cl:format cl:nil "float32 q1~%float32 q2~%float32 q3~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ScaraKinFK-request>))
  (cl:+ 0
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ScaraKinFK-request>))
  "Converts a ROS message object to a list"
  (cl:list 'ScaraKinFK-request
    (cl:cons ':q1 (q1 msg))
    (cl:cons ':q2 (q2 msg))
    (cl:cons ':q3 (q3 msg))
))
;//! \htmlinclude ScaraKinFK-response.msg.html

(cl:defclass <ScaraKinFK-response> (roslisp-msg-protocol:ros-message)
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

(cl:defclass ScaraKinFK-response (<ScaraKinFK-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ScaraKinFK-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ScaraKinFK-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name scara_command-srv:<ScaraKinFK-response> is deprecated: use scara_command-srv:ScaraKinFK-response instead.")))

(cl:ensure-generic-function 'x-val :lambda-list '(m))
(cl:defmethod x-val ((m <ScaraKinFK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:x-val is deprecated.  Use scara_command-srv:x instead.")
  (x m))

(cl:ensure-generic-function 'y-val :lambda-list '(m))
(cl:defmethod y-val ((m <ScaraKinFK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:y-val is deprecated.  Use scara_command-srv:y instead.")
  (y m))

(cl:ensure-generic-function 'z-val :lambda-list '(m))
(cl:defmethod z-val ((m <ScaraKinFK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:z-val is deprecated.  Use scara_command-srv:z instead.")
  (z m))

(cl:ensure-generic-function 'phi-val :lambda-list '(m))
(cl:defmethod phi-val ((m <ScaraKinFK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:phi-val is deprecated.  Use scara_command-srv:phi instead.")
  (phi m))

(cl:ensure-generic-function 'theta-val :lambda-list '(m))
(cl:defmethod theta-val ((m <ScaraKinFK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:theta-val is deprecated.  Use scara_command-srv:theta instead.")
  (theta m))

(cl:ensure-generic-function 'psi-val :lambda-list '(m))
(cl:defmethod psi-val ((m <ScaraKinFK-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader scara_command-srv:psi-val is deprecated.  Use scara_command-srv:psi instead.")
  (psi m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ScaraKinFK-response>) ostream)
  "Serializes a message object of type '<ScaraKinFK-response>"
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
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ScaraKinFK-response>) istream)
  "Deserializes a message object of type '<ScaraKinFK-response>"
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ScaraKinFK-response>)))
  "Returns string type for a service object of type '<ScaraKinFK-response>"
  "scara_command/ScaraKinFKResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ScaraKinFK-response)))
  "Returns string type for a service object of type 'ScaraKinFK-response"
  "scara_command/ScaraKinFKResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ScaraKinFK-response>)))
  "Returns md5sum for a message object of type '<ScaraKinFK-response>"
  "7861a4dbb7e9452dabc20f2eff991915")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ScaraKinFK-response)))
  "Returns md5sum for a message object of type 'ScaraKinFK-response"
  "7861a4dbb7e9452dabc20f2eff991915")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ScaraKinFK-response>)))
  "Returns full string definition for message of type '<ScaraKinFK-response>"
  (cl:format cl:nil "float32 x~%float32 y~%float32 z~%float32 phi~%float32 theta~%float32 psi~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ScaraKinFK-response)))
  "Returns full string definition for message of type 'ScaraKinFK-response"
  (cl:format cl:nil "float32 x~%float32 y~%float32 z~%float32 phi~%float32 theta~%float32 psi~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ScaraKinFK-response>))
  (cl:+ 0
     4
     4
     4
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ScaraKinFK-response>))
  "Converts a ROS message object to a list"
  (cl:list 'ScaraKinFK-response
    (cl:cons ':x (x msg))
    (cl:cons ':y (y msg))
    (cl:cons ':z (z msg))
    (cl:cons ':phi (phi msg))
    (cl:cons ':theta (theta msg))
    (cl:cons ':psi (psi msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'ScaraKinFK)))
  'ScaraKinFK-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'ScaraKinFK)))
  'ScaraKinFK-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ScaraKinFK)))
  "Returns string type for a service object of type '<ScaraKinFK>"
  "scara_command/ScaraKinFK")