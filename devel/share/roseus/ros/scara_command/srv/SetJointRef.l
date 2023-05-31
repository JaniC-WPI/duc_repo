;; Auto-generated. Do not edit!


(when (boundp 'scara_command::SetJointRef)
  (if (not (find-package "SCARA_COMMAND"))
    (make-package "SCARA_COMMAND"))
  (shadow 'SetJointRef (find-package "SCARA_COMMAND")))
(unless (find-package "SCARA_COMMAND::SETJOINTREF")
  (make-package "SCARA_COMMAND::SETJOINTREF"))
(unless (find-package "SCARA_COMMAND::SETJOINTREFREQUEST")
  (make-package "SCARA_COMMAND::SETJOINTREFREQUEST"))
(unless (find-package "SCARA_COMMAND::SETJOINTREFRESPONSE")
  (make-package "SCARA_COMMAND::SETJOINTREFRESPONSE"))

(in-package "ROS")





(defclass scara_command::SetJointRefRequest
  :super ros::object
  :slots (_joint_name _ref ))

(defmethod scara_command::SetJointRefRequest
  (:init
   (&key
    ((:joint_name __joint_name) "")
    ((:ref __ref) 0.0)
    )
   (send-super :init)
   (setq _joint_name (string __joint_name))
   (setq _ref (float __ref))
   self)
  (:joint_name
   (&optional __joint_name)
   (if __joint_name (setq _joint_name __joint_name)) _joint_name)
  (:ref
   (&optional __ref)
   (if __ref (setq _ref __ref)) _ref)
  (:serialization-length
   ()
   (+
    ;; string _joint_name
    4 (length _joint_name)
    ;; float64 _ref
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _joint_name
       (write-long (length _joint_name) s) (princ _joint_name s)
     ;; float64 _ref
       (sys::poke _ref (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _joint_name
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _joint_name (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; float64 _ref
     (setq _ref (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(defclass scara_command::SetJointRefResponse
  :super ros::object
  :slots (_success ))

(defmethod scara_command::SetJointRefResponse
  (:init
   (&key
    ((:success __success) nil)
    )
   (send-super :init)
   (setq _success __success)
   self)
  (:success
   (&optional (__success :null))
   (if (not (eq __success :null)) (setq _success __success)) _success)
  (:serialization-length
   ()
   (+
    ;; bool _success
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass scara_command::SetJointRef
  :super ros::object
  :slots ())

(setf (get scara_command::SetJointRef :md5sum-) "6f194d3e831f68db4bba0c86e04a9975")
(setf (get scara_command::SetJointRef :datatype-) "scara_command/SetJointRef")
(setf (get scara_command::SetJointRef :request) scara_command::SetJointRefRequest)
(setf (get scara_command::SetJointRef :response) scara_command::SetJointRefResponse)

(defmethod scara_command::SetJointRefRequest
  (:response () (instance scara_command::SetJointRefResponse :init)))

(setf (get scara_command::SetJointRefRequest :md5sum-) "6f194d3e831f68db4bba0c86e04a9975")
(setf (get scara_command::SetJointRefRequest :datatype-) "scara_command/SetJointRefRequest")
(setf (get scara_command::SetJointRefRequest :definition-)
      "string joint_name
float64 ref
---
bool success

")

(setf (get scara_command::SetJointRefResponse :md5sum-) "6f194d3e831f68db4bba0c86e04a9975")
(setf (get scara_command::SetJointRefResponse :datatype-) "scara_command/SetJointRefResponse")
(setf (get scara_command::SetJointRefResponse :definition-)
      "string joint_name
float64 ref
---
bool success

")



(provide :scara_command/SetJointRef "6f194d3e831f68db4bba0c86e04a9975")


