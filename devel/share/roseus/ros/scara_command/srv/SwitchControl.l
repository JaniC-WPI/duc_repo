;; Auto-generated. Do not edit!


(when (boundp 'scara_command::SwitchControl)
  (if (not (find-package "SCARA_COMMAND"))
    (make-package "SCARA_COMMAND"))
  (shadow 'SwitchControl (find-package "SCARA_COMMAND")))
(unless (find-package "SCARA_COMMAND::SWITCHCONTROL")
  (make-package "SCARA_COMMAND::SWITCHCONTROL"))
(unless (find-package "SCARA_COMMAND::SWITCHCONTROLREQUEST")
  (make-package "SCARA_COMMAND::SWITCHCONTROLREQUEST"))
(unless (find-package "SCARA_COMMAND::SWITCHCONTROLRESPONSE")
  (make-package "SCARA_COMMAND::SWITCHCONTROLRESPONSE"))

(in-package "ROS")





(defclass scara_command::SwitchControlRequest
  :super ros::object
  :slots (_command ))

(defmethod scara_command::SwitchControlRequest
  (:init
   (&key
    ((:command __command) "")
    )
   (send-super :init)
   (setq _command (string __command))
   self)
  (:command
   (&optional __command)
   (if __command (setq _command __command)) _command)
  (:serialization-length
   ()
   (+
    ;; string _command
    4 (length _command)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _command
       (write-long (length _command) s) (princ _command s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _command
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _command (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass scara_command::SwitchControlResponse
  :super ros::object
  :slots (_success ))

(defmethod scara_command::SwitchControlResponse
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

(defclass scara_command::SwitchControl
  :super ros::object
  :slots ())

(setf (get scara_command::SwitchControl :md5sum-) "031d24522d462b2314fd1b6270670dd2")
(setf (get scara_command::SwitchControl :datatype-) "scara_command/SwitchControl")
(setf (get scara_command::SwitchControl :request) scara_command::SwitchControlRequest)
(setf (get scara_command::SwitchControl :response) scara_command::SwitchControlResponse)

(defmethod scara_command::SwitchControlRequest
  (:response () (instance scara_command::SwitchControlResponse :init)))

(setf (get scara_command::SwitchControlRequest :md5sum-) "031d24522d462b2314fd1b6270670dd2")
(setf (get scara_command::SwitchControlRequest :datatype-) "scara_command/SwitchControlRequest")
(setf (get scara_command::SwitchControlRequest :definition-)
      "string command
---
bool success

")

(setf (get scara_command::SwitchControlResponse :md5sum-) "031d24522d462b2314fd1b6270670dd2")
(setf (get scara_command::SwitchControlResponse :datatype-) "scara_command/SwitchControlResponse")
(setf (get scara_command::SwitchControlResponse :definition-)
      "string command
---
bool success

")



(provide :scara_command/SwitchControl "031d24522d462b2314fd1b6270670dd2")


