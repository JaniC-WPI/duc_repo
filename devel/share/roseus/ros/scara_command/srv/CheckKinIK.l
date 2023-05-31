;; Auto-generated. Do not edit!


(when (boundp 'scara_command::CheckKinIK)
  (if (not (find-package "SCARA_COMMAND"))
    (make-package "SCARA_COMMAND"))
  (shadow 'CheckKinIK (find-package "SCARA_COMMAND")))
(unless (find-package "SCARA_COMMAND::CHECKKINIK")
  (make-package "SCARA_COMMAND::CHECKKINIK"))
(unless (find-package "SCARA_COMMAND::CHECKKINIKREQUEST")
  (make-package "SCARA_COMMAND::CHECKKINIKREQUEST"))
(unless (find-package "SCARA_COMMAND::CHECKKINIKRESPONSE")
  (make-package "SCARA_COMMAND::CHECKKINIKRESPONSE"))

(in-package "ROS")





(defclass scara_command::CheckKinIKRequest
  :super ros::object
  :slots ())

(defmethod scara_command::CheckKinIKRequest
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass scara_command::CheckKinIKResponse
  :super ros::object
  :slots (_state_q1 _state_q2 _state_q3 _ik_q1 _ik_q2 _ik_q3 _correct ))

(defmethod scara_command::CheckKinIKResponse
  (:init
   (&key
    ((:state_q1 __state_q1) 0.0)
    ((:state_q2 __state_q2) 0.0)
    ((:state_q3 __state_q3) 0.0)
    ((:ik_q1 __ik_q1) 0.0)
    ((:ik_q2 __ik_q2) 0.0)
    ((:ik_q3 __ik_q3) 0.0)
    ((:correct __correct) nil)
    )
   (send-super :init)
   (setq _state_q1 (float __state_q1))
   (setq _state_q2 (float __state_q2))
   (setq _state_q3 (float __state_q3))
   (setq _ik_q1 (float __ik_q1))
   (setq _ik_q2 (float __ik_q2))
   (setq _ik_q3 (float __ik_q3))
   (setq _correct __correct)
   self)
  (:state_q1
   (&optional __state_q1)
   (if __state_q1 (setq _state_q1 __state_q1)) _state_q1)
  (:state_q2
   (&optional __state_q2)
   (if __state_q2 (setq _state_q2 __state_q2)) _state_q2)
  (:state_q3
   (&optional __state_q3)
   (if __state_q3 (setq _state_q3 __state_q3)) _state_q3)
  (:ik_q1
   (&optional __ik_q1)
   (if __ik_q1 (setq _ik_q1 __ik_q1)) _ik_q1)
  (:ik_q2
   (&optional __ik_q2)
   (if __ik_q2 (setq _ik_q2 __ik_q2)) _ik_q2)
  (:ik_q3
   (&optional __ik_q3)
   (if __ik_q3 (setq _ik_q3 __ik_q3)) _ik_q3)
  (:correct
   (&optional (__correct :null))
   (if (not (eq __correct :null)) (setq _correct __correct)) _correct)
  (:serialization-length
   ()
   (+
    ;; float32 _state_q1
    4
    ;; float32 _state_q2
    4
    ;; float32 _state_q3
    4
    ;; float32 _ik_q1
    4
    ;; float32 _ik_q2
    4
    ;; float32 _ik_q3
    4
    ;; bool _correct
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _state_q1
       (sys::poke _state_q1 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _state_q2
       (sys::poke _state_q2 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _state_q3
       (sys::poke _state_q3 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _ik_q1
       (sys::poke _ik_q1 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _ik_q2
       (sys::poke _ik_q2 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _ik_q3
       (sys::poke _ik_q3 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; bool _correct
       (if _correct (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _state_q1
     (setq _state_q1 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _state_q2
     (setq _state_q2 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _state_q3
     (setq _state_q3 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _ik_q1
     (setq _ik_q1 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _ik_q2
     (setq _ik_q2 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _ik_q3
     (setq _ik_q3 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; bool _correct
     (setq _correct (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass scara_command::CheckKinIK
  :super ros::object
  :slots ())

(setf (get scara_command::CheckKinIK :md5sum-) "36714075758334348b25a38f8ec94251")
(setf (get scara_command::CheckKinIK :datatype-) "scara_command/CheckKinIK")
(setf (get scara_command::CheckKinIK :request) scara_command::CheckKinIKRequest)
(setf (get scara_command::CheckKinIK :response) scara_command::CheckKinIKResponse)

(defmethod scara_command::CheckKinIKRequest
  (:response () (instance scara_command::CheckKinIKResponse :init)))

(setf (get scara_command::CheckKinIKRequest :md5sum-) "36714075758334348b25a38f8ec94251")
(setf (get scara_command::CheckKinIKRequest :datatype-) "scara_command/CheckKinIKRequest")
(setf (get scara_command::CheckKinIKRequest :definition-)
      "---
float32 state_q1
float32 state_q2
float32 state_q3
float32 ik_q1
float32 ik_q2
float32 ik_q3
bool correct

")

(setf (get scara_command::CheckKinIKResponse :md5sum-) "36714075758334348b25a38f8ec94251")
(setf (get scara_command::CheckKinIKResponse :datatype-) "scara_command/CheckKinIKResponse")
(setf (get scara_command::CheckKinIKResponse :definition-)
      "---
float32 state_q1
float32 state_q2
float32 state_q3
float32 ik_q1
float32 ik_q2
float32 ik_q3
bool correct

")



(provide :scara_command/CheckKinIK "36714075758334348b25a38f8ec94251")


