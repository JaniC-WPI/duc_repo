;; Auto-generated. Do not edit!


(when (boundp 'scara_command::SetCartesianPos)
  (if (not (find-package "SCARA_COMMAND"))
    (make-package "SCARA_COMMAND"))
  (shadow 'SetCartesianPos (find-package "SCARA_COMMAND")))
(unless (find-package "SCARA_COMMAND::SETCARTESIANPOS")
  (make-package "SCARA_COMMAND::SETCARTESIANPOS"))
(unless (find-package "SCARA_COMMAND::SETCARTESIANPOSREQUEST")
  (make-package "SCARA_COMMAND::SETCARTESIANPOSREQUEST"))
(unless (find-package "SCARA_COMMAND::SETCARTESIANPOSRESPONSE")
  (make-package "SCARA_COMMAND::SETCARTESIANPOSRESPONSE"))

(in-package "ROS")





(defclass scara_command::SetCartesianPosRequest
  :super ros::object
  :slots (_x _y _z ))

(defmethod scara_command::SetCartesianPosRequest
  (:init
   (&key
    ((:x __x) 0.0)
    ((:y __y) 0.0)
    ((:z __z) 0.0)
    )
   (send-super :init)
   (setq _x (float __x))
   (setq _y (float __y))
   (setq _z (float __z))
   self)
  (:x
   (&optional __x)
   (if __x (setq _x __x)) _x)
  (:y
   (&optional __y)
   (if __y (setq _y __y)) _y)
  (:z
   (&optional __z)
   (if __z (setq _z __z)) _z)
  (:serialization-length
   ()
   (+
    ;; float32 _x
    4
    ;; float32 _y
    4
    ;; float32 _z
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _x
       (sys::poke _x (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _y
       (sys::poke _y (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _z
       (sys::poke _z (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _x
     (setq _x (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _y
     (setq _y (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _z
     (setq _z (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(defclass scara_command::SetCartesianPosResponse
  :super ros::object
  :slots (_success ))

(defmethod scara_command::SetCartesianPosResponse
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

(defclass scara_command::SetCartesianPos
  :super ros::object
  :slots ())

(setf (get scara_command::SetCartesianPos :md5sum-) "58d59f258ca9f2d2ba375d9428a7f1de")
(setf (get scara_command::SetCartesianPos :datatype-) "scara_command/SetCartesianPos")
(setf (get scara_command::SetCartesianPos :request) scara_command::SetCartesianPosRequest)
(setf (get scara_command::SetCartesianPos :response) scara_command::SetCartesianPosResponse)

(defmethod scara_command::SetCartesianPosRequest
  (:response () (instance scara_command::SetCartesianPosResponse :init)))

(setf (get scara_command::SetCartesianPosRequest :md5sum-) "58d59f258ca9f2d2ba375d9428a7f1de")
(setf (get scara_command::SetCartesianPosRequest :datatype-) "scara_command/SetCartesianPosRequest")
(setf (get scara_command::SetCartesianPosRequest :definition-)
      "float32 x
float32 y
float32 z
---
bool success

")

(setf (get scara_command::SetCartesianPosResponse :md5sum-) "58d59f258ca9f2d2ba375d9428a7f1de")
(setf (get scara_command::SetCartesianPosResponse :datatype-) "scara_command/SetCartesianPosResponse")
(setf (get scara_command::SetCartesianPosResponse :definition-)
      "float32 x
float32 y
float32 z
---
bool success

")



(provide :scara_command/SetCartesianPos "58d59f258ca9f2d2ba375d9428a7f1de")


