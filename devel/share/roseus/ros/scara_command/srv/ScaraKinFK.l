;; Auto-generated. Do not edit!


(when (boundp 'scara_command::ScaraKinFK)
  (if (not (find-package "SCARA_COMMAND"))
    (make-package "SCARA_COMMAND"))
  (shadow 'ScaraKinFK (find-package "SCARA_COMMAND")))
(unless (find-package "SCARA_COMMAND::SCARAKINFK")
  (make-package "SCARA_COMMAND::SCARAKINFK"))
(unless (find-package "SCARA_COMMAND::SCARAKINFKREQUEST")
  (make-package "SCARA_COMMAND::SCARAKINFKREQUEST"))
(unless (find-package "SCARA_COMMAND::SCARAKINFKRESPONSE")
  (make-package "SCARA_COMMAND::SCARAKINFKRESPONSE"))

(in-package "ROS")





(defclass scara_command::ScaraKinFKRequest
  :super ros::object
  :slots (_q1 _q2 _q3 ))

(defmethod scara_command::ScaraKinFKRequest
  (:init
   (&key
    ((:q1 __q1) 0.0)
    ((:q2 __q2) 0.0)
    ((:q3 __q3) 0.0)
    )
   (send-super :init)
   (setq _q1 (float __q1))
   (setq _q2 (float __q2))
   (setq _q3 (float __q3))
   self)
  (:q1
   (&optional __q1)
   (if __q1 (setq _q1 __q1)) _q1)
  (:q2
   (&optional __q2)
   (if __q2 (setq _q2 __q2)) _q2)
  (:q3
   (&optional __q3)
   (if __q3 (setq _q3 __q3)) _q3)
  (:serialization-length
   ()
   (+
    ;; float32 _q1
    4
    ;; float32 _q2
    4
    ;; float32 _q3
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _q1
       (sys::poke _q1 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _q2
       (sys::poke _q2 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _q3
       (sys::poke _q3 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _q1
     (setq _q1 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _q2
     (setq _q2 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _q3
     (setq _q3 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(defclass scara_command::ScaraKinFKResponse
  :super ros::object
  :slots (_x _y _z _phi _theta _psi ))

(defmethod scara_command::ScaraKinFKResponse
  (:init
   (&key
    ((:x __x) 0.0)
    ((:y __y) 0.0)
    ((:z __z) 0.0)
    ((:phi __phi) 0.0)
    ((:theta __theta) 0.0)
    ((:psi __psi) 0.0)
    )
   (send-super :init)
   (setq _x (float __x))
   (setq _y (float __y))
   (setq _z (float __z))
   (setq _phi (float __phi))
   (setq _theta (float __theta))
   (setq _psi (float __psi))
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
  (:phi
   (&optional __phi)
   (if __phi (setq _phi __phi)) _phi)
  (:theta
   (&optional __theta)
   (if __theta (setq _theta __theta)) _theta)
  (:psi
   (&optional __psi)
   (if __psi (setq _psi __psi)) _psi)
  (:serialization-length
   ()
   (+
    ;; float32 _x
    4
    ;; float32 _y
    4
    ;; float32 _z
    4
    ;; float32 _phi
    4
    ;; float32 _theta
    4
    ;; float32 _psi
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
     ;; float32 _phi
       (sys::poke _phi (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _theta
       (sys::poke _theta (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _psi
       (sys::poke _psi (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
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
   ;; float32 _phi
     (setq _phi (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _theta
     (setq _theta (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _psi
     (setq _psi (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(defclass scara_command::ScaraKinFK
  :super ros::object
  :slots ())

(setf (get scara_command::ScaraKinFK :md5sum-) "7861a4dbb7e9452dabc20f2eff991915")
(setf (get scara_command::ScaraKinFK :datatype-) "scara_command/ScaraKinFK")
(setf (get scara_command::ScaraKinFK :request) scara_command::ScaraKinFKRequest)
(setf (get scara_command::ScaraKinFK :response) scara_command::ScaraKinFKResponse)

(defmethod scara_command::ScaraKinFKRequest
  (:response () (instance scara_command::ScaraKinFKResponse :init)))

(setf (get scara_command::ScaraKinFKRequest :md5sum-) "7861a4dbb7e9452dabc20f2eff991915")
(setf (get scara_command::ScaraKinFKRequest :datatype-) "scara_command/ScaraKinFKRequest")
(setf (get scara_command::ScaraKinFKRequest :definition-)
      "float32 q1
float32 q2
float32 q3
---
float32 x
float32 y
float32 z
float32 phi
float32 theta
float32 psi
")

(setf (get scara_command::ScaraKinFKResponse :md5sum-) "7861a4dbb7e9452dabc20f2eff991915")
(setf (get scara_command::ScaraKinFKResponse :datatype-) "scara_command/ScaraKinFKResponse")
(setf (get scara_command::ScaraKinFKResponse :definition-)
      "float32 q1
float32 q2
float32 q3
---
float32 x
float32 y
float32 z
float32 phi
float32 theta
float32 psi
")



(provide :scara_command/ScaraKinFK "7861a4dbb7e9452dabc20f2eff991915")


