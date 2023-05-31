;; Auto-generated. Do not edit!


(when (boundp 'scara_command::SetCartesianVel)
  (if (not (find-package "SCARA_COMMAND"))
    (make-package "SCARA_COMMAND"))
  (shadow 'SetCartesianVel (find-package "SCARA_COMMAND")))
(unless (find-package "SCARA_COMMAND::SETCARTESIANVEL")
  (make-package "SCARA_COMMAND::SETCARTESIANVEL"))
(unless (find-package "SCARA_COMMAND::SETCARTESIANVELREQUEST")
  (make-package "SCARA_COMMAND::SETCARTESIANVELREQUEST"))
(unless (find-package "SCARA_COMMAND::SETCARTESIANVELRESPONSE")
  (make-package "SCARA_COMMAND::SETCARTESIANVELRESPONSE"))

(in-package "ROS")





(defclass scara_command::SetCartesianVelRequest
  :super ros::object
  :slots (_Vx _Vy _Vz _Wx _Wy _Wz ))

(defmethod scara_command::SetCartesianVelRequest
  (:init
   (&key
    ((:Vx __Vx) 0.0)
    ((:Vy __Vy) 0.0)
    ((:Vz __Vz) 0.0)
    ((:Wx __Wx) 0.0)
    ((:Wy __Wy) 0.0)
    ((:Wz __Wz) 0.0)
    )
   (send-super :init)
   (setq _Vx (float __Vx))
   (setq _Vy (float __Vy))
   (setq _Vz (float __Vz))
   (setq _Wx (float __Wx))
   (setq _Wy (float __Wy))
   (setq _Wz (float __Wz))
   self)
  (:Vx
   (&optional __Vx)
   (if __Vx (setq _Vx __Vx)) _Vx)
  (:Vy
   (&optional __Vy)
   (if __Vy (setq _Vy __Vy)) _Vy)
  (:Vz
   (&optional __Vz)
   (if __Vz (setq _Vz __Vz)) _Vz)
  (:Wx
   (&optional __Wx)
   (if __Wx (setq _Wx __Wx)) _Wx)
  (:Wy
   (&optional __Wy)
   (if __Wy (setq _Wy __Wy)) _Wy)
  (:Wz
   (&optional __Wz)
   (if __Wz (setq _Wz __Wz)) _Wz)
  (:serialization-length
   ()
   (+
    ;; float32 _Vx
    4
    ;; float32 _Vy
    4
    ;; float32 _Vz
    4
    ;; float32 _Wx
    4
    ;; float32 _Wy
    4
    ;; float32 _Wz
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _Vx
       (sys::poke _Vx (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _Vy
       (sys::poke _Vy (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _Vz
       (sys::poke _Vz (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _Wx
       (sys::poke _Wx (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _Wy
       (sys::poke _Wy (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _Wz
       (sys::poke _Wz (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _Vx
     (setq _Vx (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _Vy
     (setq _Vy (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _Vz
     (setq _Vz (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _Wx
     (setq _Wx (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _Wy
     (setq _Wy (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _Wz
     (setq _Wz (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(defclass scara_command::SetCartesianVelResponse
  :super ros::object
  :slots (_success ))

(defmethod scara_command::SetCartesianVelResponse
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

(defclass scara_command::SetCartesianVel
  :super ros::object
  :slots ())

(setf (get scara_command::SetCartesianVel :md5sum-) "283707b7ad403b8c31a22ddc99890608")
(setf (get scara_command::SetCartesianVel :datatype-) "scara_command/SetCartesianVel")
(setf (get scara_command::SetCartesianVel :request) scara_command::SetCartesianVelRequest)
(setf (get scara_command::SetCartesianVel :response) scara_command::SetCartesianVelResponse)

(defmethod scara_command::SetCartesianVelRequest
  (:response () (instance scara_command::SetCartesianVelResponse :init)))

(setf (get scara_command::SetCartesianVelRequest :md5sum-) "283707b7ad403b8c31a22ddc99890608")
(setf (get scara_command::SetCartesianVelRequest :datatype-) "scara_command/SetCartesianVelRequest")
(setf (get scara_command::SetCartesianVelRequest :definition-)
      "float32 Vx
float32 Vy
float32 Vz
float32 Wx
float32 Wy
float32 Wz
---
bool success

")

(setf (get scara_command::SetCartesianVelResponse :md5sum-) "283707b7ad403b8c31a22ddc99890608")
(setf (get scara_command::SetCartesianVelResponse :datatype-) "scara_command/SetCartesianVelResponse")
(setf (get scara_command::SetCartesianVelResponse :definition-)
      "float32 Vx
float32 Vy
float32 Vz
float32 Wx
float32 Wy
float32 Wz
---
bool success

")



(provide :scara_command/SetCartesianVel "283707b7ad403b8c31a22ddc99890608")


