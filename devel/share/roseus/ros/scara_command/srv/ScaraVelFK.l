;; Auto-generated. Do not edit!


(when (boundp 'scara_command::ScaraVelFK)
  (if (not (find-package "SCARA_COMMAND"))
    (make-package "SCARA_COMMAND"))
  (shadow 'ScaraVelFK (find-package "SCARA_COMMAND")))
(unless (find-package "SCARA_COMMAND::SCARAVELFK")
  (make-package "SCARA_COMMAND::SCARAVELFK"))
(unless (find-package "SCARA_COMMAND::SCARAVELFKREQUEST")
  (make-package "SCARA_COMMAND::SCARAVELFKREQUEST"))
(unless (find-package "SCARA_COMMAND::SCARAVELFKRESPONSE")
  (make-package "SCARA_COMMAND::SCARAVELFKRESPONSE"))

(in-package "ROS")





(defclass scara_command::ScaraVelFKRequest
  :super ros::object
  :slots (_q1 _q2 _q3 _q1_dot _q2_dot _q3_dot ))

(defmethod scara_command::ScaraVelFKRequest
  (:init
   (&key
    ((:q1 __q1) 0.0)
    ((:q2 __q2) 0.0)
    ((:q3 __q3) 0.0)
    ((:q1_dot __q1_dot) 0.0)
    ((:q2_dot __q2_dot) 0.0)
    ((:q3_dot __q3_dot) 0.0)
    )
   (send-super :init)
   (setq _q1 (float __q1))
   (setq _q2 (float __q2))
   (setq _q3 (float __q3))
   (setq _q1_dot (float __q1_dot))
   (setq _q2_dot (float __q2_dot))
   (setq _q3_dot (float __q3_dot))
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
  (:q1_dot
   (&optional __q1_dot)
   (if __q1_dot (setq _q1_dot __q1_dot)) _q1_dot)
  (:q2_dot
   (&optional __q2_dot)
   (if __q2_dot (setq _q2_dot __q2_dot)) _q2_dot)
  (:q3_dot
   (&optional __q3_dot)
   (if __q3_dot (setq _q3_dot __q3_dot)) _q3_dot)
  (:serialization-length
   ()
   (+
    ;; float32 _q1
    4
    ;; float32 _q2
    4
    ;; float32 _q3
    4
    ;; float32 _q1_dot
    4
    ;; float32 _q2_dot
    4
    ;; float32 _q3_dot
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
     ;; float32 _q1_dot
       (sys::poke _q1_dot (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _q2_dot
       (sys::poke _q2_dot (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _q3_dot
       (sys::poke _q3_dot (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
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
   ;; float32 _q1_dot
     (setq _q1_dot (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _q2_dot
     (setq _q2_dot (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _q3_dot
     (setq _q3_dot (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(defclass scara_command::ScaraVelFKResponse
  :super ros::object
  :slots (_Vx _Vy _Vz _Wx _Wy _Wz ))

(defmethod scara_command::ScaraVelFKResponse
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

(defclass scara_command::ScaraVelFK
  :super ros::object
  :slots ())

(setf (get scara_command::ScaraVelFK :md5sum-) "d154cd82f0b9fb9f1cf33ba88636c898")
(setf (get scara_command::ScaraVelFK :datatype-) "scara_command/ScaraVelFK")
(setf (get scara_command::ScaraVelFK :request) scara_command::ScaraVelFKRequest)
(setf (get scara_command::ScaraVelFK :response) scara_command::ScaraVelFKResponse)

(defmethod scara_command::ScaraVelFKRequest
  (:response () (instance scara_command::ScaraVelFKResponse :init)))

(setf (get scara_command::ScaraVelFKRequest :md5sum-) "d154cd82f0b9fb9f1cf33ba88636c898")
(setf (get scara_command::ScaraVelFKRequest :datatype-) "scara_command/ScaraVelFKRequest")
(setf (get scara_command::ScaraVelFKRequest :definition-)
      "float32 q1
float32 q2
float32 q3
float32 q1_dot
float32 q2_dot
float32 q3_dot
---
float32 Vx
float32 Vy
float32 Vz
float32 Wx
float32 Wy
float32 Wz

")

(setf (get scara_command::ScaraVelFKResponse :md5sum-) "d154cd82f0b9fb9f1cf33ba88636c898")
(setf (get scara_command::ScaraVelFKResponse :datatype-) "scara_command/ScaraVelFKResponse")
(setf (get scara_command::ScaraVelFKResponse :definition-)
      "float32 q1
float32 q2
float32 q3
float32 q1_dot
float32 q2_dot
float32 q3_dot
---
float32 Vx
float32 Vy
float32 Vz
float32 Wx
float32 Wy
float32 Wz

")



(provide :scara_command/ScaraVelFK "d154cd82f0b9fb9f1cf33ba88636c898")


