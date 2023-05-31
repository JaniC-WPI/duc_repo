;; Auto-generated. Do not edit!


(when (boundp 'scara_command::CheckKinFK)
  (if (not (find-package "SCARA_COMMAND"))
    (make-package "SCARA_COMMAND"))
  (shadow 'CheckKinFK (find-package "SCARA_COMMAND")))
(unless (find-package "SCARA_COMMAND::CHECKKINFK")
  (make-package "SCARA_COMMAND::CHECKKINFK"))
(unless (find-package "SCARA_COMMAND::CHECKKINFKREQUEST")
  (make-package "SCARA_COMMAND::CHECKKINFKREQUEST"))
(unless (find-package "SCARA_COMMAND::CHECKKINFKRESPONSE")
  (make-package "SCARA_COMMAND::CHECKKINFKRESPONSE"))

(in-package "ROS")





(defclass scara_command::CheckKinFKRequest
  :super ros::object
  :slots ())

(defmethod scara_command::CheckKinFKRequest
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

(defclass scara_command::CheckKinFKResponse
  :super ros::object
  :slots (_state_x _state_y _state_z _state_phi _state_theta _state_psi _fk_x _fk_y _fk_z _fk_phi _fk_theta _fk_psi _correct ))

(defmethod scara_command::CheckKinFKResponse
  (:init
   (&key
    ((:state_x __state_x) 0.0)
    ((:state_y __state_y) 0.0)
    ((:state_z __state_z) 0.0)
    ((:state_phi __state_phi) 0.0)
    ((:state_theta __state_theta) 0.0)
    ((:state_psi __state_psi) 0.0)
    ((:fk_x __fk_x) 0.0)
    ((:fk_y __fk_y) 0.0)
    ((:fk_z __fk_z) 0.0)
    ((:fk_phi __fk_phi) 0.0)
    ((:fk_theta __fk_theta) 0.0)
    ((:fk_psi __fk_psi) 0.0)
    ((:correct __correct) nil)
    )
   (send-super :init)
   (setq _state_x (float __state_x))
   (setq _state_y (float __state_y))
   (setq _state_z (float __state_z))
   (setq _state_phi (float __state_phi))
   (setq _state_theta (float __state_theta))
   (setq _state_psi (float __state_psi))
   (setq _fk_x (float __fk_x))
   (setq _fk_y (float __fk_y))
   (setq _fk_z (float __fk_z))
   (setq _fk_phi (float __fk_phi))
   (setq _fk_theta (float __fk_theta))
   (setq _fk_psi (float __fk_psi))
   (setq _correct __correct)
   self)
  (:state_x
   (&optional __state_x)
   (if __state_x (setq _state_x __state_x)) _state_x)
  (:state_y
   (&optional __state_y)
   (if __state_y (setq _state_y __state_y)) _state_y)
  (:state_z
   (&optional __state_z)
   (if __state_z (setq _state_z __state_z)) _state_z)
  (:state_phi
   (&optional __state_phi)
   (if __state_phi (setq _state_phi __state_phi)) _state_phi)
  (:state_theta
   (&optional __state_theta)
   (if __state_theta (setq _state_theta __state_theta)) _state_theta)
  (:state_psi
   (&optional __state_psi)
   (if __state_psi (setq _state_psi __state_psi)) _state_psi)
  (:fk_x
   (&optional __fk_x)
   (if __fk_x (setq _fk_x __fk_x)) _fk_x)
  (:fk_y
   (&optional __fk_y)
   (if __fk_y (setq _fk_y __fk_y)) _fk_y)
  (:fk_z
   (&optional __fk_z)
   (if __fk_z (setq _fk_z __fk_z)) _fk_z)
  (:fk_phi
   (&optional __fk_phi)
   (if __fk_phi (setq _fk_phi __fk_phi)) _fk_phi)
  (:fk_theta
   (&optional __fk_theta)
   (if __fk_theta (setq _fk_theta __fk_theta)) _fk_theta)
  (:fk_psi
   (&optional __fk_psi)
   (if __fk_psi (setq _fk_psi __fk_psi)) _fk_psi)
  (:correct
   (&optional (__correct :null))
   (if (not (eq __correct :null)) (setq _correct __correct)) _correct)
  (:serialization-length
   ()
   (+
    ;; float32 _state_x
    4
    ;; float32 _state_y
    4
    ;; float32 _state_z
    4
    ;; float32 _state_phi
    4
    ;; float32 _state_theta
    4
    ;; float32 _state_psi
    4
    ;; float32 _fk_x
    4
    ;; float32 _fk_y
    4
    ;; float32 _fk_z
    4
    ;; float32 _fk_phi
    4
    ;; float32 _fk_theta
    4
    ;; float32 _fk_psi
    4
    ;; bool _correct
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _state_x
       (sys::poke _state_x (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _state_y
       (sys::poke _state_y (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _state_z
       (sys::poke _state_z (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _state_phi
       (sys::poke _state_phi (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _state_theta
       (sys::poke _state_theta (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _state_psi
       (sys::poke _state_psi (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _fk_x
       (sys::poke _fk_x (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _fk_y
       (sys::poke _fk_y (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _fk_z
       (sys::poke _fk_z (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _fk_phi
       (sys::poke _fk_phi (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _fk_theta
       (sys::poke _fk_theta (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _fk_psi
       (sys::poke _fk_psi (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; bool _correct
       (if _correct (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _state_x
     (setq _state_x (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _state_y
     (setq _state_y (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _state_z
     (setq _state_z (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _state_phi
     (setq _state_phi (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _state_theta
     (setq _state_theta (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _state_psi
     (setq _state_psi (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _fk_x
     (setq _fk_x (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _fk_y
     (setq _fk_y (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _fk_z
     (setq _fk_z (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _fk_phi
     (setq _fk_phi (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _fk_theta
     (setq _fk_theta (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _fk_psi
     (setq _fk_psi (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; bool _correct
     (setq _correct (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass scara_command::CheckKinFK
  :super ros::object
  :slots ())

(setf (get scara_command::CheckKinFK :md5sum-) "6dd4bd6e62545926c2a7502c0ee7c4f1")
(setf (get scara_command::CheckKinFK :datatype-) "scara_command/CheckKinFK")
(setf (get scara_command::CheckKinFK :request) scara_command::CheckKinFKRequest)
(setf (get scara_command::CheckKinFK :response) scara_command::CheckKinFKResponse)

(defmethod scara_command::CheckKinFKRequest
  (:response () (instance scara_command::CheckKinFKResponse :init)))

(setf (get scara_command::CheckKinFKRequest :md5sum-) "6dd4bd6e62545926c2a7502c0ee7c4f1")
(setf (get scara_command::CheckKinFKRequest :datatype-) "scara_command/CheckKinFKRequest")
(setf (get scara_command::CheckKinFKRequest :definition-)
      "---
float32 state_x
float32 state_y
float32 state_z
float32 state_phi
float32 state_theta
float32 state_psi
float32 fk_x
float32 fk_y
float32 fk_z
float32 fk_phi
float32 fk_theta
float32 fk_psi
bool correct

")

(setf (get scara_command::CheckKinFKResponse :md5sum-) "6dd4bd6e62545926c2a7502c0ee7c4f1")
(setf (get scara_command::CheckKinFKResponse :datatype-) "scara_command/CheckKinFKResponse")
(setf (get scara_command::CheckKinFKResponse :definition-)
      "---
float32 state_x
float32 state_y
float32 state_z
float32 state_phi
float32 state_theta
float32 state_psi
float32 fk_x
float32 fk_y
float32 fk_z
float32 fk_phi
float32 fk_theta
float32 fk_psi
bool correct

")



(provide :scara_command/CheckKinFK "6dd4bd6e62545926c2a7502c0ee7c4f1")


