;; Auto-generated. Do not edit!


(when (boundp 'simple_laserscan::SimpleScan)
  (if (not (find-package "SIMPLE_LASERSCAN"))
    (make-package "SIMPLE_LASERSCAN"))
  (shadow 'SimpleScan (find-package "SIMPLE_LASERSCAN")))
(unless (find-package "SIMPLE_LASERSCAN::SIMPLESCAN")
  (make-package "SIMPLE_LASERSCAN::SIMPLESCAN"))

(in-package "ROS")
;;//! \htmlinclude SimpleScan.msg.html


(defclass simple_laserscan::SimpleScan
  :super ros::object
  :slots (_stamp _number _data ))

(defmethod simple_laserscan::SimpleScan
  (:init
   (&key
    ((:stamp __stamp) (instance ros::time :init))
    ((:number __number) 0)
    ((:data __data) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _stamp __stamp)
   (setq _number (round __number))
   (setq _data __data)
   self)
  (:stamp
   (&optional __stamp)
   (if __stamp (setq _stamp __stamp)) _stamp)
  (:number
   (&optional __number)
   (if __number (setq _number __number)) _number)
  (:data
   (&optional __data)
   (if __data (setq _data __data)) _data)
  (:serialization-length
   ()
   (+
    ;; time _stamp
    8
    ;; int64 _number
    8
    ;; float64[] _data
    (* 8    (length _data)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; time _stamp
       (write-long (send _stamp :sec) s) (write-long (send _stamp :nsec) s)
     ;; int64 _number
#+(or :alpha :irix6 :x86_64)
       (progn (sys::poke _number (send s :buffer) (send s :count) :long) (incf (stream-count s) 8))
#-(or :alpha :irix6 :x86_64)
       (cond ((and (class _number) (= (length (_number . bv)) 2)) ;; bignum
              (write-long (ash (elt (_number . bv) 0) 0) s)
              (write-long (ash (elt (_number . bv) 1) -1) s))
             ((and (class _number) (= (length (_number . bv)) 1)) ;; big1
              (write-long (elt (_number . bv) 0) s)
              (write-long (if (>= _number 0) 0 #xffffffff) s))
             (t                                         ;; integer
              (write-long _number s)(write-long (if (>= _number 0) 0 #xffffffff) s)))
     ;; float64[] _data
     (write-long (length _data) s)
     (dotimes (i (length _data))
       (sys::poke (elt _data i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; time _stamp
     (send _stamp :sec (sys::peek buf ptr- :integer)) (incf ptr- 4)  (send _stamp :nsec (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int64 _number
#+(or :alpha :irix6 :x86_64)
      (setf _number (prog1 (sys::peek buf ptr- :long) (incf ptr- 8)))
#-(or :alpha :irix6 :x86_64)
      (setf _number (let ((b0 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4)))
                  (b1 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4))))
              (cond ((= b1 -1) b0)
                     ((and (= b1  0)
                           (<= lisp::most-negative-fixnum b0 lisp::most-positive-fixnum))
                      b0)
                    ((= b1  0) (make-instance bignum :size 1 :bv (integer-vector b0)))
                    (t (make-instance bignum :size 2 :bv (integer-vector b0 (ash b1 1)))))))
   ;; float64[] _data
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _data (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _data i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;;
   self)
  )

(setf (get simple_laserscan::SimpleScan :md5sum-) "fd4bca9753f42dc52d36cbb8a7c34ff6")
(setf (get simple_laserscan::SimpleScan :datatype-) "simple_laserscan/SimpleScan")
(setf (get simple_laserscan::SimpleScan :definition-)
      "time stamp
int64 number
float64[] data

")



(provide :simple_laserscan/SimpleScan "fd4bca9753f42dc52d36cbb8a7c34ff6")


