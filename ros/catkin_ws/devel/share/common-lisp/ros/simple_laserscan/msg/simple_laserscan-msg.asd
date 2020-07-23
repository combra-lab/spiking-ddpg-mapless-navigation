
(cl:in-package :asdf)

(defsystem "simple_laserscan-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "SimpleScan" :depends-on ("_package_SimpleScan"))
    (:file "_package_SimpleScan" :depends-on ("_package"))
  ))