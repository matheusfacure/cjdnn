(ns cjdnn.core-test
  (:require [clojure.test :refer :all]
            [cjdnn.core :refer :all]))

(deftest a-test
  (testing "Dummy test"
    (is (= 0 0))))


(run-all-tests #"cjdnn.core-test")
