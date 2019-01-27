(ns cjdnn.activations-test
  (:require [clojure.test :refer :all]
            [cjdnn.test-helpers :refer :all]
            [cjdnn.activations :refer :all]
            [clojure.core.matrix :as m]))


(deftest test-relu
  (testing "ReLU test"
    (let [input[[1.0 2.0] [-1.0 2.0]]
          result (relu input)
          expected [[1.0 2.0] [0.0 2.0]]]
      (is (= result expected)))))

(deftest test-d-relu
  (testing "ReLU derivative test"
    (let [input [[1.0 2.0] [-1.0 2.0]]
          result (d-relu input)
          num-d (get-derivative relu)
          expected (num-d input)]
      (is (close? result expected 1e-5)))))

(deftest test-softmax
  (testing "softmax test"
    (let [input [[1.0 2.0 3.0] [4.0 5.0 6.0]]
          result (softmax input)
          expected [[0.090030 0.244728 0.665240]
                    [0.090030 0.244728 0.665240]]]
      (is (close? result expected 1e-4)))))

(run-all-tests #"cjdnn.activations-test")
