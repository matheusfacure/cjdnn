(ns cjdnn.loss-test
  (:require [clojure.test :refer :all]
            [cjdnn.test-helpers :refer :all]
            [cjdnn.loss :refer :all]
            [clojure.core.matrix :as m]))


(deftest test-softmax-corss-entropy
  (testing "softmax-corss-entropy"
    (let [input  [[-1.0 1.0 2.0] [4.0 5.0 6.0]]
          y-true [[0.0 0.0 1.0]  [1.0 0.0 0.0]]
          {foreword :foreword
           backward :backward} (softmax-cross-entropy)
          [y-hat cache] (foreword input)
          y-hat-exp [[0.0351190 0.2594964 0.705384]
                     [0.0900305 0.2447284 0.665240]]
          d-result (backward y-true y-hat-exp)
          d-exp [[0.035119 0.2594964 -0.294616]
                 [-0.90996 0.2447284 0.66524]]]
      (is (= cache nil))
      (is (close? d-result d-exp 1e-4))
      (is (close? y-hat y-hat-exp 1e-4)))))


(run-all-tests #"cjdnn.loss-test")
