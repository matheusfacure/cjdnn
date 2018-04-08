(ns cjdnn.activations
  (:require
    [clojure.core.matrix :as m]))

(defn relu
  [tensor]
  (let [activation #(max 0 %)]
    (m/emap activation tensor)))

(defn d-relu
  [tensor]
  (m/eif (m/lt tensor 0) 0.0 1.0))

(defn stable-logistic
  [x]
  (if (>= 0)
    (let [z (Math/exp (- x))]
      (/ 1 (+ 1 z)))
    (let [z (Math/exp x)]
      (/ z (+ 1 z)))))

(defn sigmoid
  [tensor]
  (let [activation stable-logistic]
    (m/emap activation tensor)))

