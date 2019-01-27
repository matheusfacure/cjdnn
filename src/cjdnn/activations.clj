(ns cjdnn.activations
  (:require
    [clojure.core.matrix :as m]))

(defn relu
  [tensor]
  (let [activation #(max 0.0 %)]
    (m/emap activation tensor)))

(defn d-relu
  [tensor]
  (m/eif (m/lt tensor 0) 0.0 1.0))

(defn softmax-v
  [V]
  (let [exps (->> V m/maximum (m/sub V) m/exp)]
    (m/div exps (m/esum exps))))

(defn softmax
  [M]
  (->> M m/rows (map softmax-v)))

