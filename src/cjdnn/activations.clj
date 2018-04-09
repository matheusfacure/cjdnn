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

(def sigmoid m/logistic)

(defn d-sigmoid
  [tensor]
  (m/emul (sigmoid tensor) (m/sub 1 (sigmoid tensor))))

(defn softmax
  [V]
  (let [exps (->> V m/maximum (m/sub V) m/exp)]
    (m/div exps (m/esum exps))))

;def stable_softmax(X):
;  exps = np.exp(X - np.max(X))
;  return exps / np.sum(exps)
