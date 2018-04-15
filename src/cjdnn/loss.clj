(ns cjdnn.loss
  (:require [clojure.core.matrix :as m]
            [cjdnn.activations :as act]))

(defn softmax-cross-entropy
  []
  {:foreword (fn [z] [(-> z m/rows act/softmax) nil])
   :backward (fn [y-true y-hat] (m/sub y-hat y-true))})
