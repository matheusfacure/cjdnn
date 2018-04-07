(ns cjdnn.layers
  (:require [clojure.core.matrix.random :as rnd]
            [clojure.core.matrix :as m]
            [cjdnn.activations :as act]))


(defn dense
  [input-shape neurons]
   (let [W (rnd/sample-normal [input-shape neurons])]
     (let [foreward (fn [z] (m/mmul z W))
           backward (fn [d] (m/mmul d m/transpose W))
       )
     ))
