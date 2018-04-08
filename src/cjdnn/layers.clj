(ns cjdnn.layers
  (:require [clojure.core.matrix.random :as rnd]
            [clojure.core.matrix :as m]
            [cjdnn.activations :as act]))

(defn linear
  [input-shape neurons]
   (let [W (atom (rnd/sample-normal [input-shape neurons]))]
     (let [foreword (fn [z] [(m/mmul z @W) z])
           backward (fn [d] (m/mmul d (m/transpose @W)))
           update!  (fn [lr d cache]
                     (let [grad (-> cache
                                    m/transpose
                                    (m/mmul d)
                                    (m/div (first (m/shape d)))
                                    (m/mul lr))]
                       (swap! m/sub grad)))]
       {:foreword foreword
        :backward backward
        :update!  update!})))


(defn relu
  []

  {:foreword (fn [z] [(act/relu z) nil])
   :backward act/d-relu})
