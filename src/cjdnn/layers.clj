(ns cjdnn.layers
  (:require [clojure.core.matrix.random :as rnd]
            [clojure.core.matrix :as m]
            [cjdnn.activations :as act]))

(defn linear
  [input-shape neurons]
   (let [W (atom (m/scale (rnd/sample-normal [input-shape neurons]) 0.1))]
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

(defn sigmoid
  []
  {:foreword (fn [z] [(act/sigmoid z) nil])
   :backward act/d-sigmoid})

(defn softmax
  []
  {:foreword (fn [z] [(-> z m/rows act/softmax) nil])
   :backward #()})
