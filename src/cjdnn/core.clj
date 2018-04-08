(ns cjdnn.core
  (:require
    [cjdnn.reader :as reader]
    [cjdnn.activations :as act]
    [cjdnn.layers :as layers]
    [clojure.core.reducers :as r]
    [clojure.core.matrix :as m]
    [clojure.core.matrix.random :as rnd]))

(defn model
  [& layers]
  #(->> layers
       (map :foreword)
       (reduce (fn [data-seq pass-foreword]
                 (let [z (-> data-seq first first)]
                   (cons (pass-foreword z) data-seq)))
               (list [% nil]))))


(def data (reader/read-digits "mnist_dev.csv"))
(def X (:x (reader/get-batch data 100)))

(def my-model (model (layers/dense 784 30)
                     (layers/dense 30 20)
                     (layers/dense 20 10)
                     (layers/dense 10 1)))

(def ls (map :foreword my-layers))

(def result (reduce (fn [data-seq foreword-fn]
                      (let [z (-> data-seq first first)]
                        (cons (foreword-fn z) data-seq)))
                    (list [X nil])
                    ls))




(defn -main
  [& args]
  nil)

