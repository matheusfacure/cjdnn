(ns cjdnn.core
  (:require
    [cjdnn.reader :as reader]
    [cjdnn.activations :as act]
    [cjdnn.layers :as layers]
    [clojure.core.matrix :as m]
    [clojure.core.matrix.random :as rnd]))

(defn model
  [& layers]
  (apply comp (reverse layers)))


(def data (reader/read-digits "mnist_dev.csv"))
(def X (:x (reader/get-batch data 100)))
(def my-model (model (layers/dense 784 128)
                     (layers/dense 128 32)
                     (layers/dense 32 10)))

(defn -main
  [& args]
  nil)

