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
  (let [foreword #(->> layers
                       (map :foreword)                      ; get foreword fn for each layer
                       (reduce (fn [data-seq pass-foreword] ; make seq '([z0 cache0] ... [z_n cache_n])
                                 (let [z (-> data-seq first first)]
                                   (print "\n" (m/shape z))
                                   (print (m/shape (pass-foreword z)))
                                   (cons (pass-foreword z) data-seq))) ; one foreword pass iteration
                               (list [% nil])))             ; start iterating with net input and no cache
        backward #()
        update! #()]
    {:foreword foreword
     :backward backward
     :update!  update!}))


(def data (reader/read-digits "mnist_dev.csv"))
(def X (:x (reader/get-batch data 100)))

(def my-model (model (layers/linear 784 30)
                     (layers/relu)
                     (layers/linear 30 20)
                     (layers/relu)
                     (layers/linear 20 10)))

(m/shape ((:foreword my-model) X))


(defn -main
  [& args]
  nil)

