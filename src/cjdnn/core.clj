(ns cjdnn.core
  (:require
    [cjdnn.reader :as reader]
    [cjdnn.activations :as act]
    [cjdnn.layers :as layers]
    [cjdnn.loss :as loss]
    [clojure.core.matrix :as m]
    [clojure.core.matrix.random :as rnd]))

(defn model
  [& layers]
  (let [foreword #(->> layers
                       (map :foreword)                      ; get foreword fn for each layer
                       (reduce (fn [data-seq pass-foreword] ; make seq '([z0 cache0] ... [z_n cache_n])
                                 (let [z (-> data-seq first first)]
                                   (cons (pass-foreword z) data-seq))) ; one foreword pass iteration
                               (list [% nil])))             ; start iterating with net input and no cache

        backward (fn [y-true y-hat]
                   (let [loss-backward (-> layers last :backward)
                         d0 (loss-backward y-true y-hat)]
                     (->> layers
                          reverse
                          rest
                          (map :backward)
                          (reduce (fn [deriv-seq pass-backward]
                                    (let [d (-> deriv-seq first)]
                                      (cons (pass-backward d) deriv-seq)))
                                  (list d0)))))

        update! #()]
    {:foreword foreword
     :backward backward
     :update!  update!}))


(def data (reader/read-digits "mnist_dev.csv"))
(def X (:x (reader/get-batch data 20)))
(def y (-> data
           (reader/get-batch 20)
           :y
           (reader/one-hot-encode 10)))

(def my-model (model (layers/linear 784 30)
                     (layers/relu)
                     (layers/linear 30 20)
                     (layers/relu)
                     (layers/linear 20 10)
                     (loss/softmax-cross-entropy)))


(def layers (list (layers/linear 784 30)
                  (layers/relu)
                  (layers/linear 30 20)
                  (layers/relu)
                  (layers/linear 20 10)
                  (loss/softmax-cross-entropy)))

(def backward-cache ((:foreword my-model) X))

(def y-hat (-> backward-cache first first))

(def derivatives ((:backward my-model) y y-hat))

(defn -main
  [& args]
  nil)

