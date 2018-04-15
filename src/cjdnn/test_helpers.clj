(ns cjdnn.test-helpers
  (:require [clojure.core.matrix :as m]))

(defn absolute-difference
  [x y]
  (m/abs (m/sub x y)))

(defn close?
  [x y tolerance]
  (let [diff (m/lt (absolute-difference x y) tolerance)
        [nrow ncol] (m/shape diff)
        compare-to (m/add (m/zero-matrix nrow ncol) 1)]

    (m/equals diff compare-to)))

(defn get-derivative
  [f]
  (let [e 1e-5]
    #(m/div (m/sub (f (m/add % e))
                   (f (m/sub % e)))
            (m/mul e 2))))
