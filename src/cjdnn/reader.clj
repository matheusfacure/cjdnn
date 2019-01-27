(ns cjdnn.reader
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.core.matrix :as m]))

(defn read-digits
  [path]
  (->> path
       io/resource
       slurp
       csv/read-csv
       (map #(map read-string %))
       (map (fn [line] {:y (first line)
                        :x (rest line)}))))

(defn get-batch
  [data-buffer batch-size]
  (let [batch-data (take batch-size data-buffer)]
    {:y (m/matrix (map :y batch-data))
     :x (m/matrix (map :x batch-data))}))

(defn one-hot-encode
  [y n-class]
  (let [empty-matrix (m/zero-matrix (count y) n-class)
        get-oh #(m/mset %1 %2 (float %2))]
    (map get-oh (m/rows empty-matrix) y)))
