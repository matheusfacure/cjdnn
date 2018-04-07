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
