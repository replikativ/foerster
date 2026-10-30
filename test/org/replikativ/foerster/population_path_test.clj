(ns org.replikativ.foerster.population-path-test
  (:require [clojure.test :refer [deftest is testing]]
            [is.simm.partial-cps.async :as pcps]
            [is.simm.partial-cps.runtime :as cps-runtime]
            [org.replikativ.foerster.execution :as execution]
            [org.replikativ.foerster.population.path :as path]
            [org.replikativ.foerster.weights :as weights]))

(deftest retention-conserves-parent-mass
  (let [population {:particles [:a :b] :mass (weights/uniform 2)}
        result (path/retain population [[:a1] [:b1 :b2 :b3]] false)]
    (is (= [:a1 :b1 :b2 :b3] (:particles result)))
    (is (= [0.5 0.16666666666666669 0.16666666666666669 0.16666666666666669]
           (mapv #(Math/exp %) (get-in result [:mass :log-weights]))))
    (is (= 0.0 (get-in result [:mass :log-mass]))))
  (testing "Waste-free retention keeps initial states and canonical uniform mass"
    (let [population {:particles [:a :b] :mass (weights/uniform 2)}
          result (path/retain population [[:a1] [:b1]] true)]
      (is (= [:a :a1 :b :b1] (:particles result)))
      (is (= (weights/uniform 4) (:mass result))))))

(defn- synthetic-run [stages waste-free]
  (path/run {:particles [0 0 0 0] :mass (weights/uniform 4) :position 0}
            {:done? #(= stages %) :advance (fn [ps _ p] {:pos (inc p) :log-incr (vec (repeat (count ps) 0.25))})
             :exhausted #(ex-info "exhausted" {:position %})}
            {:move (fn [starts _ _ moves]
                     (pcps/async {:chains (mapv #(vec (take moves (iterate inc (inc %)))) starts)}))}
            {:n 4 :waste-free waste-free :seed 42 :keys {:resample ::selection} :max-steps (inc stages)}))

(deftest shared-path-synchronous-and-nested-trampoline
  (doseq [waste-free [nil 2]]
    (let [result (path/run-sync (synthetic-run 2000 waste-free))]
      (is (= 4 (count (:particles result))))
      (is (= 2000 (:steps result)))
      (is (< (Math/abs (- 500.0 (get-in result [:mass :log-mass]))) 1e-10))))
  (is (= 3 (:steps (path/run-sync (pcps/async (path/run-sync (synthetic-run 3 nil))))))))

(deftest asynchronous-backend-is-explicit
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"suspended"
                        (path/run-sync (fn [_ _] nil)))))

(deftest parallel-fails-fast-and-preserves-order
  (let [started (atom []) error (ex-info "start failed" {})]
    (is (identical? error
                    (try (path/run-sync (execution/parallel nil 4
                                                            (fn [i resolve _]
                                                              (swap! started conj i)
                                                              (if (= 1 i) (throw error) (resolve i)))))
                         (catch Throwable e e))))
    (is (= [0 1] @started)))
  (is (= [0 1 2] (path/run-sync (execution/parallel nil 3
                                                    (fn [i resolve reject]
                                                      ((pcps/async i) resolve reject)))))))

(deftest suspended-phases-resume-the-same-path
  (let [pending (atom nil)
        result (atom nil)
        error (atom nil)
        drive (fn [operation]
                (loop [t (operation)]
                  (when (cps-runtime/thunk? t) (recur (cps-runtime/force-thunk t)))))
        operation (path/run
                   {:particles [0 1] :mass (weights/uniform 2) :position 0}
                   {:done? #(= 2 %) :advance (fn [_ _ p] {:pos (inc p) :log-incr [0.5 0.5]})
                    :exhausted #(ex-info "exhausted" {:position %})}
                   {:move (fn [starts _ _ _]
                            (fn [resolve _]
                              (reset! pending #(resolve {:chains (mapv (fn [x] [(inc x)]) starts)}))))}
                   {:seed 7 :keys {:resample ::selection}})]
    (drive #(operation (partial reset! result) (partial reset! error)))
    (is (nil? @result))
    (drive @pending)
    (is (nil? @result))
    (drive @pending)
    (is (nil? @error))
    (is (= 2 (:steps @result)))
    (is (= 1.0 (get-in @result [:mass :log-mass])))))
